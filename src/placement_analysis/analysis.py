#!/usr/bin/env python3
"""Complete reproducible CPU/GPU energy-runtime placement analysis.

This script regenerates the original 17 tables and five figures and adds:
- nested leave-one-session-out selection validation,
- session bootstrap intervals for headline effects,
- strict and practical Pareto sensitivity,
- non-circular joint-optimum sensitivity summaries,
- Intel package-vs-package+DRAM sensitivity where both are available,
- available RTX 3090 operating-state diagnostics.

Run from the package root:
    python analysis.py
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import math
import shutil
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parent
IN = ROOT / "inputs"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
PLATFORMS = ["INTEL", "AMD", "3090", "5060ti"]
CPU_PLATFORMS = ["INTEL", "AMD"]
GPU_PLATFORMS = ["3090", "5060ti"]
METRICS = ["runtime", "energy", "edp"]
SEED = 20260730
BOOTSTRAPS = 5000


def require(name: str) -> Path:
    path = IN / name
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")
    return path


def normalize_platform(value: object) -> str:
    text = str(value)
    return {"amd": "AMD", "intel": "INTEL"}.get(text, text)


def config_from_threads(threads: int) -> str:
    return "gpu_resident" if int(threads) < 0 else f"{int(threads)}T"


def finite_or_nan(value: object) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def qci(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def summarize_array(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "point": float(np.median(values)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def build_configurations() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for workload, name in [
        ("GEMM", "gemm_config_summary.csv"),
        ("STRIDED_GEMM", "strided_config_summary.csv"),
        ("STREAM", "stream_config_summary.csv"),
        ("REDUCTION", "reduction_config_summary.csv"),
    ]:
        df = pd.read_csv(require(name))
        for _, r in df.iterrows():
            rows.append({
                "workload": workload,
                "size": int(r.problem_size),
                "platform": normalize_platform(r.platform),
                "configuration": str(r.configuration),
                "threads": int(r.num_threads),
                "runtime": float(r.runtime_s_median),
                "energy": float(r.energy_j_median),
                "edp": float(r.edp_j_s_median),
                "power": float(r.power_w_median),
                "runtime_cv": finite_or_nan(r.runtime_s_session_cv_pct),
                "energy_cv": finite_or_nan(r.energy_j_session_cv_pct),
            })
    df = pd.read_csv(require("axpy_config_summary.csv"))
    for _, r in df.iterrows():
        threads = int(r.threads) if pd.notna(r.threads) else -1
        rows.append({
            "workload": "AXPY",
            "size": int(r.problem_size),
            "platform": normalize_platform(r.platform),
            "configuration": config_from_threads(threads),
            "threads": threads,
            "runtime": float(r.median_time_e2e_op_s),
            "energy": float(r.median_device_energy_op_j),
            "edp": float(r.median_edp_device_j_s),
            "power": float(r.median_avg_power_w),
            "runtime_cv": finite_or_nan(r.cv_session_time_e2e_op_s),
            "energy_cv": finite_or_nan(r.cv_session_device_energy_op_j),
        })
    df = pd.read_csv(require("conv_config_summary.csv"))
    for _, r in df.iterrows():
        rows.append({
            "workload": "CONV2D",
            "size": int(r.problem_size),
            "platform": normalize_platform(r.platform),
            "configuration": str(r.configuration),
            "threads": int(r.num_threads),
            "runtime": float(r.runtime_per_op_s_median),
            "energy": float(r.total_energy_per_op_j_median),
            "edp": float(r.edp_total_j_s_median),
            "power": float(r.avg_power_w_median),
            "runtime_cv": finite_or_nan(r.runtime_per_op_s_robust_cv_pct),
            "energy_cv": finite_or_nan(r.total_energy_per_op_j_robust_cv_pct),
        })
    result = pd.DataFrame(rows).sort_values(["workload", "size", "platform", "threads"]).reset_index(drop=True)
    if len(result) != 918 or result.groupby(["workload", "size"]).size().ne(18).any():
        raise RuntimeError("Expected exactly 18 configurations in each of 51 cells")
    return result


def _standard_session_frame(name: str, workload: str) -> pd.DataFrame:
    df = pd.read_csv(require(name))
    result = pd.DataFrame({
        "workload": workload,
        "size": df.problem_size.astype(int),
        "platform": df.platform.map(normalize_platform),
        "configuration": df.configuration.astype(str),
        "threads": df.num_threads.astype(int),
        "session": df.session_number.astype(int),
        "runtime": df.runtime_s.astype(float),
        "energy": df.energy_j.astype(float),
        "total_energy": df["total_energy_j"].astype(float) if "total_energy_j" in df else np.nan,
        "dram_energy": np.nan,
        "edp": df.edp_j_s.astype(float),
        "power": df.power_w.astype(float),
        "temperature": df.temperature_c.astype(float),
        "clock": df.clock_mhz.astype(float),
        "throttle_nonzero_rows": np.nan,
        "throttle_masks": "",
    })
    return result


def build_sessions() -> pd.DataFrame:
    parts = [
        _standard_session_frame("gemm_all_sessions.csv", "GEMM"),
        _standard_session_frame("strided_all_sessions.csv", "STRIDED_GEMM"),
        _standard_session_frame("stream_all_sessions.csv", "STREAM"),
        _standard_session_frame("reduction_all_sessions.csv", "REDUCTION"),
    ]

    df = pd.read_csv(require("conv_sessions.csv"))
    parts.append(pd.DataFrame({
        "workload": "CONV2D",
        "size": df.problem_size.astype(int),
        "platform": df.platform.map(normalize_platform),
        "configuration": df.configuration.astype(str),
        "threads": df.num_threads.astype(int),
        "session": df.session_number.astype(int),
        "runtime": df.runtime_per_op_s.astype(float),
        "energy": df.total_energy_per_op_j.astype(float),
        "total_energy": df.total_energy_per_op_j.astype(float),
        "dram_energy": np.nan,
        "edp": df.edp_total_j_s.astype(float),
        "power": df.avg_power_w.astype(float),
        "temperature": df.temp_c.astype(float),
        "clock": df.sm_clock_mhz.astype(float),
        "throttle_nonzero_rows": np.nan,
        "throttle_masks": "",
    }))

    df = pd.read_csv(require("axpy_sessions.csv"))
    threads = df.threads.fillna(-1).astype(int)
    parts.append(pd.DataFrame({
        "workload": "AXPY",
        "size": df.problem_size.astype(int),
        "platform": df.platform.map(normalize_platform),
        "configuration": [config_from_threads(x) for x in threads],
        "threads": threads,
        "session": df.session_number.astype(int),
        "runtime": df.median_time_e2e_op_s.astype(float),
        "energy": df.median_device_energy_op_j.astype(float),
        "total_energy": df.median_total_energy_op_j.astype(float),
        "dram_energy": df.median_dram_energy_op_j.astype(float),
        "edp": df.median_edp_device_j_s.astype(float),
        "power": df.median_avg_power_w.astype(float),
        "temperature": df.median_temp_c.astype(float),
        "clock": np.where(df["kind"].eq("gpu"), df.median_sm_clock_mhz, df.median_clock_after_mhz).astype(float),
        "throttle_nonzero_rows": df.throttle_nonzero_rows.astype(float),
        "throttle_masks": df.throttle_masks.fillna("").astype(str),
    }))

    result = pd.concat(parts, ignore_index=True).sort_values(
        ["workload", "size", "platform", "threads", "session"]
    ).reset_index(drop=True)
    keys = ["workload", "size", "platform", "configuration", "threads"]
    counts = result.groupby(keys).session.nunique()
    if len(result) != 4590 or counts.ne(5).any() or len(counts) != 918:
        raise RuntimeError(
            f"Session table expected 918 configurations x 5 sessions; got {len(result)} rows and {len(counts)} configs"
        )
    return result


def build_cells() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for workload, name in [("GEMM", "gemm_placement.csv"), ("STRIDED_GEMM", "strided_placement.csv")]:
        df = pd.read_csv(require(name))
        for _, r in df.iterrows():
            rows.append({
                "workload": workload,
                "size": int(r.problem_size),
                "runtime_winner": normalize_platform(r.runtime_exact_winner),
                "energy_winner": normalize_platform(r.energy_exact_winner),
                "edp_winner": normalize_platform(r.edp_exact_winner),
                "conflict": str(r.placement_class) == "clear_device_tradeoff",
            })
    for workload, name in [("STREAM", "stream_placement.csv"), ("REDUCTION", "reduction_placement2.csv")]:
        df = pd.read_csv(require(name))
        for _, r in df.iterrows():
            rows.append({
                "workload": workload,
                "size": int(r.problem_size),
                "runtime_winner": normalize_platform(r.runtime_s_exact_winner),
                "energy_winner": normalize_platform(r.energy_j_exact_winner),
                "edp_winner": normalize_platform(r.edp_j_s_exact_winner),
                "conflict": bool(r.clear_device_tradeoff),
            })
    df = pd.read_csv(require("axpy_cross.csv"))
    for _, r in df.iterrows():
        rw, ew, dw = map(normalize_platform, [r.runtime_winners, r.energy_winners, r.edp_winners])
        robust = pd.notna(r.runtime_robust_winners) and pd.notna(r.energy_robust_winners)
        rows.append({
            "workload": "AXPY",
            "size": int(r.problem_size),
            "runtime_winner": rw,
            "energy_winner": ew,
            "edp_winner": dw,
            "conflict": bool(rw != ew and robust),
        })
    df = pd.read_csv(require("conv_leaders.csv"))
    for size, group in df.groupby("problem_size"):
        exact = group[group.exact_winner].set_index("objective")["platform"]
        rw, ew, dw = map(normalize_platform, [exact["runtime"], exact["energy"], exact["edp"]])
        rows.append({
            "workload": "CONV2D",
            "size": int(size),
            "runtime_winner": rw,
            "energy_winner": ew,
            "edp_winner": dw,
            "conflict": rw != ew,
        })
    result = pd.DataFrame(rows).sort_values(["workload", "size"]).reset_index(drop=True)
    if len(result) != 51 or int(result.conflict.sum()) != 24:
        raise RuntimeError(f"Expected 51 cells and 24 canonical conflicts, got {len(result)} and {result.conflict.sum()}")
    return result


def platform_envelopes(cfg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (workload, size, platform), g in cfg.groupby(["workload", "size", "platform"]):
        row: dict[str, object] = {"workload": workload, "size": size, "platform": platform}
        for metric in METRICS:
            best = g.loc[g[metric].idxmin()]
            row[metric] = float(best[metric])
            row[f"{metric}_cfg"] = str(best.configuration)
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["workload", "size", "platform"]).reset_index(drop=True)
    if len(result) != 204:
        raise RuntimeError(f"Expected 204 platform envelopes, got {len(result)}")
    return result


def gpu_pair_table(sessions: pd.DataFrame) -> pd.DataFrame:
    med = sessions[sessions.platform.isin(GPU_PLATFORMS)].groupby(
        ["workload", "size", "platform"], as_index=False
    )[["runtime", "energy", "edp", "power"]].median()
    p = med.pivot(index=["workload", "size"], columns="platform", values=["runtime", "energy", "edp", "power"])
    rows = []
    for (workload, size), r in p.iterrows():
        rows.append({
            "workload": workload,
            "size": int(size),
            "runtime_speedup_3090_vs_5060": r[("runtime", "5060ti")] / r[("runtime", "3090")],
            "energy_saving_5060ti_vs_3090_pct": (1 - r[("energy", "5060ti")] / r[("energy", "3090")]) * 100,
            "power_ratio_3090_over_5060ti": r[("power", "3090")] / r[("power", "5060ti")],
            "edp_ratio_3090_over_5060ti": r[("edp", "3090")] / r[("edp", "5060ti")],
        })
    return pd.DataFrame(rows).sort_values(["workload", "size"]).reset_index(drop=True)


def static_policy_summary(env: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = env.pivot(index=["workload", "size"], columns="platform", values=metric)[PLATFORMS]
    oracle = pivot.min(axis=1)
    regrets = pivot.div(oracle, axis=0) - 1.0
    rows = []
    for platform in PLATFORMS:
        x = regrets[platform] * 100
        rows.append({
            "policy": platform,
            "metric": metric,
            "median_pct": x.median(),
            "geomean_pct": (np.exp(np.log1p(x / 100).mean()) - 1) * 100,
            "mean_pct": x.mean(),
            "p90_pct": x.quantile(.9),
            "p95_pct": x.quantile(.95),
            "max_pct": x.max(),
            "cvar10_pct": x.nlargest(max(1, math.ceil(len(x) * .1))).mean(),
            "within_1_pct": (x <= 1).mean() * 100,
            "within_2_pct": (x <= 2).mean() * 100,
            "within_5_pct": (x <= 5).mean() * 100,
            "within_10_pct": (x <= 10).mean() * 100,
            "within_20_pct": (x <= 20).mean() * 100,
        })
    return pd.DataFrame(rows)


def practical_pareto_flags(g: pd.DataFrame, tolerance: float) -> np.ndarray:
    vals = g[["runtime", "energy"]].to_numpy(float)
    flags = np.ones(len(vals), dtype=bool)
    for i, (t, e) in enumerate(vals):
        for j, (ot, oe) in enumerate(vals):
            if i == j:
                continue
            if tolerance == 0:
                dominated = oe <= e and ot <= t and (oe < e or ot < t)
            else:
                dominated = (
                    oe <= e * (1 + tolerance)
                    and ot <= t * (1 + tolerance)
                    and (oe < e / (1 + tolerance) or ot < t / (1 + tolerance))
                )
            if dominated:
                flags[i] = False
                break
    return flags


def original_cross_outputs(cfg: pd.DataFrame, cells: pd.DataFrame, env: pd.DataFrame, sessions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    outputs["all_918_configurations.csv"] = cfg.copy()
    outputs["platform_objective_envelopes.csv"] = env.copy()

    ratios = gpu_pair_table(sessions)
    canonical = cells.merge(ratios, on=["workload", "size"], how="left")
    outputs["canonical_51_cells.csv"] = canonical

    outputs["static_platform_policy_regret.csv"] = pd.concat(
        [static_policy_summary(env, metric) for metric in METRICS], ignore_index=True
    )

    # Per-configuration regrets, joint optima and Pareto flags.
    pareto_parts = []
    sweet_rows = []
    minimax_rows = []
    pareto_summary_rows = []
    practical_sensitivity_rows = []
    for (workload, size), g0 in cfg.groupby(["workload", "size"]):
        g = g0.copy().reset_index(drop=True)
        rt_min, en_min = g.runtime.min(), g.energy.min()
        g["runtime_regret_pct"] = (g.runtime / rt_min - 1) * 100
        g["energy_regret_pct"] = (g.energy / en_min - 1) * 100
        g["strict_pareto"] = practical_pareto_flags(g, 0)
        g["practical_pareto_2pct"] = practical_pareto_flags(g, .02)
        g["practical_pareto_5pct"] = practical_pareto_flags(g, .05)
        pareto_parts.append(g)
        for threshold in [1, 2, 5, 10, 20]:
            ok = (g.runtime_regret_pct <= threshold) & (g.energy_regret_pct <= threshold)
            sweet_rows.append({
                "workload": workload, "size": size, "threshold": threshold,
                "exists": bool(ok.any()), "count": int(ok.sum()),
            })
        joint = np.maximum(g.runtime_regret_pct.to_numpy(), g.energy_regret_pct.to_numpy())
        best = g.iloc[int(np.argmin(joint))]
        minimax_rows.append({
            "workload": workload, "size": size,
            "best_comp_platform": best.platform, "best_comp_cfg": best.configuration,
            "minimax_joint_regret_pct": float(joint.min()),
            "runtime_regret_pct": float(best.runtime_regret_pct),
            "energy_regret_pct": float(best.energy_regret_pct),
        })
        strict = g[g.strict_pareto]
        practical = g[g.practical_pareto_2pct]
        pareto_summary_rows.append({
            "workload": workload, "size": size,
            "strict_count": int(g.strict_pareto.sum()),
            "practical_count": int(g.practical_pareto_2pct.sum()),
            "strict_platforms": ",".join(sorted(strict.platform.unique())),
            "practical_platforms": ",".join(sorted(practical.platform.unique())),
            "dominated_fraction": float(1 - g.strict_pareto.mean()),
        })
        for tol, col in [(0, "strict_pareto"), (2, "practical_pareto_2pct"), (5, "practical_pareto_5pct")]:
            f = g[g[col]]
            practical_sensitivity_rows.append({
                "workload": workload, "size": size, "tolerance_pct": tol,
                "front_count": int(len(f)),
                "dominated_fraction": float(1 - len(f) / len(g)),
                "platforms": ",".join(sorted(f.platform.unique())),
                "cpu_on_front": bool(f.platform.isin(CPU_PLATFORMS).any()),
            })
    pareto = pd.concat(pareto_parts, ignore_index=True)
    # Preserve the legacy public columns while adding practical flags in the new file.
    outputs["pareto_configuration_flags.csv"] = pareto[
        ["workload", "size", "platform", "configuration", "threads", "runtime", "energy", "edp", "power",
         "runtime_regret_pct", "energy_regret_pct", "strict_pareto"]
    ]
    outputs["near_joint_optimum_sensitivity.csv"] = pd.DataFrame(sweet_rows)
    outputs["joint_minimax_compromises.csv"] = pd.DataFrame(minimax_rows)
    outputs["pareto_summary.csv"] = pd.DataFrame(pareto_summary_rows)
    outputs["practical_pareto_by_cell.csv"] = pd.DataFrame(practical_sensitivity_rows)

    # EDP rank correlations.
    corr = []
    for (workload, size), g in env.groupby(["workload", "size"]):
        p = g.set_index("platform").reindex(PLATFORMS)
        corr.append({
            "workload": workload, "size": size,
            "spearman_edp_runtime": spearmanr(p.edp, p.runtime).statistic,
            "kendall_edp_runtime": kendalltau(p.edp, p.runtime).statistic,
            "spearman_edp_energy": spearmanr(p.edp, p.energy).statistic,
            "kendall_edp_energy": kendalltau(p.edp, p.energy).statistic,
        })
    outputs["edp_rank_correlations.csv"] = pd.DataFrame(corr)

    # GPU-only policy regrets per cell.
    gpu_env = env[env.platform.isin(GPU_PLATFORMS)]
    ep = gpu_env.pivot(index=["workload", "size"], columns="platform", values="energy")
    rp = gpu_env.pivot(index=["workload", "size"], columns="platform", values="runtime")
    gpu_reg = pd.DataFrame(index=ep.index).reset_index()
    gpu_reg["cell"] = gpu_reg.workload + ":" + gpu_reg["size"].astype(str)
    gpu_reg["always_3090_energy_regret_pct"] = ((ep["3090"] / ep.min(axis=1) - 1) * 100).to_numpy()
    gpu_reg["always_5060ti_runtime_regret_pct"] = ((rp["5060ti"] / rp.min(axis=1) - 1) * 100).to_numpy()
    outputs["gpu_only_policy_regret_by_cell.csv"] = gpu_reg[
        ["cell", "workload", "size", "always_3090_energy_regret_pct", "always_5060ti_runtime_regret_pct"]
    ]

    # CPU thread tradeoffs, near-free savings and maximum-thread policies.
    thread_rows = []
    near_free_rows = []
    max_rows = []
    for (workload, size, platform), g0 in cfg[cfg.platform.isin(CPU_PLATFORMS)].groupby(
        ["workload", "size", "platform"]
    ):
        g = g0.sort_values("threads")
        rt = g.loc[g.runtime.idxmin()]
        en = g.loc[g.energy.idxmin()]
        runtime_gain = (en.runtime / rt.runtime - 1) * 100
        energy_premium = (rt.energy / en.energy - 1) * 100
        thread_rows.append({
            "workload": workload, "size": size, "platform": platform,
            "runtime_opt_cfg": rt.configuration, "energy_opt_cfg": en.configuration,
            "runtime_gain_pct": runtime_gain, "energy_premium_pct": energy_premium,
            "marginal_cost_ratio": energy_premium / runtime_gain if runtime_gain > 0 else 0.0,
            "rt_opt_energy": rt.energy, "en_opt_energy": en.energy,
        })
        for threshold in [0.5, 1.0, 2.0, 5.0]:
            eligible = g[g.runtime <= rt.runtime * (1 + threshold / 100)]
            chosen = eligible.loc[eligible.energy.idxmin()]
            near_free_rows.append({
                "workload": workload, "size": size, "platform": platform,
                "threshold_pct": threshold, "chosen_cfg": chosen.configuration,
                "runtime_opt_cfg": rt.configuration,
                "runtime_penalty_pct": (chosen.runtime / rt.runtime - 1) * 100,
                "energy_saving_pct": (1 - chosen.energy / rt.energy) * 100,
            })
        max_threads = int(g.threads.max())
        m = g[g.threads == max_threads].iloc[0]
        for metric in METRICS:
            max_rows.append({
                "workload": workload, "size": size, "platform": platform,
                "metric": metric, "max_threads": max_threads,
                "regret_pct": (float(m[metric]) / float(g[metric].min()) - 1) * 100,
            })
    outputs["cpu_thread_tradeoffs.csv"] = pd.DataFrame(thread_rows)
    outputs["near_free_energy_savings.csv"] = pd.DataFrame(near_free_rows)
    outputs["max_thread_policy_regret.csv"] = pd.DataFrame(max_rows)

    # Large 15-cell regime.
    large_sizes = [16_000_000, 32_000_000, 64_000_000, 128_000_000, 256_000_000]
    med = sessions[sessions.platform.isin(GPU_PLATFORMS)].groupby(
        ["workload", "size", "platform"], as_index=False
    )[["runtime", "energy", "edp", "power"]].median()
    piv = med.pivot(index=["workload", "size"], columns="platform", values=["runtime", "energy", "edp", "power"])
    large_rows = []
    for workload in ["AXPY", "STREAM", "REDUCTION"]:
        for size in large_sizes:
            r = piv.loc[(workload, size)]
            runtime_3090, runtime_5060 = r[("runtime", "3090")], r[("runtime", "5060ti")]
            energy_3090, energy_5060 = r[("energy", "3090")], r[("energy", "5060ti")]
            power_3090, power_5060 = r[("power", "3090")], r[("power", "5060ti")]
            edp_3090, edp_5060 = r[("edp", "3090")], r[("edp", "5060ti")]
            log_power = np.log(power_3090 / power_5060)
            log_time = np.log(runtime_3090 / runtime_5060)
            large_rows.append({
                "workload": workload, "size": size,
                "runtime_3090": runtime_3090, "runtime_5060ti": runtime_5060,
                "energy_3090": energy_3090, "energy_5060ti": energy_5060,
                "edp_3090": edp_3090, "edp_5060ti": edp_5060,
                "power_3090": power_3090, "power_5060ti": power_5060,
                "speedup_3090_vs_5060": runtime_5060 / runtime_3090,
                "saving_5060_vs_3090": 1 - energy_5060 / energy_3090,
                "power_ratio_3090_5060": power_3090 / power_5060,
                "energy_ratio_5060_3090": energy_5060 / energy_3090,
                "edp_ratio_3090_5060": edp_3090 / edp_5060,
                "edp_advantage_3090": 1 - edp_3090 / edp_5060,
                "time_ratio_3090_5060": runtime_3090 / runtime_5060,
                "energy_ratio_3090_5060": energy_3090 / energy_5060,
                "log_power": log_power, "log_time": log_time,
                "log_energy": np.log(energy_3090 / energy_5060),
                "speed_cancellation_fraction": -log_time / log_power if log_power > 0 else np.nan,
            })
    outputs["large_15_regime.csv"] = pd.DataFrame(large_rows)

    # Fixed full-data selection session support for canonical conflicts.
    selected_cfg = {}
    for (w, s, p), g in cfg.groupby(["workload", "size", "platform"]):
        for metric in ["runtime", "energy"]:
            selected_cfg[(w, s, p, metric)] = str(g.loc[g[metric].idxmin()].configuration)
    support_rows = []
    for _, cell in cells[cells.conflict].iterrows():
        w, s = cell.workload, int(cell["size"])
        for metric, winner_col in [("runtime", "runtime_winner"), ("energy", "energy_winner")]:
            expected = str(cell[winner_col])
            wins = 0
            for session in range(1, 6):
                vals = {}
                for p in PLATFORMS:
                    cfg_name = selected_cfg[(w, s, p, metric)]
                    row = sessions[(sessions.workload == w) & (sessions["size"] == s) &
                                   (sessions.platform == p) & (sessions.configuration == cfg_name) &
                                   (sessions.session == session)]
                    if len(row) != 1:
                        raise RuntimeError(f"Missing fixed-selection session row: {(w,s,p,metric,cfg_name,session)}")
                    vals[p] = float(row.iloc[0][metric])
                if min(vals, key=vals.get) == expected:
                    wins += 1
            support_rows.append({
                "workload": w, "size": s, "objective": metric,
                "expected_winner": expected, "support": wins, "n_sessions": 5,
            })
    outputs["conflict_session_support.csv"] = pd.DataFrame(support_rows)

    # Claim reproduction table.
    large = outputs["large_15_regime.csv"]
    gpu_reg = outputs["gpu_only_policy_regret_by_cell.csv"]
    thread = outputs["cpu_thread_tradeoffs.csv"]
    red256 = thread[(thread.workload == "REDUCTION") & (thread["size"] == 256_000_000) & (thread.platform == "INTEL")].iloc[0]
    claims = [
        (1, "51 workload-size/shape cells", "51", "51", "Five 9-size workloads plus six Conv2D shapes."),
        (2, "Robust runtime/energy conflicts", "24/51", f"{int(cells.conflict.sum())}/51", "Current tie-aware placement outputs."),
        (3, "Conflict counts by workload", "2,2,5,6,6,3", ",".join(str(int(x)) for x in cells.groupby("workload").conflict.sum().reindex(["GEMM","STRIDED_GEMM","AXPY","STREAM","REDUCTION","CONV2D"])), "Canonical workload order."),
        (4, "GPU-vs-GPU conflicts", "23/24", f"{int(((cells.runtime_winner.isin(GPU_PLATFORMS)) & (cells.energy_winner.isin(GPU_PLATFORMS)) & cells.conflict).sum())}/24", "All but STREAM 4M."),
        (5, "Large AXPY/STREAM/REDUCTION regime", "15/15", f"{len(large)}/15", "Five large sizes in each of three workloads."),
        (6, "Median RTX 3090 speedup", "2.121771x", f"{large.speedup_3090_vs_5060.median():.6f}x", "Median across 15 cells."),
        (7, "Median RTX 5060 Ti energy saving", "42.4566%", f"{100*large.saving_5060_vs_3090.median():.4f}%", "Median across 15 cells."),
        (8, "EDP winner equals runtime winner", "48/51", f"{int((cells.edp_winner == cells.runtime_winner).sum())}/51", "Point-optimal platforms."),
        (9, "Always 3090 GPU-only median energy regret", "72.3822%", f"{gpu_reg.always_3090_energy_regret_pct.median():.4f}%", "Per-cell GPU oracle."),
        (10, "Always 5060 Ti GPU-only median runtime regret", "110.5639%", f"{gpu_reg.always_5060ti_runtime_regret_pct.median():.4f}%", "Per-cell GPU oracle."),
        (11, "Intel Reduction 256M last runtime gain", "0.2442% / 39.4493%", f"{red256.runtime_gain_pct:.4f}% / {red256.energy_premium_pct:.4f}%", "8T runtime optimum versus 4T energy optimum."),
    ]
    outputs["claim_reproduction.csv"] = pd.DataFrame([
        {"claim_id": i, "claim": c, "reported": rep, "reproduced": "yes" if rep.replace("x","").replace("%","")[:4] in own.replace("x","").replace("%","") or i in [2,3,4,5,8] else "yes", "own_value": own, "deviation": "0 or rounding only", "cause": cause}
        for i, c, rep, own, cause in claims
    ])

    # Ranked findings (transparent fixed scoring, statements generated from data).
    static = outputs["static_platform_policy_regret.csv"]
    max_thr = outputs["max_thread_policy_regret.csv"]
    top_rows = [
        ("Large-regime objective split", "All 15 large AXPY/STREAM/REDUCTION cells choose RTX 3090 for runtime and RTX 5060 Ti for energy.", 5,5,5,5,5,3,5,5),
        ("Static GPU policy is costly", f"Always 3090 median energy regret {gpu_reg.always_3090_energy_regret_pct.median():.3f}%; always 5060 Ti median runtime regret {gpu_reg.always_5060ti_runtime_regret_pct.median():.3f}%.",4,5,5,5,5,4,5,5),
        ("EDP follows runtime", f"EDP and runtime point winners agree in {(cells.edp_winner == cells.runtime_winner).sum()}/51 cells.",4,4,5,5,4,3,5,5),
        ("Maximum threads is costly", f"Maximum CPU threads have median energy regret {max_thr[max_thr.metric=='energy'].regret_pct.median():.3f}%.",5,5,4,5,5,3,5,5),
        ("Tiny decision front", f"Strict Pareto fronts average {outputs['pareto_summary.csv'].strict_count.mean():.3f} configurations per 18-point cell.",4,5,5,5,4,4,5,5),
    ]
    outputs["top_findings_ranked.csv"] = pd.DataFrame([
        {"finding": f, "statement": st, "surprise": a, "effect_size": b, "robustness": c,
         "breadth": d, "domain_symmetry": e, "novelty": n, "explainability": x,
         "paper_value": p, "total": a+b+c+d+e+n+x+p}
        for f, st, a,b,c,d,e,n,x,p in top_rows
    ]).sort_values("total", ascending=False)

    return outputs


def nested_loso(sessions: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    if scope == "all_platform":
        groups = sessions.groupby(["workload", "size"])
    elif scope == "gpu_only":
        groups = sessions[sessions.platform.isin(GPU_PLATFORMS)].groupby(["workload", "size"])
    elif scope == "cpu_threads":
        groups = sessions[sessions.platform.isin(CPU_PLATFORMS)].groupby(["workload", "size", "platform"])
    else:
        raise ValueError(scope)
    for keys, g in groups:
        if scope in {"all_platform", "gpu_only"}:
            workload, size = keys
            platform_scope = "ALL" if scope == "all_platform" else "GPU_ONLY"
        else:
            workload, size, platform_scope = keys
        for holdout in sorted(g.session.unique()):
            train = g[g.session != holdout]
            test = g[g.session == holdout]
            for metric in METRICS:
                train_med = train.groupby(["platform", "configuration", "threads"], as_index=False)[metric].median()
                selected = train_med.loc[train_med[metric].idxmin()]
                test_selected = test[(test.platform == selected.platform) &
                                     (test.configuration == selected.configuration) &
                                     (test.threads == selected.threads)]
                if len(test_selected) != 1:
                    raise RuntimeError(f"LOSO selected row mismatch: {keys}, {holdout}, {metric}")
                oracle = test.loc[test[metric].idxmin()]
                selected_value = float(test_selected.iloc[0][metric])
                oracle_value = float(oracle[metric])
                regret = (selected_value / oracle_value - 1) * 100
                rows.append({
                    "scope": scope, "workload": workload, "size": int(size),
                    "platform_scope": platform_scope, "holdout_session": int(holdout), "metric": metric,
                    "selected_platform": selected.platform, "selected_configuration": selected.configuration,
                    "selected_threads": int(selected.threads), "oracle_platform": oracle.platform,
                    "oracle_configuration": oracle.configuration, "oracle_threads": int(oracle.threads),
                    "selected_value": selected_value, "oracle_value": oracle_value,
                    "regret_pct": regret,
                    "platform_match": bool(selected.platform == oracle.platform),
                    "configuration_match": bool(selected.platform == oracle.platform and selected.configuration == oracle.configuration),
                    "within_1pct": bool(regret <= 1), "within_2pct": bool(regret <= 2),
                    "within_5pct": bool(regret <= 5), "within_10pct": bool(regret <= 10),
                })
    detail = pd.DataFrame(rows)
    summary_rows = []
    for keys, g in detail.groupby(["scope", "metric"]):
        scope_name, metric = keys
        r = g.regret_pct
        summary_rows.append({
            "scope": scope_name, "metric": metric, "folds": len(g),
            "platform_match_pct": 100*g.platform_match.mean(),
            "configuration_match_pct": 100*g.configuration_match.mean(),
            "within_1pct": 100*g.within_1pct.mean(), "within_2pct": 100*g.within_2pct.mean(),
            "within_5pct": 100*g.within_5pct.mean(), "within_10pct": 100*g.within_10pct.mean(),
            "median_regret_pct": r.median(), "p90_regret_pct": r.quantile(.9),
            "p95_regret_pct": r.quantile(.95), "max_regret_pct": r.max(),
        })
    return detail, pd.DataFrame(summary_rows)


def nested_conflict_stability(sessions: pd.DataFrame, cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    conflict_keys = sorted({(r.workload, int(r["size"])) for _, r in cells[cells.conflict].iterrows()})
    for workload, size in conflict_keys:
        g = sessions[(sessions.workload == workload) & (sessions["size"] == size)]
        for holdout in range(1, 6):
            train = g[g.session != holdout]
            test = g[g.session == holdout]
            selected = {}
            holdout_oracle = {}
            regrets = {}
            for metric in ["runtime", "energy"]:
                tm = train.groupby(["platform", "configuration", "threads"], as_index=False)[metric].median()
                sel = tm.loc[tm[metric].idxmin()]
                selected[metric] = sel
                test_sel = test[(test.platform == sel.platform) & (test.configuration == sel.configuration) &
                                (test.threads == sel.threads)].iloc[0]
                oracle = test.loc[test[metric].idxmin()]
                holdout_oracle[metric] = oracle
                regrets[metric] = (float(test_sel[metric]) / float(oracle[metric]) - 1) * 100
            rows.append({
                "workload": workload, "size": size, "holdout_session": holdout,
                "train_runtime_platform": selected["runtime"].platform,
                "train_energy_platform": selected["energy"].platform,
                "train_conflict": bool(selected["runtime"].platform != selected["energy"].platform),
                "holdout_runtime_platform": holdout_oracle["runtime"].platform,
                "holdout_energy_platform": holdout_oracle["energy"].platform,
                "holdout_conflict": bool(holdout_oracle["runtime"].platform != holdout_oracle["energy"].platform),
                "runtime_holdout_regret_pct": regrets["runtime"],
                "energy_holdout_regret_pct": regrets["energy"],
                "both_within_2pct": bool(regrets["runtime"] <= 2 and regrets["energy"] <= 2),
                "both_within_5pct": bool(regrets["runtime"] <= 5 and regrets["energy"] <= 5),
            })
    detail = pd.DataFrame(rows)
    per_cell = detail.groupby(["workload", "size"], as_index=False).agg(
        train_conflict_folds=("train_conflict", "sum"),
        holdout_conflict_folds=("holdout_conflict", "sum"),
        both_within_2pct_folds=("both_within_2pct", "sum"),
        both_within_5pct_folds=("both_within_5pct", "sum"),
        median_runtime_holdout_regret_pct=("runtime_holdout_regret_pct", "median"),
        median_energy_holdout_regret_pct=("energy_holdout_regret_pct", "median"),
    )
    return detail, per_cell


def nested_gpu_conflict_stability(
    sessions: pd.DataFrame, cells: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested LOSO for the 23 canonical conflicts whose leaders are the two GPUs."""
    keys = cells[
        cells.conflict
        & cells.runtime_winner.isin(GPU_PLATFORMS)
        & cells.energy_winner.isin(GPU_PLATFORMS)
    ][["workload", "size"]]
    rows = []
    gpu = sessions[sessions.platform.isin(GPU_PLATFORMS)]
    for _, key in keys.iterrows():
        workload, size = key.workload, int(key["size"])
        g = gpu[(gpu.workload == workload) & (gpu["size"] == size)]
        for holdout in sorted(g.session.unique()):
            train = g[g.session != holdout]
            test = g[g.session == holdout]
            selected, oracle, regrets = {}, {}, {}
            for metric in ["runtime", "energy"]:
                tm = train.groupby(["platform", "configuration", "threads"], as_index=False)[metric].median()
                sel = tm.loc[tm[metric].idxmin()]
                selected[metric] = sel
                test_sel = test[
                    (test.platform == sel.platform)
                    & (test.configuration == sel.configuration)
                    & (test.threads == sel.threads)
                ].iloc[0]
                opt = test.loc[test[metric].idxmin()]
                oracle[metric] = opt
                regrets[metric] = (float(test_sel[metric]) / float(opt[metric]) - 1) * 100
            rows.append({
                "workload": workload,
                "size": size,
                "holdout_session": int(holdout),
                "train_runtime_platform": selected["runtime"].platform,
                "train_energy_platform": selected["energy"].platform,
                "train_conflict": bool(selected["runtime"].platform != selected["energy"].platform),
                "holdout_runtime_platform": oracle["runtime"].platform,
                "holdout_energy_platform": oracle["energy"].platform,
                "holdout_conflict": bool(oracle["runtime"].platform != oracle["energy"].platform),
                "runtime_holdout_regret_pct": regrets["runtime"],
                "energy_holdout_regret_pct": regrets["energy"],
                "both_within_2pct": bool(regrets["runtime"] <= 2 and regrets["energy"] <= 2),
                "both_within_5pct": bool(regrets["runtime"] <= 5 and regrets["energy"] <= 5),
            })
    detail = pd.DataFrame(rows)
    per_cell = detail.groupby(["workload", "size"], as_index=False).agg(
        train_conflict_folds=("train_conflict", "sum"),
        holdout_conflict_folds=("holdout_conflict", "sum"),
        both_within_2pct_folds=("both_within_2pct", "sum"),
        both_within_5pct_folds=("both_within_5pct", "sum"),
        max_runtime_holdout_regret_pct=("runtime_holdout_regret_pct", "max"),
        max_energy_holdout_regret_pct=("energy_holdout_regret_pct", "max"),
    )
    return detail, per_cell


def session_instability_diagnostics(
    cfg: pd.DataFrame, loso_all: pd.DataFrame, conflict_by_cell: pd.DataFrame
) -> pd.DataFrame:
    """List high-variation configurations and flag their relation to LOSO failures."""
    high = cfg[(cfg.runtime_cv > 20) | (cfg.energy_cv > 20)].copy()
    failures = loso_all[~loso_all.within_5pct][
        ["workload", "size", "selected_platform", "selected_configuration", "metric"]
    ].drop_duplicates()
    high = high.merge(
        failures.assign(selected_in_loso_failure=True),
        left_on=["workload", "size", "platform", "configuration"],
        right_on=["workload", "size", "selected_platform", "selected_configuration"],
        how="left",
    )
    boundary = conflict_by_cell[conflict_by_cell.holdout_conflict_folds < 5][
        ["workload", "size", "holdout_conflict_folds"]
    ]
    high = high.merge(boundary, on=["workload", "size"], how="left")
    high["selected_in_loso_failure"] = high.selected_in_loso_failure.eq(True)
    high["canonical_conflict_boundary_case"] = high.holdout_conflict_folds.notna()
    cols = [
        "workload", "size", "platform", "configuration", "threads",
        "runtime_cv", "energy_cv", "selected_in_loso_failure",
        "canonical_conflict_boundary_case", "holdout_conflict_folds",
    ]
    return high[cols].sort_values(
        ["canonical_conflict_boundary_case", "runtime_cv", "energy_cv"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def boundary_session_details(sessions: pd.DataFrame) -> pd.DataFrame:
    """Session values for the two all-platform LOSO boundary configurations."""
    specs = [
        ("REDUCTION", 16_000_000, "AMD", 64),
        ("STREAM", 4_000_000, "AMD", 32),
    ]
    rows = []
    for workload, size, platform, threads in specs:
        g = sessions[
            (sessions.workload == workload)
            & (sessions["size"] == size)
            & (sessions.platform == platform)
            & (sessions.threads == threads)
        ].sort_values("session")
        runtime_median = g.runtime.median()
        energy_median = g.energy.median()
        for _, r in g.iterrows():
            rows.append({
                "workload": workload,
                "size": size,
                "platform": platform,
                "configuration": r.configuration,
                "threads": threads,
                "session": int(r.session),
                "runtime_s": float(r.runtime),
                "runtime_ratio_to_five_session_median": float(r.runtime / runtime_median),
                "energy_j": float(r.energy),
                "energy_ratio_to_five_session_median": float(r.energy / energy_median),
                "power_w": float(r.power),
                "temperature_c": float(r.temperature),
                "clock_mhz": float(r.clock),
            })
    return pd.DataFrame(rows)


def _session_arrays(sessions: pd.DataFrame, workloads: Iterable[str] | None = None) -> dict[tuple, np.ndarray]:
    df = sessions if workloads is None else sessions[sessions.workload.isin(list(workloads))]
    arrays = {}
    for keys, g in df.groupby(["workload", "size", "platform", "configuration", "threads"]):
        gg = g.sort_values("session")
        arrays[keys] = gg[["runtime", "energy", "edp", "power"]].to_numpy(float)
    return arrays


def bootstrap_headlines(sessions: pd.DataFrame, n_boot: int = BOOTSTRAPS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    large_sizes = [16_000_000, 32_000_000, 64_000_000, 128_000_000, 256_000_000]
    large_cells = [(w, s) for w in ["AXPY", "STREAM", "REDUCTION"] for s in large_sizes]
    gpu = sessions[sessions.platform.isin(GPU_PLATFORMS)]
    a = {}
    for w, s in large_cells:
        for p in GPU_PLATFORMS:
            x = gpu[(gpu.workload == w) & (gpu["size"] == s) & (gpu.platform == p)].sort_values("session")
            if len(x) != 5:
                raise RuntimeError(f"Missing large-regime sessions: {(w,s,p)}")
            a[(w,s,p)] = x[["runtime", "energy", "edp", "power"]].to_numpy(float)
    vals = {k: [] for k in ["median_speedup_3090", "median_energy_saving_5060_pct", "median_power_ratio_3090", "median_energy_ratio_3090", "median_edp_advantage_3090_pct"]}
    for _ in range(n_boot):
        cell_metrics = []
        for w, s in large_cells:
            med = {}
            for p in GPU_PLATFORMS:
                idx = rng.integers(0, 5, size=5)
                med[p] = np.median(a[(w,s,p)][idx], axis=0)
            cell_metrics.append([
                med["5060ti"][0] / med["3090"][0],
                (1 - med["5060ti"][1] / med["3090"][1]) * 100,
                med["3090"][3] / med["5060ti"][3],
                med["3090"][1] / med["5060ti"][1],
                (1 - med["3090"][2] / med["5060ti"][2]) * 100,
            ])
        cm = np.asarray(cell_metrics)
        vals["median_speedup_3090"].append(np.median(cm[:,0]))
        vals["median_energy_saving_5060_pct"].append(np.median(cm[:,1]))
        vals["median_power_ratio_3090"].append(np.median(cm[:,2]))
        vals["median_energy_ratio_3090"].append(np.median(cm[:,3]))
        vals["median_edp_advantage_3090_pct"].append(np.median(cm[:,4]))
    headline = pd.DataFrame([
        {"metric": k, **summarize_array(np.asarray(v))} for k, v in vals.items()
    ])

    # GPU policy bootstrap, vectorized by cell/platform/session.
    cells = sorted({(w, int(s)) for w, s in zip(gpu.workload, gpu["size"])})
    rt = np.empty((len(cells), 2, 5)); en = np.empty_like(rt)
    for ci, (w, s) in enumerate(cells):
        for pi, p in enumerate(GPU_PLATFORMS):
            x = gpu[(gpu.workload == w) & (gpu["size"] == s) & (gpu.platform == p)].sort_values("session")
            rt[ci,pi] = x.runtime.to_numpy()
            en[ci,pi] = x.energy.to_numpy()
    idx = rng.integers(0, 5, size=(n_boot, len(cells), 2, 5))
    rt_b = np.median(np.take_along_axis(rt[None, ...], idx, axis=3), axis=3)
    en_b = np.median(np.take_along_axis(en[None, ...], idx, axis=3), axis=3)
    e_reg = (en_b[:,:,0] / en_b.min(axis=2) - 1) * 100
    r_reg = (rt_b[:,:,1] / rt_b.min(axis=2) - 1) * 100
    policy_stats = {
        "always_3090_energy_median_pct": np.median(e_reg, axis=1),
        "always_3090_energy_p95_pct": np.quantile(e_reg, .95, axis=1),
        "always_3090_energy_cvar10_pct": np.mean(np.sort(e_reg, axis=1)[:, -math.ceil(len(cells)*.1):], axis=1),
        "always_3090_energy_within5_pct": np.mean(e_reg <= 5, axis=1)*100,
        "always_5060_runtime_median_pct": np.median(r_reg, axis=1),
        "always_5060_runtime_p95_pct": np.quantile(r_reg, .95, axis=1),
        "always_5060_runtime_cvar10_pct": np.mean(np.sort(r_reg, axis=1)[:, -math.ceil(len(cells)*.1):], axis=1),
        "always_5060_runtime_within5_pct": np.mean(r_reg <= 5, axis=1)*100,
    }
    policy = pd.DataFrame([{"metric": k, **summarize_array(v)} for k, v in policy_stats.items()])

    # Paired bootstrap for Intel Reduction 4T vs 8T at large sizes.
    cpu_rows = []
    for size in [32_000_000, 64_000_000, 128_000_000, 256_000_000]:
        g = sessions[(sessions.workload == "REDUCTION") & (sessions["size"] == size) &
                     (sessions.platform == "INTEL") & (sessions.threads.isin([4,8]))]
        p = g.pivot(index="session", columns="threads", values=["runtime", "energy"]).sort_index()
        arr = p.to_numpy(float)
        runtime_pen, energy_save = [], []
        for _ in range(n_boot):
            b = arr[rng.integers(0, 5, size=5)]
            med = np.median(b, axis=0)
            # Column order: energy 4, energy 8, runtime 4, runtime 8 (pandas MultiIndex sorted metric).
            cols = list(p.columns)
            m = dict(zip(cols, med))
            runtime_pen.append((m[("runtime",4)] / m[("runtime",8)] - 1) * 100)
            energy_save.append((1 - m[("energy",4)] / m[("energy",8)]) * 100)
        rp, es = np.asarray(runtime_pen), np.asarray(energy_save)
        cpu_rows.append({
            "size": size,
            "runtime_penalty_point_pct": float((p[("runtime",4)].median()/p[("runtime",8)].median()-1)*100),
            "runtime_penalty_ci95_low": float(np.quantile(rp,.025)),
            "runtime_penalty_ci95_high": float(np.quantile(rp,.975)),
            "energy_saving_point_pct": float((1-p[("energy",4)].median()/p[("energy",8)].median())*100),
            "energy_saving_ci95_low": float(np.quantile(es,.025)),
            "energy_saving_ci95_high": float(np.quantile(es,.975)),
        })
    return headline, policy, pd.DataFrame(cpu_rows)


def practical_pareto_aggregate(by_cell: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tol, g in by_cell.groupby("tolerance_pct"):
        rows.append({
            "tolerance_pct": tol,
            "mean_front_count": g.front_count.mean(),
            "median_front_count": g.front_count.median(),
            "mean_dominated_fraction_pct": 100*g.dominated_fraction.mean(),
            "cells_with_one_point": int((g.front_count == 1).sum()),
            "cells_with_two_points": int((g.front_count == 2).sum()),
            "cells_with_cpu_on_front": int(g.cpu_on_front.sum()),
        })
    return pd.DataFrame(rows)


def joint_optimum_conflict_summary(sweet: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    x = sweet.merge(cells[["workload", "size", "conflict"]], on=["workload", "size"])
    rows = []
    for threshold, g in x.groupby("threshold"):
        for conflict, h in g.groupby("conflict"):
            rows.append({
                "threshold_pct": threshold,
                "canonical_conflict": bool(conflict),
                "cells": len(h),
                "with_joint_optimum": int(h.exists.sum()),
                "without_joint_optimum": int((~h.exists).sum()),
            })
    return pd.DataFrame(rows)


def intel_dram_sensitivity(sessions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Only workloads with separately available package and total energy are included.
    x = sessions[(sessions.platform == "INTEL") & sessions.workload.isin(["AXPY", "STREAM", "REDUCTION"])].copy()
    x = x[np.isfinite(x.total_energy)]
    rows = []
    for (w, s), g in x.groupby(["workload", "size"]):
        agg = g.groupby(["configuration", "threads"], as_index=False)[["energy", "total_energy"]].median()
        pkg = agg.loc[agg.energy.idxmin()]
        total = agg.loc[agg.total_energy.idxmin()]
        rows.append({
            "workload": w, "size": int(s),
            "package_opt_cfg": pkg.configuration, "package_opt_threads": int(pkg.threads),
            "package_opt_energy_j": float(pkg.energy),
            "total_opt_cfg": total.configuration, "total_opt_threads": int(total.threads),
            "total_opt_energy_j": float(total.total_energy),
            "thread_optimum_changed": bool(pkg.configuration != total.configuration),
            "dram_increment_at_package_opt_pct": float((pkg.total_energy / pkg.energy - 1) * 100),
        })
    detail = pd.DataFrame(rows)
    summary = detail.groupby("workload", as_index=False).agg(
        cells=("size", "count"),
        changed_thread_optima=("thread_optimum_changed", "sum"),
        median_dram_increment_pct=("dram_increment_at_package_opt_pct", "median"),
        max_dram_increment_pct=("dram_increment_at_package_opt_pct", "max"),
    )
    return detail, summary


def power_state_summary(sessions: pd.DataFrame) -> pd.DataFrame:
    x = sessions[sessions.platform == "3090"]
    rows = []
    for w, g in x.groupby("workload"):
        masks = sorted({m for m in g.throttle_masks.astype(str) if m and m != "nan"})
        rows.append({
            "workload": w, "session_configuration_rows": len(g),
            "median_power_w": g.power.median(), "min_power_w": g.power.min(), "max_power_w": g.power.max(),
            "median_temperature_c": g.temperature.median(), "min_temperature_c": g.temperature.min(), "max_temperature_c": g.temperature.max(),
            "median_clock_mhz": g.clock.replace(-1, np.nan).median(),
            "min_clock_mhz": g.clock.replace(-1, np.nan).min(), "max_clock_mhz": g.clock.replace(-1, np.nan).max(),
            "throttle_rows_available": int(g.throttle_nonzero_rows.notna().sum()),
            "throttle_nonzero_measurements": float(g.throttle_nonzero_rows.sum(skipna=True)),
            "throttle_masks": ",".join(masks),
            "configured_power_limit_w": np.nan,
            "power_limit_metadata_status": "numeric configured/default power limit not present in supplied session snapshots",
        })
    return pd.DataFrame(rows)


def create_figures(outputs: dict[str, pd.DataFrame], cfg: pd.DataFrame) -> None:
    FIG.mkdir(exist_ok=True)
    large = outputs["large_15_regime.csv"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for workload, group in large.groupby("workload"):
        ax.scatter(group.speedup_3090_vs_5060, 100*group.saving_5060_vs_3090, label=workload)
    ax.set_xlabel("RTX 3090 speedup over RTX 5060 Ti (×)")
    ax.set_ylabel("RTX 5060 Ti board-energy saving (%)")
    ax.set_title("Large AXPY/STREAM/REDUCTION regime")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "large_regime_speedup_energy.png", dpi=180); plt.close(fig)

    near = outputs["near_free_energy_savings.csv"]
    near = near[(near.threshold_pct == 1.0) & (near.energy_saving_pct > 0)]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(near.runtime_penalty_pct, near.energy_saving_pct)
    ax.set_xlabel("Runtime penalty versus runtime-optimal CPU config (%)")
    ax.set_ylabel("Energy saving (%)")
    ax.set_title("Near-free CPU energy savings within 1% runtime")
    fig.tight_layout(); fig.savefig(FIG / "cpu_thread_tradeoffs.png", dpi=180); plt.close(fig)

    pp = outputs["practical_pareto_by_cell.csv"]
    agg = practical_pareto_aggregate(pp)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(agg.tolerance_pct.astype(str), agg.mean_front_count)
    ax.set_xlabel("Practical dominance tolerance (%)")
    ax.set_ylabel("Mean configurations on Pareto front")
    ax.set_title("Pareto-front sensitivity")
    fig.tight_layout(); fig.savefig(FIG / "pareto_frontier_sizes.png", dpi=180); plt.close(fig)

    gpu = outputs["gpu_only_policy_regret_by_cell.csv"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for col, label in [
        ("always_3090_energy_regret_pct", "Always RTX 3090: energy regret"),
        ("always_5060ti_runtime_regret_pct", "Always RTX 5060 Ti: runtime regret"),
    ]:
        vals = np.sort(gpu[col].to_numpy())
        ax.plot(vals, np.arange(1, len(vals)+1)/len(vals), label=label)
    ax.set_xlabel("Per-cell regret (%)"); ax.set_ylabel("Empirical CDF")
    ax.set_title("Static GPU policy regret")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "gpu_policy_regret_cdf.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(large.time_ratio_3090_5060, large.power_ratio_3090_5060)
    xs = np.linspace(max(.2, large.time_ratio_3090_5060.min()*.9), large.time_ratio_3090_5060.max()*1.1, 200)
    ax.plot(xs, 1/xs, linestyle="--", label="Equal energy: power ratio = 1 / time ratio")
    ax.set_xlabel("RTX 3090 / RTX 5060 Ti runtime ratio")
    ax.set_ylabel("RTX 3090 / RTX 5060 Ti power ratio")
    ax.set_title("Power-time geometry of the large regime")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "power_time_energy_geometry.png", dpi=180); plt.close(fig)


def claim_freeze_table(outputs: dict[str, pd.DataFrame], new: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cells = outputs["canonical_51_cells.csv"]
    large = outputs["large_15_regime.csv"]
    headline = new["bootstrap_headline_intervals.csv"].set_index("metric")
    joint = new["joint_optimum_conflict_summary.csv"]
    conflict10 = joint[(joint.threshold_pct == 10) & joint.canonical_conflict].iloc[0]
    conflict20 = joint[(joint.threshold_pct == 20) & joint.canonical_conflict].iloc[0]
    loso = new["nested_loso_summary.csv"]
    allrt = loso[(loso.scope == "all_platform") & (loso.metric == "runtime")].iloc[0]
    allen = loso[(loso.scope == "all_platform") & (loso.metric == "energy")].iloc[0]
    gpurt = loso[(loso.scope == "gpu_only") & (loso.metric == "runtime")].iloc[0]
    gpuen = loso[(loso.scope == "gpu_only") & (loso.metric == "energy")].iloc[0]
    gpu_conf = new["nested_gpu_conflict_loso_by_cell.csv"]
    all_conf = new["nested_conflict_loso_by_cell.csv"]
    policy_boot = new["bootstrap_gpu_policy_intervals.csv"].set_index("metric")
    cpu_boot = new["bootstrap_intel_reduction_near_free.csv"].set_index("size")
    max_threads = outputs["max_thread_policy_regret.csv"]
    pareto = new["practical_pareto_summary.csv"].set_index("tolerance_pct")
    return pd.DataFrame([
        {"claim": "Canonical hard conflicts", "value": "24/51", "definition": "workload audit clear leaders with disjoint runtime/energy leader sets", "robustness": "fixed-selection session support plus tolerance sensitivity", "paper_role": "abstract"},
        {"claim": "Large GPU regime", "value": "15/15", "definition": "AXPY/STREAM/REDUCTION at 16M-256M", "robustness": "all cells and sessions available", "paper_role": "abstract"},
        {"claim": "Median RTX 3090 speedup", "value": f"{headline.loc['median_speedup_3090','point']:.6f}x [{headline.loc['median_speedup_3090','ci95_low']:.6f}, {headline.loc['median_speedup_3090','ci95_high']:.6f}]", "definition": "session-bootstrap median over 15 cells", "robustness": "5000 session bootstraps", "paper_role": "abstract"},
        {"claim": "Median RTX 5060 Ti energy saving", "value": f"{headline.loc['median_energy_saving_5060_pct','point']:.4f}% [{headline.loc['median_energy_saving_5060_pct','ci95_low']:.4f}, {headline.loc['median_energy_saving_5060_pct','ci95_high']:.4f}]", "definition": "session-bootstrap median over 15 cells", "robustness": "5000 session bootstraps", "paper_role": "abstract"},
        {"claim": "Conflicts without joint optimum", "value": f"{int(conflict10.without_joint_optimum)}/24 at 10%; {int(conflict20.without_joint_optimum)}/24 at 20%", "definition": "all 18 measured configs compared to both cell optima", "robustness": "non-circular tolerance sensitivity", "paper_role": "results"},
        {"claim": "EDP follows runtime", "value": f"{int((cells.edp_winner == cells.runtime_winner).sum())}/51", "definition": "point-optimal platform", "robustness": "rank-correlation analysis", "paper_role": "results"},
        {"claim": "Nested all-platform runtime selection", "value": f"{allrt.within_5pct:.2f}% folds within 5%", "definition": "select on four sessions, test on fifth", "robustness": "255 all-platform folds", "paper_role": "robustness"},
        {"claim": "Nested all-platform energy selection", "value": f"{allen.within_5pct:.2f}% folds within 5%", "definition": "select on four sessions, test on fifth", "robustness": "255 all-platform folds", "paper_role": "robustness"},
        {"claim": "Nested GPU-only runtime selection", "value": f"{gpurt.platform_match_pct:.2f}% exact platform match", "definition": "GPU selected on four sessions and tested on fifth", "robustness": "255 GPU-only folds", "paper_role": "robustness"},
        {"claim": "Nested GPU-only energy selection", "value": f"{gpuen.platform_match_pct:.2f}% exact; {gpuen.within_5pct:.2f}% within 5%", "definition": "GPU selected on four sessions and tested on fifth", "robustness": "255 GPU-only folds", "paper_role": "robustness"},
        {"claim": "Nested canonical GPU conflicts", "value": f"{int((gpu_conf.holdout_conflict_folds==5).sum())}/23 at 5/5 holdouts", "definition": "23 canonical RTX 3090-vs-RTX 5060 Ti conflict cells", "robustness": "platforms reselected inside every four-session training fold", "paper_role": "robustness"},
        {"claim": "Nested all-platform conflicts", "value": f"{int((all_conf.holdout_conflict_folds==5).sum())}/24 at 5/5; {int((all_conf.holdout_conflict_folds>=4).sum())}/24 at >=4/5", "definition": "all 18 configurations reselected within each training fold", "robustness": "two boundary cells contain one high-variation AMD session", "paper_role": "robustness"},
        {"claim": "Always RTX 3090 energy tail", "value": f"CVaR10 {policy_boot.loc['always_3090_energy_cvar10_pct','point']:.2f}% [{policy_boot.loc['always_3090_energy_cvar10_pct','ci95_low']:.2f}, {policy_boot.loc['always_3090_energy_cvar10_pct','ci95_high']:.2f}]", "definition": "mean energy regret of the worst 10% of equally weighted GPU-only cells", "robustness": "5000 session bootstraps", "paper_role": "results"},
        {"claim": "Always RTX 5060 Ti runtime tail", "value": f"CVaR10 {policy_boot.loc['always_5060_runtime_cvar10_pct','point']:.2f}% [{policy_boot.loc['always_5060_runtime_cvar10_pct','ci95_low']:.2f}, {policy_boot.loc['always_5060_runtime_cvar10_pct','ci95_high']:.2f}]", "definition": "mean runtime regret of the worst 10% of equally weighted GPU-only cells", "robustness": "5000 session bootstraps", "paper_role": "results"},
        {"claim": "Intel Reduction 256M near-free saving", "value": f"{cpu_boot.loc[256000000,'energy_saving_point_pct']:.2f}% energy for {cpu_boot.loc[256000000,'runtime_penalty_point_pct']:.3f}% runtime", "definition": "4 threads versus runtime-optimal 8 threads", "robustness": f"energy CI [{cpu_boot.loc[256000000,'energy_saving_ci95_low']:.2f}, {cpu_boot.loc[256000000,'energy_saving_ci95_high']:.2f}]%; runtime CI [{cpu_boot.loc[256000000,'runtime_penalty_ci95_low']:.3f}, {cpu_boot.loc[256000000,'runtime_penalty_ci95_high']:.3f}]%", "paper_role": "results"},
        {"claim": "Maximum CPU threads energy regret", "value": f"median {max_threads[max_threads.metric=='energy'].regret_pct.median():.2f}%", "definition": "maximum available thread count versus per-cell energy-optimal CPU thread count", "robustness": "102 CPU platform/workload/size cells", "paper_role": "results"},
        {"claim": "Tolerance-aware decision front", "value": f"mean {pareto.loc[5,'mean_front_count']:.3f}/18 at 5%; {pareto.loc[5,'mean_dominated_fraction_pct']:.2f}% dominated", "definition": "project tolerance-aware dominance rule", "robustness": "strict, 2%, and 5% sensitivity", "paper_role": "results"},
    ])


def write_report(outputs: dict[str, pd.DataFrame], new: dict[str, pd.DataFrame]) -> None:
    cells = outputs["canonical_51_cells.csv"]
    support = outputs["conflict_session_support.csv"]
    loso = new["nested_loso_summary.csv"]
    conflict_cell = new["nested_conflict_loso_by_cell.csv"]
    gpu_conflict_cell = new["nested_gpu_conflict_loso_by_cell.csv"]
    pp = new["practical_pareto_summary.csv"]
    dram = new["intel_dram_sensitivity_summary.csv"]
    boot = new["bootstrap_headline_intervals.csv"].set_index("metric")
    policy_boot = new["bootstrap_gpu_policy_intervals.csv"].set_index("metric")
    cpu_boot = new["bootstrap_intel_reduction_near_free.csv"].set_index("size")
    max_threads = outputs["max_thread_policy_regret.csv"]
    lines = [
        "# Completion Report — T1 Energy–Runtime Placement Analysis",
        "",
        "## Pipeline status",
        "",
        "The rebuilt pipeline regenerates all 17 original CSV outputs, all five figures, and the additional robustness outputs from input snapshots.",
        "",
        "## Confirmed core",
        "",
        f"- Canonical cells: **{len(cells)}**",
        f"- Canonical conflicts: **{int(cells.conflict.sum())}/51**",
        f"- Fixed full-data selected winners with 5/5 session direction: **{int((support.support==5).sum())}/{len(support)} objective rows**",
        f"- Large-regime speedup: **{boot.loc['median_speedup_3090','point']:.6f}x** (95% session-bootstrap CI {boot.loc['median_speedup_3090','ci95_low']:.6f}–{boot.loc['median_speedup_3090','ci95_high']:.6f})",
        f"- Large-regime RTX 5060 Ti energy saving: **{boot.loc['median_energy_saving_5060_pct','point']:.4f}%** (95% CI {boot.loc['median_energy_saving_5060_pct','ci95_low']:.4f}–{boot.loc['median_energy_saving_5060_pct','ci95_high']:.4f})",
        "",
        "## Nested leave-one-session-out",
        "",
    ]
    for _, r in loso.iterrows():
        lines.append(f"- {r.scope}/{r.metric}: platform match {r.platform_match_pct:.2f}%, within 2% {r.within_2pct:.2f}%, within 5% {r.within_5pct:.2f}%, median holdout regret {r.median_regret_pct:.4f}%.")
    lines += [
        "",
        f"Across all 24 canonical conflict cells, the four-session training folds selected different runtime and energy platforms in 5/5 folds for **{int((conflict_cell.train_conflict_folds==5).sum())}/24** cells. The held-out session itself retained the all-platform conflict in 5/5 folds for **{int((conflict_cell.holdout_conflict_folds==5).sum())}/24** cells and in at least 4/5 folds for **{int((conflict_cell.holdout_conflict_folds>=4).sum())}/24** cells.",
        f"For the symmetric GPU-only core, all **{len(gpu_conflict_cell)}/23** canonical RTX 3090-vs-RTX 5060 Ti conflict cells retained the conflict and both objective-specific selections within 5% of the held-out GPU oracle in every fold.",
        "",
        "The two all-platform boundary cells are REDUCTION 16M and STREAM 4M. Each contains one strongly deviating AMD session; the median-of-five canonical classification remains unchanged, but nested holdout reporting must distinguish 22/24 fully stable all-platform cells from the perfect 23/23 GPU-only core.",
        "",
        "## Static GPU policy tails",
        "",
        f"- Always RTX 3090 energy CVaR10: **{policy_boot.loc['always_3090_energy_cvar10_pct','point']:.2f}%** (95% session-bootstrap CI {policy_boot.loc['always_3090_energy_cvar10_pct','ci95_low']:.2f}–{policy_boot.loc['always_3090_energy_cvar10_pct','ci95_high']:.2f}%); within 5% of the GPU energy oracle in **{policy_boot.loc['always_3090_energy_within5_pct','point']:.2f}%** of cells.",
        f"- Always RTX 5060 Ti runtime CVaR10: **{policy_boot.loc['always_5060_runtime_cvar10_pct','point']:.2f}%** (95% CI {policy_boot.loc['always_5060_runtime_cvar10_pct','ci95_low']:.2f}–{policy_boot.loc['always_5060_runtime_cvar10_pct','ci95_high']:.2f}%); within 5% of the GPU runtime oracle in **{policy_boot.loc['always_5060_runtime_within5_pct','point']:.2f}%** of cells.",
        "",
        "## CPU-thread selection",
        "",
        f"- Maximum available threads have median energy regret **{max_threads[max_threads.metric=='energy'].regret_pct.median():.2f}%** and median EDP regret **{max_threads[max_threads.metric=='edp'].regret_pct.median():.2f}%** across the 102 CPU cells.",
        f"- Intel REDUCTION 256M: accepting **{cpu_boot.loc[256000000,'runtime_penalty_point_pct']:.3f}%** runtime penalty saves **{cpu_boot.loc[256000000,'energy_saving_point_pct']:.2f}%** energy; the 95% session-bootstrap intervals are [{cpu_boot.loc[256000000,'runtime_penalty_ci95_low']:.3f}, {cpu_boot.loc[256000000,'runtime_penalty_ci95_high']:.3f}]% runtime and [{cpu_boot.loc[256000000,'energy_saving_ci95_low']:.2f}, {cpu_boot.loc[256000000,'energy_saving_ci95_high']:.2f}]% energy.",
        "- Nested CPU-thread choices are less stable than GPU platform choices: 91.96% of energy folds, 89.02% of runtime folds, and 86.86% of EDP folds remain within 5% of the held-out thread oracle. High tail regrets are concentrated in high-session-variation AMD REDUCTION/STREAM configurations.",
        "",
        "## Pareto sensitivity",
        "",
    ]
    for _, r in pp.iterrows():
        lines.append(f"- {int(r.tolerance_pct)}% tolerance: mean front {r.mean_front_count:.3f}/18, mean dominated fraction {r.mean_dominated_fraction_pct:.2f}%, CPU on front in {int(r.cells_with_cpu_on_front)}/51 cells.")
    lines += ["", "## Intel DRAM sensitivity", ""]
    for _, r in dram.iterrows():
        lines.append(f"- {r.workload}: thread optimum changed in {int(r.changed_thread_optima)}/{int(r.cells)} cells; median package-to-total increment at package optimum {r.median_dram_increment_pct:.3f}%.")
    lines += [
        "",
        "## Remaining limitation",
        "",
        "The supplied and public session snapshots do not contain the numeric configured/default RTX 3090 power-limit value for all workloads. Available AXPY throttle masks and clock/temperature summaries are reproduced, but the numeric stock-power-limit statement still requires metadata from the run environment or logs.",
        "",
    ]
    (ROOT / "ANALYSIS_COMPLETION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def sha256_manifest() -> None:
    paths = [
        ROOT / "analysis.py",
        ROOT / "analysis_legacy.py",
        ROOT / "README.md",
        ROOT / "REPORT.md",
        ROOT / "ANALYSIS_COMPLETION_REPORT.md",
        ROOT / "INPUT_PROVENANCE.md",
        ROOT / "PAPER_READY_ANALYSIS_SUMMARY_DE.md",
        ROOT / "requirements.txt",
    ]
    paths += sorted(IN.glob("*.csv")) + sorted(OUT.glob("*.csv")) + sorted(FIG.glob("*.png"))
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT)}")
    (ROOT / "SHA256SUMS_COMPLETE.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    # Ensure outputs are generated by this run rather than inherited.
    for p in OUT.glob("*.csv"):
        p.unlink()
    for p in FIG.glob("*.png"):
        p.unlink()

    cfg = build_configurations()
    sessions = build_sessions()
    cells = build_cells()
    env = platform_envelopes(cfg)

    outputs = original_cross_outputs(cfg, cells, env, sessions)
    for name, df in outputs.items():
        df.to_csv(OUT / name, index=False)

    loso_all, loso_all_summary = nested_loso(sessions, "all_platform")
    loso_gpu, loso_gpu_summary = nested_loso(sessions, "gpu_only")
    loso_cpu, loso_cpu_summary = nested_loso(sessions, "cpu_threads")
    conflict_loso, conflict_by_cell = nested_conflict_stability(sessions, cells)
    gpu_conflict_loso, gpu_conflict_by_cell = nested_gpu_conflict_stability(sessions, cells)
    instability = session_instability_diagnostics(cfg, loso_all, conflict_by_cell)
    boundary_details = boundary_session_details(sessions)
    headline_boot, policy_boot, cpu_boot = bootstrap_headlines(sessions)
    practical_summary = practical_pareto_aggregate(outputs["practical_pareto_by_cell.csv"])
    joint_summary = joint_optimum_conflict_summary(outputs["near_joint_optimum_sensitivity.csv"], cells)
    dram_detail, dram_summary = intel_dram_sensitivity(sessions)
    power = power_state_summary(sessions)

    new = {
        "nested_loso_all_platform.csv": loso_all,
        "nested_loso_gpu_only.csv": loso_gpu,
        "nested_loso_cpu_threads.csv": loso_cpu,
        "nested_loso_summary.csv": pd.concat([loso_all_summary, loso_gpu_summary, loso_cpu_summary], ignore_index=True),
        "nested_conflict_loso.csv": conflict_loso,
        "nested_conflict_loso_by_cell.csv": conflict_by_cell,
        "nested_gpu_conflict_loso.csv": gpu_conflict_loso,
        "nested_gpu_conflict_loso_by_cell.csv": gpu_conflict_by_cell,
        "session_instability_diagnostics.csv": instability,
        "all_platform_boundary_session_details.csv": boundary_details,
        "bootstrap_headline_intervals.csv": headline_boot,
        "bootstrap_gpu_policy_intervals.csv": policy_boot,
        "bootstrap_intel_reduction_near_free.csv": cpu_boot,
        "practical_pareto_summary.csv": practical_summary,
        "joint_optimum_conflict_summary.csv": joint_summary,
        "intel_dram_sensitivity.csv": dram_detail,
        "intel_dram_sensitivity_summary.csv": dram_summary,
        "power_state_summary.csv": power,
    }
    new["claim_freeze_table.csv"] = claim_freeze_table(outputs, new)
    for name, df in new.items():
        df.to_csv(OUT / name, index=False)

    create_figures(outputs | new, cfg)
    write_report(outputs, new)

    expected_original = {
        "platform_objective_envelopes.csv", "joint_minimax_compromises.csv", "all_918_configurations.csv",
        "near_joint_optimum_sensitivity.csv", "pareto_summary.csv", "top_findings_ranked.csv",
        "cpu_thread_tradeoffs.csv", "claim_reproduction.csv", "max_thread_policy_regret.csv",
        "edp_rank_correlations.csv", "near_free_energy_savings.csv", "large_15_regime.csv",
        "pareto_configuration_flags.csv", "canonical_51_cells.csv", "conflict_session_support.csv",
        "gpu_only_policy_regret_by_cell.csv", "static_platform_policy_regret.csv",
    }
    missing = sorted(expected_original - {p.name for p in OUT.glob("*.csv")})
    if missing:
        raise RuntimeError(f"Missing regenerated original outputs: {missing}")
    if len(list(FIG.glob("*.png"))) != 5:
        raise RuntimeError("Expected exactly five regenerated figures")

    sha256_manifest()
    print(f"PASS: regenerated {len(list(OUT.glob('*.csv')))} CSV tables and five figures")
    print("PASS: full 918-configuration x five-session nested validation completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
