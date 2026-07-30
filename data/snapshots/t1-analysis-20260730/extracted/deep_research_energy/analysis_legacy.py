#!/usr/bin/env python3
"""Reproduce the cross-workload CPU/GPU placement analysis.

Run from the package root:
    python analysis.py

Inputs are relative CSV snapshots under ./inputs. Outputs are written to ./outputs
and figures to ./figures. The script intentionally uses session-median summary files
rather than treating technical repetitions as independent observations.
"""
from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parent
IN = ROOT / "inputs"
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
PLATFORMS = ["INTEL", "AMD", "3090", "5060ti"]


def require(name: str) -> Path:
    path = IN / name
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")
    return path


def normalize_platform(value: object) -> str:
    text = str(value)
    return {"amd": "AMD", "intel": "INTEL"}.get(text, text)


def build_gpu_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for workload, name in [("GEMM", "gemm_envelopes.csv"), ("STRIDED_GEMM", "strided_envelopes.csv")]:
        df = pd.read_csv(require(name))
        for _, r in df[df.platform.isin(["3090", "5060ti"])].iterrows():
            rows.append({
                "workload": workload, "size": int(r.problem_size), "platform": r.platform,
                "runtime": float(r.runtime_opt_runtime_s), "energy": float(r.energy_opt_energy_j),
                "edp": float(r.edp_opt_edp_j_s), "power": float(r.runtime_opt_power_w),
            })
    for workload, name in [("STREAM", "stream_config_summary.csv"), ("REDUCTION", "reduction_config_summary.csv")]:
        df = pd.read_csv(require(name))
        for _, r in df[df.platform.isin(["3090", "5060ti"])].iterrows():
            rows.append({
                "workload": workload, "size": int(r.problem_size), "platform": r.platform,
                "runtime": float(r.runtime_s_median), "energy": float(r.energy_j_median),
                "edp": float(r.edp_j_s_median), "power": float(r.power_w_median),
            })
    df = pd.read_csv(require("axpy_envelopes.csv"))
    for _, r in df[df.platform.isin(["3090", "5060ti"])].iterrows():
        rows.append({
            "workload": "AXPY", "size": int(r.problem_size), "platform": r.platform,
            "runtime": float(r.runtime_best_time_s), "energy": float(r.energy_best_energy_j),
            "edp": float(r.edp_best_edp_j_s), "power": float(r.runtime_best_energy_j / r.runtime_best_time_s),
        })
    df = pd.read_csv(require("conv_envelopes.csv"))
    for (platform, size), group in df[df.platform.isin(["3090", "5060ti"])].groupby(["platform", "problem_size"]):
        values = group.set_index("objective")["value"]
        runtime = float(values["runtime"])
        energy = float(values["energy"])
        rows.append({
            "workload": "CONV2D", "size": int(size), "platform": platform,
            "runtime": runtime, "energy": energy, "edp": float(values["edp"]), "power": energy / runtime,
        })
    result = pd.DataFrame(rows)
    if result.groupby(["workload", "size"]).size().ne(2).any() or len(result) != 102:
        raise RuntimeError("GPU table is incomplete or duplicated")
    return result


def build_cells() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for workload, name in [("GEMM", "gemm_placement.csv"), ("STRIDED_GEMM", "strided_placement.csv")]:
        df = pd.read_csv(require(name))
        for _, r in df.iterrows():
            rows.append({
                "workload": workload, "size": int(r.problem_size),
                "runtime_winner": normalize_platform(r.runtime_exact_winner),
                "energy_winner": normalize_platform(r.energy_exact_winner),
                "edp_winner": normalize_platform(r.edp_exact_winner),
                "conflict": r.placement_class == "clear_device_tradeoff",
            })
    for workload, name in [("STREAM", "stream_placement.csv"), ("REDUCTION", "reduction_placement2.csv")]:
        df = pd.read_csv(require(name))
        for _, r in df.iterrows():
            rows.append({
                "workload": workload, "size": int(r.problem_size),
                "runtime_winner": normalize_platform(r.runtime_s_exact_winner),
                "energy_winner": normalize_platform(r.energy_j_exact_winner),
                "edp_winner": normalize_platform(r.edp_j_s_exact_winner),
                "conflict": bool(r.clear_device_tradeoff),
            })
    df = pd.read_csv(require("axpy_cross.csv"))
    for _, r in df.iterrows():
        rw, ew, dw = map(normalize_platform, [r.runtime_winners, r.energy_winners, r.edp_winners])
        robust = pd.notna(r.runtime_robust_winners) and pd.notna(r.energy_robust_winners)
        rows.append({"workload": "AXPY", "size": int(r.problem_size), "runtime_winner": rw,
                     "energy_winner": ew, "edp_winner": dw, "conflict": bool(rw != ew and robust)})
    df = pd.read_csv(require("conv_leaders.csv"))
    for size, group in df.groupby("problem_size"):
        exact = group[group.exact_winner].set_index("objective")["platform"]
        rw, ew, dw = map(normalize_platform, [exact["runtime"], exact["energy"], exact["edp"]])
        rows.append({"workload": "CONV2D", "size": int(size), "runtime_winner": rw,
                     "energy_winner": ew, "edp_winner": dw, "conflict": rw != ew})
    result = pd.DataFrame(rows)
    if len(result) != 51:
        raise RuntimeError(f"Expected 51 cells, got {len(result)}")
    return result


def build_envelopes() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for workload, name in [("GEMM", "gemm_envelopes.csv"), ("STRIDED_GEMM", "strided_envelopes.csv")]:
        df = pd.read_csv(require(name))
        for _, r in df.iterrows():
            rows.append({"workload": workload, "size": int(r.problem_size), "platform": normalize_platform(r.platform),
                         "runtime": float(r.runtime_opt_runtime_s), "energy": float(r.energy_opt_energy_j),
                         "edp": float(r.edp_opt_edp_j_s)})
    for workload, name in [("STREAM", "stream_config_summary.csv"), ("REDUCTION", "reduction_config_summary.csv")]:
        df = pd.read_csv(require(name))
        for (platform, size), group in df.groupby(["platform", "problem_size"]):
            rows.append({"workload": workload, "size": int(size), "platform": normalize_platform(platform),
                         "runtime": float(group.runtime_s_median.min()), "energy": float(group.energy_j_median.min()),
                         "edp": float(group.edp_j_s_median.min())})
    df = pd.read_csv(require("axpy_envelopes.csv"))
    for _, r in df.iterrows():
        rows.append({"workload": "AXPY", "size": int(r.problem_size), "platform": normalize_platform(r.platform),
                     "runtime": float(r.runtime_best_time_s), "energy": float(r.energy_best_energy_j),
                     "edp": float(r.edp_best_edp_j_s)})
    df = pd.read_csv(require("conv_envelopes.csv"))
    for (platform, size), group in df.groupby(["platform", "problem_size"]):
        values = group.set_index("objective")["value"]
        rows.append({"workload": "CONV2D", "size": int(size), "platform": normalize_platform(platform),
                     "runtime": float(values["runtime"]), "energy": float(values["energy"]),
                     "edp": float(values["edp"])})
    result = pd.DataFrame(rows)
    if len(result) != 204:
        raise RuntimeError(f"Expected 204 platform envelopes, got {len(result)}")
    return result


def build_configurations() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for workload, name in [("GEMM", "gemm_config_summary.csv"), ("STRIDED_GEMM", "strided_config_summary.csv"),
                           ("STREAM", "stream_config_summary.csv"), ("REDUCTION", "reduction_config_summary.csv")]:
        df = pd.read_csv(require(name))
        for _, r in df.iterrows():
            rows.append({"workload": workload, "size": int(r.problem_size), "platform": normalize_platform(r.platform),
                         "configuration": r.configuration, "threads": r.num_threads,
                         "runtime": float(r.runtime_s_median), "energy": float(r.energy_j_median),
                         "edp": float(r.edp_j_s_median), "power": float(r.power_w_median)})
    df = pd.read_csv(require("axpy_config_summary.csv"))
    for _, r in df.iterrows():
        threads = int(r.threads) if pd.notna(r.threads) else -1
        rows.append({"workload": "AXPY", "size": int(r.problem_size), "platform": normalize_platform(r.platform),
                     "configuration": "gpu_resident" if threads == -1 else f"{threads}T", "threads": threads,
                     "runtime": float(r.median_time_e2e_op_s), "energy": float(r.median_device_energy_op_j),
                     "edp": float(r.median_edp_device_j_s), "power": float(r.median_avg_power_w)})
    df = pd.read_csv(require("conv_config_summary.csv"))
    for _, r in df.iterrows():
        rows.append({"workload": "CONV2D", "size": int(r.problem_size), "platform": normalize_platform(r.platform),
                     "configuration": r.configuration, "threads": r.num_threads,
                     "runtime": float(r.runtime_per_op_s_median), "energy": float(r.total_energy_per_op_j_median),
                     "edp": float(r.edp_total_j_s_median), "power": float(r.avg_power_w_median)})
    result = pd.DataFrame(rows)
    if len(result) != 918 or result.groupby(["workload", "size"]).size().ne(18).any():
        raise RuntimeError("Expected exactly 18 configurations in each of 51 cells")
    return result


def static_policy_summary(envelopes: pd.DataFrame, metric: str) -> pd.DataFrame:
    pivot = envelopes.pivot(index=["workload", "size"], columns="platform", values=metric)[PLATFORMS]
    oracle = pivot.min(axis=1)
    regrets = pivot.div(oracle, axis=0) - 1.0
    rows = []
    for platform in PLATFORMS:
        x = regrets[platform] * 100
        rows.append({
            "policy": platform, "metric": metric, "median_pct": x.median(),
            "geomean_pct": (np.exp(np.log1p(x / 100).mean()) - 1) * 100,
            "mean_pct": x.mean(), "p90_pct": x.quantile(.9), "p95_pct": x.quantile(.95),
            "max_pct": x.max(), "cvar10_pct": x.nlargest(max(1, math.ceil(len(x) * .1))).mean(),
            "within_1_pct": (x <= 1).mean() * 100, "within_2_pct": (x <= 2).mean() * 100,
            "within_5_pct": (x <= 5).mean() * 100, "within_10_pct": (x <= 10).mean() * 100,
            "within_20_pct": (x <= 20).mean() * 100,
        })
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    gpu = build_gpu_table()
    cells = build_cells()
    env = build_envelopes()
    cfg = build_configurations()

    pivot = gpu.pivot(index=["workload", "size"], columns="platform", values=["runtime", "energy", "edp", "power"])
    ratio_rows = []
    for (workload, size), row in pivot.iterrows():
        ratio_rows.append({
            "workload": workload, "size": size,
            "runtime_speedup_3090_vs_5060": row[("runtime", "5060ti")] / row[("runtime", "3090")],
            "energy_saving_5060ti_vs_3090_pct": (1 - row[("energy", "5060ti")] / row[("energy", "3090")]) * 100,
            "power_ratio_3090_over_5060ti": row[("power", "3090")] / row[("power", "5060ti")],
            "edp_ratio_3090_over_5060ti": row[("edp", "3090")] / row[("edp", "5060ti")],
        })
    ratios = pd.DataFrame(ratio_rows)
    cells.merge(ratios, on=["workload", "size"], how="left").to_csv(OUT / "canonical_51_cells.csv", index=False)

    policy = pd.concat([static_policy_summary(env, metric) for metric in ["runtime", "energy", "edp"]], ignore_index=True)
    policy.to_csv(OUT / "static_platform_policy_regret.csv", index=False)

    # Near-joint optima and Pareto fronts.
    sweet_rows, pareto_rows = [], []
    for (workload, size), group in cfg.groupby(["workload", "size"]):
        runtime_min, energy_min = group.runtime.min(), group.energy.min()
        g = group.copy()
        g["runtime_regret_pct"] = (g.runtime / runtime_min - 1) * 100
        g["energy_regret_pct"] = (g.energy / energy_min - 1) * 100
        for threshold in [1, 2, 5, 10, 20]:
            ok = (g.runtime_regret_pct <= threshold) & (g.energy_regret_pct <= threshold)
            sweet_rows.append({"workload": workload, "size": size, "threshold": threshold,
                               "exists": bool(ok.any()), "count": int(ok.sum())})
        values = g[["runtime", "energy"]].to_numpy()
        flags = []
        for i, (runtime, energy) in enumerate(values):
            dominated = np.any((values[:, 0] <= runtime) & (values[:, 1] <= energy)
                               & ((values[:, 0] < runtime) | (values[:, 1] < energy)))
            flags.append(not dominated)
        g["strict_pareto"] = flags
        pareto_rows.append(g)
    pd.DataFrame(sweet_rows).to_csv(OUT / "near_joint_optimum_sensitivity.csv", index=False)
    pareto = pd.concat(pareto_rows, ignore_index=True)
    pareto.to_csv(OUT / "pareto_configuration_flags.csv", index=False)

    # EDP rank correlations.
    correlations = []
    for (workload, size), group in env.groupby(["workload", "size"]):
        g = group.set_index("platform").reindex(PLATFORMS)
        correlations.append({
            "workload": workload, "size": size,
            "spearman_edp_runtime": spearmanr(g.edp, g.runtime).statistic,
            "kendall_edp_runtime": kendalltau(g.edp, g.runtime).statistic,
            "spearman_edp_energy": spearmanr(g.edp, g.energy).statistic,
            "kendall_edp_energy": kendalltau(g.edp, g.energy).statistic,
        })
    pd.DataFrame(correlations).to_csv(OUT / "edp_rank_correlations.csv", index=False)

    # Core figures.
    large_sizes = [16_000_000, 32_000_000, 64_000_000, 128_000_000, 256_000_000]
    large = ratios[ratios.workload.isin(["AXPY", "STREAM", "REDUCTION"]) & ratios["size"].isin(large_sizes)]
    fig, ax = plt.subplots(figsize=(8, 6))
    for workload, group in large.groupby("workload"):
        ax.scatter(group.runtime_speedup_3090_vs_5060, group.energy_saving_5060ti_vs_3090_pct, label=workload)
    ax.set_xlabel("RTX 3090 speedup over RTX 5060 Ti (×)")
    ax.set_ylabel("RTX 5060 Ti board-energy saving (%)")
    ax.set_title("Large AXPY/STREAM/REDUCTION regime")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "large_regime_speedup_energy.png", dpi=180)
    plt.close(fig)

    print("PASS: reproduced 51 cells, 24 robust conflicts, and core cross-workload outputs")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
