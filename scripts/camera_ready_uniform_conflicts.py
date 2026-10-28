from pathlib import Path
import pandas as pd

BASE = Path(
    "data/snapshots/t1-analysis-20260730/"
    "extracted/deep_research_energy/outputs"
)

OUT = Path("results/camera-ready")

cfg = pd.read_csv(BASE / "all_918_configurations.csv")
legacy = pd.read_csv(BASE / "canonical_51_cells.csv")

rows = []

for (workload, size), g in cfg.groupby(["workload", "size"]):

    assert len(g) == 18, (workload, size)

    runtime_best = g.loc[g.runtime.idxmin()]
    energy_best = g.loc[g.energy.idxmin()]

    exact_conflict = (
        runtime_best.platform,
        runtime_best.configuration
    ) != (
        energy_best.platform,
        energy_best.configuration
    )

    def unresolved(tolerance):
        joint = (
            (g.runtime <= (1 + tolerance) * runtime_best.runtime)
            &
            (g.energy <= (1 + tolerance) * energy_best.energy)
        )
        return not bool(joint.any())

    rows.append({
        "workload": workload,
        "size": size,
        "runtime_platform": runtime_best.platform,
        "runtime_config": runtime_best.configuration,
        "energy_platform": energy_best.platform,
        "energy_config": energy_best.configuration,
        "exact_conflict": exact_conflict,
        "unresolved_10pct": unresolved(0.10),
        "unresolved_20pct": unresolved(0.20),
    })

result = pd.DataFrame(rows)

result = result.merge(
    legacy[["workload", "size", "conflict"]],
    on=["workload", "size"],
    validate="one_to_one"
).rename(columns={"conflict": "legacy_conflict"})

counts = (
    len(result),
    int(result.exact_conflict.sum()),
    int(result.unresolved_10pct.sum()),
    int(result.unresolved_20pct.sum()),
)

assert counts == (51, 25, 22, 19), counts

OUT.mkdir(parents=True, exist_ok=True)

result.to_csv(
    OUT / "uniform_conflicts_51_cells.csv",
    index=False
)

print("PASS: Einheitliche Konfliktanalyse")
print("Zellen:", counts[0])
print("Exakte Konflikte:", counts[1])
print("Ungelöst bei 10 %:", counts[2])
print("Ungelöst bei 20 %:", counts[3])

print("\nAbweichungen zur ursprünglichen Klassifikation:")
print(
    result.loc[
        result.exact_conflict != result.legacy_conflict,
        ["workload", "size", "legacy_conflict", "exact_conflict"]
    ].to_string(index=False)
)

# Camera-ready: Leave-one-session-out robustness
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.placement_analysis import analysis as a

a.IN = (BASE.parent / "inputs").resolve()

loso_cells = result.rename(columns={
    "exact_conflict": "conflict",
    "runtime_platform": "runtime_winner",
    "energy_platform": "energy_winner",
})

sessions = a.build_sessions()

all_detail, all_cells = a.nested_conflict_stability(
    sessions, loso_cells
)

gpu_detail, gpu_cells = a.nested_gpu_conflict_stability(
    sessions, loso_cells
)

# Validate expected results
assert len(all_cells) == 25
assert len(gpu_cells) == 24

assert (all_cells.holdout_conflict_folds == 5).sum() == 22
assert (gpu_cells.holdout_conflict_folds == 5).sum() == 23

# Save complete results
all_detail.to_csv(OUT / "uniform_loso_all_detail.csv", index=False)
all_cells.to_csv(OUT / "uniform_loso_all_by_cell.csv", index=False)

gpu_detail.to_csv(OUT / "uniform_loso_gpu_detail.csv", index=False)
gpu_cells.to_csv(OUT / "uniform_loso_gpu_by_cell.csv", index=False)

print("\nPASS: Leave-one-session-out analysis")
print("All-platform stable:", 22, "/ 25")
print("GPU-GPU stable:", 23, "/ 24")


# Camera-ready: Per-cell conflict margins and session stability

annex_rows = []

for _, cell in result[result.exact_conflict].iterrows():
    w, s = cell.workload, int(cell["size"])

    g = cfg[(cfg.workload == w) & (cfg["size"] == s)]
    rt = g.loc[g.runtime.idxmin()]
    en = g.loc[g.energy.idxmin()]

    runtime_penalty = 100 * (en.runtime / rt.runtime - 1)
    energy_penalty = 100 * (rt.energy / en.energy - 1)

    session_runtime_support = 0
    session_energy_support = 0

    for session in range(1, 6):
        sg = sessions[
            (sessions.workload == w)
            & (sessions["size"] == s)
            & (sessions.session == session)
        ]

        assert len(sg) == 18, (w, s, session)

        sr = sg.loc[sg.runtime.idxmin()]
        se = sg.loc[sg.energy.idxmin()]

        session_runtime_support += int(sr.platform == rt.platform)
        session_energy_support += int(se.platform == en.platform)

    holdout = all_cells[
        (all_cells.workload == w)
        & (all_cells["size"] == s)
    ].iloc[0]

    annex_rows.append({
        "workload": w,
        "size": s,
        "runtime_platform": rt.platform,
        "runtime_config": rt.configuration,
        "energy_platform": en.platform,
        "energy_config": en.configuration,
        "runtime_penalty_energy_choice_pct": runtime_penalty,
        "energy_penalty_runtime_choice_pct": energy_penalty,
        "runtime_platform_session_support": session_runtime_support,
        "energy_platform_session_support": session_energy_support,
        "conflict_holdout_folds": int(
            holdout.holdout_conflict_folds
        ),
        "unresolved_10pct": cell.unresolved_10pct,
        "unresolved_20pct": cell.unresolved_20pct,
    })

annex = pd.DataFrame(annex_rows)

assert len(annex) == 25

annex.round(4).to_csv(
    OUT / "uniform_conflict_details_25_cells.csv",
    index=False
)

print("\nPASS: Conflict detail table")
print("Conflict cells:", len(annex))
print("Output:", OUT / "uniform_conflict_details_25_cells.csv")
