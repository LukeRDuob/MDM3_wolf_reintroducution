"""
Parameter sweep for Scottish wolf reintroduction scenario.
HPC-safe (SLURM + multiprocessing compatible)
"""

from mesa import batch_run
from Model import SpeciesModel
import pandas as pd
import time
import os


# ============================================================
# Utility: get number of CPUs from SLURM
# ============================================================
def get_n_processes():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", 1))


# ============================================================
# Main experiment function
# ============================================================
def run_experiments():

    n_proc = get_n_processes()
    print(f"Using {n_proc} CPU cores")

    # ========================================================
    # PHASE 1: Coarse sweep
    # ========================================================
    print("=" * 60)
    print("PHASE 1: Coarse parameter sweep")
    print("=" * 60)

    coarse_params = {
        "init_predators": [10, 15, 20, 30, 40, 100],
        "init_deer": [1000],
        "max_steps": [100000],
        "step_size": [0.25],
        "energy_decrease": [0.0005, 0.001, 0.002, 0.003],
        "seed": range(5),
        "use_base": [False],
        "use_veg": [False],
        "use_boundary_conditions": [True],
    }

    start = time.time()

    coarse_results = batch_run(
        SpeciesModel,
        parameters=coarse_params,
        iterations=1,  # OK for now (just a warning)
        max_steps=50_000,
        number_processes=n_proc,   # ✅ matches SLURM
        data_collection_period=-1,
        display_progress=True,
    )

    elapsed = time.time() - start
    print(f"Phase 1 complete in {elapsed/60:.1f} minutes")

    df = pd.DataFrame(coarse_results)
    df.to_csv("coarse_sweep_results.csv", index=False)

    # Summary
    summary = df.groupby(
        ["init_predators", "energy_decrease"]
    ).agg(
        mean_deer=("Deer", "mean"),
        std_deer=("Deer", "std"),
        mean_wolves=("Wolf", "mean"),
        std_wolves=("Wolf", "std"),
        extinctions=("Deer", lambda x: (x == 0).sum()),
    ).reset_index()

    summary["distance_from_target"] = abs(summary["mean_deer"] - 300)
    summary = summary.sort_values("distance_from_target")

    summary.to_csv("coarse_sweep_summary.csv", index=False)

    # ========================================================
    # PHASE 2: Fine sweep
    # ========================================================
    best = summary.iloc[0]
    best_pred = int(best["init_predators"])
    best_energy = float(best["energy_decrease"])

    print(f"\nBest coarse params: pred={best_pred}, energy={best_energy}")
    print("Running fine sweep...")

    fine_params = {
        "init_predators": [max(5, best_pred - 5), best_pred, best_pred + 5],
        "init_deer": [1000],
        "max_steps": [100000],
        "step_size": [0.25],
        "energy_decrease": [
            best_energy * 0.75,
            best_energy,
            best_energy * 1.25,
        ],
        "seed": range(10),
        "use_base": [False],
        "use_veg": [False],
        "use_boundary_conditions": [True],
    }

    start = time.time()

    fine_results = batch_run(
        SpeciesModel,
        parameters=fine_params,
        iterations=1,
        max_steps=200_000,
        number_processes=n_proc,   # ✅ matches SLURM
        data_collection_period=1000,
        display_progress=True,
    )

    elapsed = time.time() - start
    print(f"Phase 2 complete in {elapsed/60:.1f} minutes")

    fine_df = pd.DataFrame(fine_results)
    fine_df.to_csv("fine_sweep_results.csv", index=False)

    fine_summary = fine_df.groupby(
        ["init_predators", "energy_decrease"]
    ).agg(
        mean_deer=("Deer", "mean"),
        std_deer=("Deer", "std"),
        mean_wolves=("Wolf", "mean"),
        extinctions=("Deer", lambda x: (x == 0).sum()),
    ).reset_index()

    fine_summary["distance_from_target"] = abs(fine_summary["mean_deer"] - 300)
    fine_summary = fine_summary.sort_values("distance_from_target")

    fine_summary.to_csv("fine_sweep_summary.csv", index=False)

    print("All simulations complete ✅")


# ============================================================
# ENTRY POINT (CRITICAL FOR HPC)
# ============================================================
if __name__ == "__main__":
    run_experiments()