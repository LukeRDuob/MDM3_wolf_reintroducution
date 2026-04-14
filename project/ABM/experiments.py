"""
Parameter sweep for Scottish wolf reintroduction scenario.
Optimised for i7-13700 (24 threads).
"""
from mesa import batch_run
from Model import SpeciesModel
import pandas as pd
import time
import importlib
import WolfClass
importlib.reload(WolfClass)
from WolfClass import Wolf
import DeerClass
importlib.reload(DeerClass)
from DeerClass import Deer  
import Model
importlib.reload(Model)
from Model import SpeciesModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mesa

# ============================================================
# PHASE 1: Coarse sweep (step_size=1.0, ~10 year runs)
# ============================================================
print("=" * 60)
print("PHASE 1: Coarse parameter sweep")
print("=" * 60)

coarse_params = {
    "init_predators": [10, 15, 20, 30, 40],
    "init_deer": [1000],
    "max_steps": [50_000],        # ~10 years at step_size=1.0
    "step_size": [1.0],
    "energy_decrease": [0.0005, 0.001, 0.002, 0.003],
    "seed": range(5),             # 5 replicates
    "use_base": [False],
    "use_veg": [False],
    "use_boundary_conditions": [True],
}

n_combos = 5 * 4 * 5  # 100 runs
print(f"Running {n_combos} simulations across 20 cores...")

start = time.time()

results = batch_run(
    SpeciesModel,
    parameters=coarse_params,
    iterations=1,
    max_steps=50_000,
    number_processes=20,
    data_collection_period=-1,
    display_progress=True,
)

elapsed = time.time() - start
print(f"Phase 1 complete in {elapsed/60:.1f} minutes")

df = pd.DataFrame(results)
df.to_csv("coarse_sweep_results.csv", index=False)

# Summarise
summary = df.groupby(
    ["init_predators", "energy_decrease"]
).agg(
    mean_deer=("Deer", "mean"),
    std_deer=("Deer", "std"),
    mean_wolves=("Wolf", "mean"),
    std_wolves=("Wolf", "std"),
    extinctions=("Deer", lambda x: (x == 0).sum()),
).reset_index()

# Target: ~300 deer (3/km² × 100km²)
summary["distance_from_target"] = abs(summary["mean_deer"] - 300)
summary = summary.sort_values("distance_from_target")

print("\n" + "=" * 60)
print("TOP 10 PARAMETER COMBINATIONS (closest to 300 deer)")
print("=" * 60)
print(summary.head(10).to_string(index=False))

# ============================================================
# PHASE 2: Fine sweep around best parameters
# ============================================================
best = summary.iloc[0]
best_pred = int(best["init_predators"])
best_energy = float(best["energy_decrease"])

print(f"\nBest coarse params: pred={best_pred}, energy_dec={best_energy}")
print("Running fine sweep around these values...")

fine_params = {
    "init_predators": [max(5, best_pred - 5), best_pred, best_pred + 5],
    "init_deer": [1000],
    "max_steps": [200_000],       # ~10 years at step_size=0.25
    "step_size": [0.25],          # Full resolution
    "energy_decrease": [
        best_energy * 0.75,
        best_energy,
        best_energy * 1.25,
    ],
    "seed": range(10),            # More replicates for confidence
    "use_base": [False],
    "use_veg": [False],
    "use_boundary_conditions": [True],
}

n_fine = 3 * 3 * 10  # 90 runs
print(f"Running {n_fine} fine simulations...")

start = time.time()

fine_results = batch_run(
    SpeciesModel,
    parameters=fine_params,
    iterations=1,
    max_steps=200_000,
    number_processes=20,
    data_collection_period=1000,  # Collect every 1000 steps for time series
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

#save results to csv
fine_summary.to_csv("fine_sweep_summary.csv", index=False)