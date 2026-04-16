# run_weekly_model.py

import pandas as pd
from WeeklyModel_copy import WeeklySpeciesModel

all_results = []

for run_id in range(5):
    model = WeeklySpeciesModel(max_steps=50000, seed=run_id)

    while model.running:
        model.step()

    results = model.datacollector.get_model_vars_dataframe().copy()
    results["Week"] = results.index
    results["Year"] = results["Week"] / 52
    results["Run"] = run_id
    all_results.append(results)

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv("weekly_model_multiple_runs.csv", index=False)

print("Saved results to weekly_model_multiple_runs.csv")