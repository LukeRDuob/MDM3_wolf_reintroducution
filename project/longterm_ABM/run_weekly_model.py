import pandas as pd
from WeeklyModel_ import WeeklySpeciesModel

all_results = []

for run_id in range(2):
    model = WeeklySpeciesModel(max_steps=50000)

    while model.running:
        model.step()

    results = model.datacollector.get_model_vars_dataframe().copy()
    results["Week"] = results.index * 0.5
    results["Year"] = results["Week"] / 52
    results["Run"] = run_id

    
    results["Extinction"] = model.stop_reason == "extinction"

    all_results.append(results)

    if model.stop_reason == "extinction":
        print(f"Model {run_id} went extinct at step {model.extinction_step}. Saved with extinction label.")
    else:
        print(f"Model {run_id} finished at max steps and was saved.")

combined = pd.concat(all_results, ignore_index=True)
combined.to_csv("weekly_model_multiple_runs.csv", index=False)

print("Saved results to weekly_model_multiple_runs.csv")