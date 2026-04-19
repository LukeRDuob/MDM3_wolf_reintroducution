import pandas as pd
from WeeklyModel_ import WeeklySpeciesModel

wolf_numbers = [20, 40, 45, 50]


for wolf_n in wolf_numbers:
    all_results = []

    for run_id in range(20):
        model = WeeklySpeciesModel(max_steps=50000, init_total_wolves=wolf_n)

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
    combined.to_csv(f"Model_wolf{wolf_n}_20runs.csv", index=False)

    print(f"Model_wolf{wolf_n}_20runs.csv")