from WolfClass import Wolf
from DeerClass import Deer  
from Model import SpeciesModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from mesa import batch_run
import os
import sys

def run_experiment(params, max_steps=100000, n_proc = 1, n_iterations=5):

    # Creating batch runner to extract results.
    results = batch_run(
        SpeciesModel,
        parameters=params,
        max_steps=max_steps,
        iterations= n_iterations,
        data_collection_period = 120,
        number_processes=n_proc
    )

    df = pd.DataFrame(results)

    return df

def plot_wolf_population(df):

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Wolf", label="Wolves")
    plt.title("Population Dynamics of Wolves")
    plt.xlabel("Time Steps")
    plt.ylabel("Population Size")
    plt.legend()
    plt.grid()
    plt.savefig('wolf_population.png')
    plt.close()

def plot_deer_population(df):

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Deer", label="Deer", color='orange')
    plt.title("Population Dynamics of Deer")
    plt.xlabel("Time Steps")
    plt.ylabel("Population Size")
    plt.legend()
    plt.grid()
    plt.savefig('deer_population.png')
    plt.close()
def plot_deers_killed(df):

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Deer Hunted", label="Deer Hunted", color='red')
    plt.title("Number of Deer Hunted Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Deer Hunted")
    plt.legend()
    plt.grid()
    plt.savefig('deer_killed.png')
    plt.close()


def run_single(params, max_steps):
    """Run a single model instance."""
    model = SpeciesModel(**params)
    
    # Manual stepping with periodic data collection
    for step in range(max_steps):
        model.step()
        if model.running is False:
            break

    return model.datacollector.get_model_dataframe()

def old_main():
    n_proc = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    
    days = 60 
    max_steps = days*(60*24)

    # Seeds and iterations
    n_iterations = 16  # matches the number of cores
    seeds = list(range(n_iterations))
    params = {
            "max_steps": max_steps,
            "init_predators": 10, # changed to 6 as more realistic )
            "init_deer": 1000,  # approx 10 deer per km^2 (1000 deer)
            "height": 10,
            "width": 10,
            "seed": seeds,
            "step_size": 1/60,  # 1 hour per step
            "yearly_sunlight_hours": 5000,
            "predator": 'Wolf',  # Helper attribute to avoid imports when accessing agent type
            "energy_decrease": 0.002,  # Energy decrease parameter 
            "pack_limit": 12,  # packs will split if too large 
            
            # Options to control complexity of the model
            "use_base": False,
            "use_pack_dynamics": True,
            "use_random_movement": False,
            "use_veg": False,
            "given_positions": False, # whether to use random positions or pre-chosen positions (for testing purposes)
            "use_boundary_conditions": True, # whether to use boundary conditions (reflecting off walls) or toroidal space
        }

    start = time.time()

    results = run_experiment(params, max_steps, n_proc, n_iterations=n_iterations)

    end = time.time()
    print(f"Experiment completed in {end - start} seconds.")

    # plot results
    plot_wolf_population(results)
    plot_deer_population(results)
    plot_deers_killed(results)

    # hourly_results = results[results["Step"] % 60 == 0]
    results.to_csv("hourly_results.csv", index=False)

def new_main():
    # Get array task ID from SLURM (this is the seed)
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))

    days = 60
    max_steps = days * (60 * 24)

    params = {
        "max_steps": max_steps,
        "init_predators": 10,
        "init_deer": 1000,
        "height": 10,
        "width": 10,
        "seed": task_id,                # ← Each array task gets a unique seed
        "step_size": 1 / 60,
        "yearly_sunlight_hours": 5000,
        "predator": 'Wolf',
        "energy_decrease": 0.002,
        "pack_limit": 12,
        "use_base": False,
        "use_pack_dynamics": True,
        "use_random_movement": False,
        "use_veg": False,
        "given_positions": False,
        "use_boundary_conditions": True,
    }

    print(f"Task {task_id}: Starting with seed={task_id}")
    start = time.time()

    df = run_single(params, max_steps)

    end = time.time()
    print(f"Task {task_id}: Completed in {end - start:.1f} seconds.")

    # Save per-task results
    os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/run_seed_{task_id}.csv")
    print(f"Task {task_id}: Saved results/run_seed_{task_id}.csv")


if __name__ == "__main__":
    old_main()
    new_main()
    