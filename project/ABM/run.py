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

def run_experiment(params, max_steps=100000, n_proc = 1):

    # Creating batch runner to extract results.
    results = batch_run(
        SpeciesModel,
        parameters=params,
        max_steps=max_steps,
        iterations=1,
        data_collection_period = 1,
        number_processes=n_proc
    )

    df = pd.DataFrame(results)

    return df

def plot_wolf_population(df):

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Wolf Population", label="Wolves")
    plt.title("Population Dynamics of Wolves")
    plt.xlabel("Time Steps")
    plt.ylabel("Population Size")
    plt.legend()
    plt.grid()
    plt.savefig('wolf_population.png')

def plot_deer_population(df):

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Deer Population", label="Deer", color='orange')
    plt.title("Population Dynamics of Deer")
    plt.xlabel("Time Steps")
    plt.ylabel("Population Size")
    plt.legend()
    plt.grid()
    plt.savefig('deer_population.png')

def plot_deers_killed(df):

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y="Deer Hunted", label="Deer Hunted", color='red')
    plt.title("Number of Deer Hunted Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Deer Hunted")
    plt.legend()
    plt.grid()
    plt.savefig('deer_killed.png')



if __name__ == "__main__":

    n_proc = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

    max_steps = 100000

    params = {
            "max_steps": max_steps,
            "init_predators": 10, # changed to 6 as more realistic )
            "init_deer": 1000,  # approx 10 deer per km^2 (1000 deer)
            "height": 10,
            "width": 10,
            "step_size": 1/60,  # 1 hour per step
            "yearly_sunlight_hours": 5000,
            "seed": 42,
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

    results = run_experiment(params, max_steps, n_proc)

    end = time.time()
    print(f"Experiment completed in {end - start} seconds.")

    # plot results
    plot_wolf_population(results)
    plot_deer_population(results)
    plot_deers_killed(results)

    # take every 60th step to get hourly data 
    hourly_results = results[results["Step"] % 60 == 0]
    hourly_results.to_csv("hourly_results.csv", index=False)

