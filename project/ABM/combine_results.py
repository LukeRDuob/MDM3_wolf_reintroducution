import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os


def combine_results(results_dir="results"):
    """Combine all per-seed CSV files into one DataFrame."""
    files = sorted(glob.glob(os.path.join(results_dir, "run_seed_*.csv")))

    if not files:
        raise FileNotFoundError(f"No result files found in {results_dir}/")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # Extract seed from filename
        seed = int(os.path.basename(f).split("_")[-1].replace(".csv", ""))
        df["seed"] = seed
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined {len(files)} files: {combined.shape}")
    return combined


def plot_population(df, column, label, color, filename):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Step", y=column, label=label, color=color)
    plt.title(f"Population Dynamics: {label}")
    plt.xlabel("Time Steps")
    plt.ylabel("Population Size")
    plt.legend()
    plt.grid()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


if __name__ == "__main__":

    results = combine_results()

    # Save combined
    results.to_csv("combined_results.csv", index=False)

    # Plot with confidence intervals across seeds
    plot_population(results, "Wolf", "Wolves", "blue", "wolf_population.png")
    plot_population(results, "Deer", "Deer", "orange", "deer_population.png")
    plot_population(results, "Deer Hunted", "Deer Hunted", "red", "deer_killed.png")