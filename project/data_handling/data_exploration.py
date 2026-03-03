import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio

def get_elevation_data(path):
    with rasterio.open(path) as src:
        dem = src.read(1)          # 2D numpy array of elevation

    print("Shape:", dem.shape)
    return dem

def get_deer_data(path):
    df = pd.read_csv(path, low_memory=False)
    print(df.head())
    print(df.columns)
    return df


def plot_elevation(dem):    
    plt.figure(figsize=(8, 6))
    plt.imshow(dem, cmap="terrain")
    plt.colorbar(label="Elevation (m)")
    plt.title("DEM tile - elevation")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    path = r"elevation_data\nh\NH02.asc"  # Example tile
    dem = get_elevation_data(path)
    plot_elevation(dem)

    # get_deer_data(r'deer_data\Deer_Counts_Deer_Groups.csv')