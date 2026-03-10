import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import os
import re
from rasterio.transform import from_bounds

# OS National Grid 500km squares and their layout
# Arranged as they appear on the map (North at top)
OS_GRID_500K = [
    ['SV', 'SW', 'SX', 'SY', 'SZ', 'TV'],
    ['SQ', 'SR', 'SS', 'ST', 'SU', 'TQ', 'TR'],
    ['SL', 'SM', 'SN', 'SO', 'SP', 'TL', 'TM'],
    ['SF', 'SG', 'SH', 'SJ', 'SK', 'TF', 'TG'],
    ['SA', 'SB', 'SC', 'SD', 'SE', 'TA', 'TB'],
    ['NV', 'NW', 'NX', 'NY', 'NZ', 'OV'],
    ['NQ', 'NR', 'NS', 'NT', 'NU', 'OQ'],
    ['NL', 'NM', 'NN', 'NO', 'NP', 'OL'],
    ['NF', 'NG', 'NH', 'NI', 'NJ', 'NK', 'OF'],
    ['NA', 'NB', 'NC', 'ND', 'NE', 'OA'],
    ['HV', 'HW', 'HX', 'HY', 'HZ'],
    ['HQ', 'HR', 'HS', 'HT', 'HU'],
    ['HL', 'HM', 'HN', 'HO', 'HP'],
]
# Reverse so index 0 = southernmost (easier for northing arithmetic)
OS_GRID_500K = list(reversed(OS_GRID_500K))

def get_deer_data(path):
    df = pd.read_csv(path, low_memory=False)
    print(df.head())
    print(df.columns)
    return df

def get_elevation_data(path):
    with rasterio.open(path) as src:
        dem = src.read(1)
        transform = src.transform
        print(f"  x pixel size: {transform.a:.1f}, y pixel size: {transform.e:.1f}")
        
        # OS Terrain 50 .asc stores columns East→West (negative x step)
        # Flip horizontally so West is left, East is right
        if transform.a < 0:
            dem = np.fliplr(dem)

    print("Shape:", dem.shape)
    return dem


def get_elevation_data_with_bounds(path):
    """Returns dem array and its (left, right, bottom, top) bounds in map coords."""
    with rasterio.open(path) as src:
        dem = src.read(1)
        bounds = src.bounds  # left, bottom, right, top
        res = src.res        # (pixel_height, pixel_width)
        print(f"  {os.path.basename(path)}: bounds={bounds}, res={res}")
    return dem, bounds


def plot_tile_elevation(dem):    
    plt.figure(figsize=(8, 6))
    plt.imshow(dem, cmap="terrain")
    plt.colorbar(label="Elevation (m)")
    plt.title("DEM tile - elevation")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def get_os_neighbour(prefix, easting_10km, northing_10km, dx, dy):
    """
    Given a 100km prefix and 10km tile numbers, find the neighbouring tile.
    Returns (new_prefix, new_easting, new_northing) or None if outside grid.
    
    easting_10km:  0-9 (the units digit of the tile number)
    northing_10km: 0-9 (the tens digit of the tile number)
    """
    # New 10km indices within the 100km square
    new_e = easting_10km  + dx
    new_n = northing_10km + dy

    new_prefix = prefix

    # Check if we've crossed into adjacent 100km square
    # Find current prefix position in grid
    prefix_col, prefix_row = None, None
    for row_i, row in enumerate(OS_GRID_500K):
        if prefix in row:
            prefix_row = row_i
            prefix_col = row.index(prefix)
            break

    if prefix_row is None:
        return None  # Unknown prefix

    # Handle crossing 100km easting boundary
    if new_e < 0:
        new_e += 10
        prefix_col -= 1
    elif new_e > 9:
        new_e -= 10
        prefix_col += 1

    # Handle crossing 100km northing boundary
    if new_n < 0:
        new_n += 10
        prefix_row -= 1
    elif new_n > 9:
        new_n -= 10
        prefix_row += 1

    # Bounds check
    if prefix_row < 0 or prefix_row >= len(OS_GRID_500K):
        return None
    if prefix_col < 0 or prefix_col >= len(OS_GRID_500K[prefix_row]):
        return None

    new_prefix = OS_GRID_500K[prefix_row][prefix_col]
    tile_num = new_e + new_n * 10
    return new_prefix, tile_num


def plot_elevation_georeferenced(tiles, prefix, num):
    """
    Plot each tile using its real geographic bounds (guarantees correct alignment regardless of internal pixel order)
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Compute overall extent
    all_bounds = [t[1] for t in tiles if t[1] is not None]
    xmin = min(b.left   for b in all_bounds)
    xmax = max(b.right  for b in all_bounds)
    ymin = min(b.bottom for b in all_bounds)
    ymax = max(b.top    for b in all_bounds)

    # Normalise elevations across all tiles for consistent colour scale
    all_vals = [t[0] for t in tiles if t[0] is not None]
    vmin = min(a.min() for a in all_vals)
    vmax = max(a.max() for a in all_vals)

    for dem, bounds, tile_name in tiles:
        if dem is None:
            continue
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        ax.imshow(dem, cmap="terrain", extent=extent,
                  vmin=vmin, vmax=vmax,
                  origin="upper",          # row 0 = top = North
                  aspect="equal")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='terrain',
                                norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Elevation (m)', shrink=0.7)

    ax.set_title(f"3x3 grid  (centre: {prefix}{num:02d})")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    plt.tight_layout()
    plt.show()


def build_elevation_array(tiles):
    """
    Stitch tiles into a single numpy array using their geographic bounds.
    Returns:
        dem_array  : 2D numpy array of elevation values
        transform  : rasterio Affine transform (maps pixel to coordinates)
        bounds     : (xmin, ymin, xmax, ymax) of the full array
    """
    valid_tiles = [(dem, bounds, name) for dem, bounds, name in tiles
                   if dem is not None and bounds is not None]

    if not valid_tiles:
        print("No valid tiles.")
        return None, None, None

    # Global bounds
    xmin = min(b.left   for _, b, _ in valid_tiles)
    xmax = max(b.right  for _, b, _ in valid_tiles)
    ymin = min(b.bottom for _, b, _ in valid_tiles)
    ymax = max(b.top    for _, b, _ in valid_tiles)

    # Get pixel resolution from first tile
    dem0, bounds0, _ = valid_tiles[0]
    tile_h, tile_w = dem0.shape
    res_x = (bounds0.right - bounds0.left) / tile_w 
    res_y = (bounds0.top   - bounds0.bottom) / tile_h

    # Full array dimensions
    total_w = int(round((xmax - xmin) / res_x))
    total_h = int(round((ymax - ymin) / res_y))

    # Initialise with NaN
    dem_array = np.full((total_h, total_w), np.nan, dtype=np.float32)

    for dem, bounds, tile_name in valid_tiles:
        # Compute pixel offsets for this tile within the full array
        col_off = int(round((bounds.left   - xmin) / res_x))
        row_off = int(round((ymax          - bounds.top)  / res_y))  # flip: top of array = ymax

        h, w = dem.shape
        dem_array[row_off:row_off+h, col_off:col_off+w] = dem
        print(f"  Placed {tile_name} at row={row_off} col={col_off}")

    # Build affine transform: top-left origin, positive x, negative y
    transform = from_bounds(xmin, ymin, xmax, ymax, total_w, total_h)

    print(f"\nFull array shape: {dem_array.shape}")
    print(f"Resolution: {res_x}m x {res_y}m")
    print(f"Bounds: E={xmin:.0f}-{xmax:.0f}, N={ymin:.0f}-{ymax:.0f}")
    print(f"Min elevation: {np.nanmin(dem_array):.1f}m")
    print(f"Max elevation: {np.nanmax(dem_array):.1f}m")

    return dem_array, transform, (xmin, ymin, xmax, ymax)

def plot_3x3_elevation_map(path):
    path = os.path.normpath(path)
    
    # Use regular expression to pick out correct tile(s)
    match = re.search(r"([a-z]{2})[\\/]+([A-Z]{2})([0-9]{2})\.asc$", path)
    if not match:
        print("Invalid path or filename format.")
        return None, None

    folder, prefix, num_str = match.groups()
    num = int(num_str)
    base_dir = os.path.dirname(os.path.dirname(path))

    # Extract 10km easting/northing digits from tile number
    easting_10km  = num % 10       # units digit
    northing_10km = num // 10      # tens digit

    tiles = []

    # Form grid
    for dy in [1, 0, -1]:
        for dx in [-1, 0, 1]:
            result = get_os_neighbour(prefix, easting_10km, northing_10km, dx, dy)
            if result is None:
                print(f"  Off-grid: dx={dx} dy={dy}")
                tiles.append((None, None, "OFF_GRID"))
                continue

            nb_prefix, nb_num = result
            nb_folder  = nb_prefix.lower()
            tile_name  = f"{nb_prefix}{nb_num:02d}.asc"
            tile_path  = os.path.join(base_dir, nb_folder, tile_name)

            print(f"  dy={dy:+d} dx={dx:+d} → {tile_name}: "
                  f"{'EXISTS' if os.path.exists(tile_path) else 'MISSING'}")

            if os.path.exists(tile_path):
                dem, bounds = get_elevation_data_with_bounds(tile_path)
                tiles.append((dem, bounds, tile_name))
            else:
                tiles.append((None, None, tile_name))

    # Build stitched array
    dem_array, transform, bounds = build_elevation_array(tiles)

    # Plot
    plot_elevation_georeferenced(tiles, prefix, num)

    return dem_array, transform 

# Save glen affric elevation data to csv
def save_el_data(el_arr, transform, name):    
    rows, cols = el_arr.shape
    records = []
    for row in range(rows):
        for col in range(cols):
            elev = el_arr[row, col]
            if not np.isnan(elev):
                # Convert pixel back to OS easting/northing using transform
                easting, northing = transform * (col + 0.5, row + 0.5)  # pixel centre
                records.append({
                    'easting(x)':   round(easting,  1),
                    'northing(y)':  round(northing, 1),
                    'elevation(z)': round(float(elev), 2)
                })

    df = pd.DataFrame(records)
    out_path = os.path.join('project', 'data','clean_data', f'{name}_elevation.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    print(df.head())



if __name__ == '__main__':
    # Example usage for HP50.asc
    # center_tile = os.path.join('project', 'data', 'elevation_data', 'hu', 'HU40.asc')
    # plot_3x3_elevation_map(center_tile)
    # path = r"elevation_data\nh\NH02.asc"  # Example tile
    # dem = get_elevation_data(path)
    # plot_tile_elevation(dem)

    # get_deer_data(r'deer_data\Deer_Counts_Deer_Groups.csv')

    # Possible center tiles for 3x3 plots
    study_tiles = {
        # 'cairngorms_centre':  os.path.join('project', 'data', 'elevation_data', 'nh', 'NH80.asc'),
        'glen_affric':        os.path.join('project', 'data', 'elevation_data', 'nh', 'NH12.asc')
        # 'knoydart':           os.path.join('project', 'data', 'elevation_data', 'nm', 'NM89.asc'),
        # 'torridon':           os.path.join('project', 'data', 'elevation_data', 'ng', 'NG95.asc'),
    }

    # Plot (and save) elevation data
    for name, tile in study_tiles.items():
        # Plot data
        print(f"\nPlotting: {name}")
        el_arr, transform = plot_3x3_elevation_map(tile)
        
        # Save elevation data
        save_el_data(el_arr, transform, name)
        