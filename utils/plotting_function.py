import numpy as np
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pdb
def plot_geopotential_comparison(model_output, truth, lat, lon, 
                                  timesteps=None, save_path=None, 
                                  vmin=-300, vmax=300):
    """
    Plot geopotential height comparison between model output and ground truth.
    
    Parameters:
    -----------
    model_output : np.ndarray
        Model predictions with shape [time, lat, lon]
    truth : np.ndarray
        Ground truth data with shape [time, lat, lon]
    lat : np.ndarray
        Latitude values, shape [lat,]
    lon : np.ndarray
        Longitude values, shape [lon,]
    timesteps : list or None
        Specific timesteps to plot. If None, plots all timesteps
    save_path : str or None
        Path to save the figure. If None, displays interactively
    vmin, vmax : float
        Color scale limits for the plots
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure object
    """
    # Ensure inputs are numpy arrays
    model_output = np.array(model_output)
    truth = np.array(truth)
    
    # Validate shapes
    assert model_output.shape == truth.shape, \
        f"Shape mismatch: model {model_output.shape} vs truth {truth.shape}"
    assert model_output.shape[1:] == (len(lat), len(lon)), \
        f"Spatial dimensions don't match lat/lon: {model_output.shape[1:]} vs ({len(lat)}, {len(lon)})"
    
    # Create meshgrid
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    # Calculate difference
    difference = model_output - truth
    
    # Determine timesteps to plot
    if timesteps is None:
        timesteps = range(model_output.shape[0])
    
    # Hours for labeling (assumes 6-hourly data)
    hours = [18, 0, 6, 12]
    
    # Create subplots for specified timesteps
    n_times = len(timesteps)
    fig = plt.figure(figsize=(18, 5 * n_times))
    
    cmap = plt.get_cmap('coolwarm')
    
    for idx, i in enumerate(timesteps):
        # Model output
        ax1 = plt.subplot(n_times, 3, idx * 3 + 1, projection=ccrs.PlateCarree())
        ax1.set_global()
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS, linestyle=':')
        mesh1 = ax1.pcolormesh(lon2d, lat2d, model_output[i], cmap=cmap, 
                               shading='auto', vmin=vmin, vmax=vmax)
        day = i // 4 + 1
        hour = hours[i % 4]
        ax1.set_title(f'Model - Day {day} Hour {hour}', fontsize=12)
        plt.colorbar(mesh1, ax=ax1, orientation='horizontal', pad=0.05, 
                    aspect=30, label='Height (m)')
        
        # Ground truth
        ax2 = plt.subplot(n_times, 3, idx * 3 + 2, projection=ccrs.PlateCarree())
        ax2.set_global()
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS, linestyle=':')
        mesh2 = ax2.pcolormesh(lon2d, lat2d, truth[i], cmap=cmap, 
                               shading='auto', vmin=vmin, vmax=vmax)
        ax2.set_title(f'Truth - Day {day} Hour {hour}', fontsize=12)
        plt.colorbar(mesh2, ax=ax2, orientation='horizontal', pad=0.05, 
                    aspect=30, label='Height (m)')
        
        # Difference
        ax3 = plt.subplot(n_times, 3, idx * 3 + 3, projection=ccrs.PlateCarree())
        ax3.set_global()
        ax3.coastlines()
        ax3.add_feature(cfeature.BORDERS, linestyle=':')
        mesh3 = ax3.pcolormesh(lon2d, lat2d, difference[i], cmap='RdBu_r', 
                               shading='auto', vmin=-100, vmax=100)
        ax3.set_title(f'Difference - Day {day} Hour {hour}', fontsize=12)
        plt.colorbar(mesh3, ax=ax3, orientation='horizontal', pad=0.05, 
                    aspect=30, label='Difference (m)')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig


# Example usage:
if __name__ == "__main__":
    from netCDF4 import Dataset
    
    # Load data
    netcdfloc = "/eagle/datascience/vsastry/projects/LatentTwinShared/hgt.2024.nc"
    ds = Dataset(netcdfloc)
    lat = ds.variables['lat'][:]
    lon = ds.variables['lon'][:]
    lat = lat.filled(np.nan)
    lon = lon.filled(np.nan)
    np.save("lat.npy", lat)
    np.save("lon.npy",lon)
    pdb.set_trace() 
    hgt = np.array(ds.variables['hgt'][:, 0, :, :])
    # Simulate model output (you would replace this with actual model predictions)
    model_output = hgt + np.random.randn(*hgt.shape) * 20  # Add some noise
    
    # Plot first 4 timesteps as static comparison
    fig = plot_geopotential_comparison(
        model_output=model_output,
        truth=hgt,
        lat=lat,
        lon=lon,
        timesteps=[0, 4, 8, 12],  # Plot specific timesteps
        save_path=None  # Set to filename to save
    )
