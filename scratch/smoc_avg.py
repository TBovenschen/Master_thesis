import numpy as np
import xarray as xr

path_dataSMOC = '/data/oceanparcels/input_data/CMEMS/GLOBAL_ANALYSIS_FORECAST_PHY_001_024_SMOC/'



ds = xr.open_mfdataset(path_dataSMOC+'SMOC_2018*.nc',combine='by_coords', join='left')
ds = ds.sel(longitude=slice(-65,-45),latitude=slice(55,65))

ds = ds.resample(time='1M', loffset='-15D').mean(dim='time')

ds.to_netcdf('/scratch/tycho/SMOC_monthly_mean_2018.nc')
