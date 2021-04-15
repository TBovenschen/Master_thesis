#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:58:22 2021

@author: tychobovenschen
"""
from plotting_functions import *
import xarray as xr
import matplotlib.pyplot as plt
from datetime import timedelta

ds = xr.open_dataset('/Users/tychobovenschen/Documents/MasterJaar2/Thesis/data/Mean_velocities_eulerian.nc')

# ds = ds.sel(longitude=slice(-65,-45), latitude=slice(55,65)).isel(depth=0)
# ds.plot.scatter(hue='Age' ,x='lon', y='lat')
ds = ds.interp(longitude=np.linspace(-65,-45,50), latitude = np.linspace(55,65,50), method = 'linear')

# test = timedelta(hours=3)
# plt.figure()
plot_basicmap()
test = plt.quiver(ds.longitude, ds.latitude, ds.uo.mean(dim='time'), ds.vo.mean(dim='time'), scale=0.4, scale_units='inches')

# test = plt.quiver(ds.longitude, ds.latitude, ds.uo.isel(time=0), ds.vo.isel(time=50), scale=0.4, scale_units='inches')
quiversize=0.5
plt.quiverkey(test, 0.05,0.05,quiversize,str(quiversize)+' m\s')
plt.title('Mean velocities in the Labrador Sea', size=20)
plt.show()