#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:59:58 2020

@author: tychobovenschen
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import datetime
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

f = nc.Dataset('data/gdp_interpolated_drifter_785a_706a_5aab.nc')
# f.set_auto_mask(False)
time = f.variables['time'][:]
lon = f.variables['longitude'][:]
lat = f.variables['latitude'][:] 
plt.figure()
plt.scatter(lon,lat,s=0.001)
plt.show()
   
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,12), subplot_kw={'projection': ccrs.PlateCarree()})

hist = ax.hist2d(lon,lat,bins=360,cmax = 500, cmap='rainbow')
ax.coastlines() 
fig.suptitle('Number of data points from april 2018 till april 2020',fontsize=18)
cbar = fig.colorbar(hist[3])
# cbar.set_ticks(np.linspace(0,2000,10))

plt.show()
#%%
test = f.variables['ID','longitude'][:]
counter=1
for i in range(len(test)-1):
    if test[i]!= test[i+1]:
        counter+=1
#%%
test= f.variables['deploy_date'][:]