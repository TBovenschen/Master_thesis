#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 15:40:32 2020

@author: tychobovenschen
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs
from matplotlib import ticker, cm
import cartopy.feature as cfeature
from global_land_mask import globe
import cartopy.mpl.ticker as cticker
from scipy import interpolate
import matplotlib.colors as colors
# #COPERNICUS ANALYSIS DATA:
# https://resources.marine.copernicus.eu/?option=com_csw&view=order&record_id=eec7a997-c57e-4dfa-9194-4c72154f5cc5vv

def reanalysis_meanvel(Nbin):
    """ Function to read reanalyasis data and interpolate it to the same grid as the lagrangian data"""
    Path_data = '/Users/tychobovenschen/Documents/MasterJaar2/Thesis/data/'
    File_data = 'global-analysis-forecast-phy-001-024-monthly_1607528507055.nc'
    
    data = nc.Dataset(Path_data+File_data)
    
    
    
    lat= data['latitude'][:]
    lon= data['longitude'][:]
    depths = data['depth'][:]
    time = data['time'][:]
    v = data['vo'][:,9,:,:] #data, depth 9 is 13.8m 
    u = data['uo'][:,9,:,:]
    
    
    LON,LAT =np.meshgrid(lon,lat)
    LON = LON.flatten()
    LAT = LAT.flatten()
    
    x = np.linspace(-65,-45,Nbin)
    y = np.linspace(55,65,Nbin)
    X,Y = np.meshgrid(x,y)
    X1 = X.flatten()
    Y1 = Y.flatten()
    
    u_pertime = np.ma.zeros((len(time), len(y), len(x)))
    v_pertime = np.ma.zeros((len(time), len(y), len(x)))
    
    for i in range(len(time)):
        interp_u = interpolate.RegularGridInterpolator((lat,lon), u[i,:,:],fill_value=np.nan)
        interp_v = interpolate.RegularGridInterpolator((lat,lon), v[i,:,:],fill_value=np.nan)
        u_new = interp_u((Y1,X1),method='nearest')
        u_new = np.reshape(u_new,np.shape(X))
        v_new = interp_v((Y1,X1),method='nearest')
        v_new = np.reshape(v_new,np.shape(X))
        u_pertime[i,:,:] = u_new
        v_pertime[i,:,:] = v_new
        u_pertime[i,:,:].filled(np.nan)
        v_pertime[i,:,:].filled(np.nan)
    return u_pertime, v_pertime, time