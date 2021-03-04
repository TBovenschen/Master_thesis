#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 13:13:27 2020

@author: tychobovenschen
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import tqdm

def calc_residual_vel_eul(df):
    #Name of paths and files
    Path_data = '/Users/tychobovenschen/Documents/MasterJaar2/Thesis/data/'
    reanalysis_file = 'global-reanalysis-phy-001-030-monthly_1612428165441.nc'
    analysis_file = 'global-analysis-forecast-phy-001-024-monthly_1612428303941.nc'
    
    
    reanalysis = xr.open_dataset(Path_data+reanalysis_file)
    analysis = xr.open_dataset(Path_data+analysis_file)
    

    analysis = analysis.sel(depth=13.6, method='nearest')
    reanalysis = reanalysis.squeeze(dim='depth')
    
    ds = reanalysis.combine_first(analysis)
    ds.to_netcdf(Path_data+'Mean_velocities_eulerian.nc')
    
    # filter out grid cells where the ice concentration is more then 10%:
    # ds = ds.where((ds.siconc<0.1) | (xr.ufuncs.isnan(ds.siconc)))
    
    #gridsize where to interpolate to
    # Nbin=40
    # #Interpolate to new grid:
    # x = np.linspace(-65,-45,Nbin)
    # y = np.linspace(55,65,Nbin)
    # ds = ds.interp(longitude=x,  latitude=y, method='linear')
    
    #Reset index of dataframe and create arrays:
    df.reset_index(drop=True,inplace=True)
    # u_res =np.zeros(len(df))
    # v_res =np.zeros(len(df))
    
    #Calculate the residual velocities
    # for i in range(len(df)):
    # u_res = df['ve']/100 - ds.uo.sel(time=df['datetime'],longitude=df['lon'], latitude=df['lat'], method='nearest')
    # v_res = df['vn']/100 - ds.vo.sel(time=df['datetime'],longitude=df['lon'], latitude=df['lat'], method='nearest')
    u_res = df['ve']/100 - ds.uo.sel(time=xr.DataArray(df['datetime']),longitude=xr.DataArray(df['lon']), latitude=xr.DataArray(df['lat']), method='nearest')
    v_res = df['vn']/100 - ds.vo.sel(time=xr.DataArray(df['datetime']),longitude=xr.DataArray(df['lon']), latitude=xr.DataArray(df['lat']), method='nearest')
    u_res.to_pickle(Path_data+'u_residual_eulerian.npy')
    v_res.to_pickle(Path_data+'v_residual_eulerian.npy')
    # np.save(Path_data+'u_residual_eulerian_ice.npy', u_res, allow_pickle=True)
    # np.save(Path_data+'v_residual_eulerian_ice.npy', v_res, allow_pickle=True)
    return ds, u_res, v_res
    
