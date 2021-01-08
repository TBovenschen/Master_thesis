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

def calc_residual_vel_eul(df):
    #Name of paths and files
    Path_data = '/Users/tychobovenschen/Documents/MasterJaar2/Thesis/data/'
    reanalysis_file = 'global-reanalysis-phy-001-030-monthly_1610013590138.nc'
    analysis_file = 'global-analysis-forecast-phy-001-024-monthly_1607528507055.nc'
    
    
    reanalysis = xr.open_dataset(Path_data+reanalysis_file)
    analysis = xr.open_dataset(Path_data+analysis_file)
    

    analysis = analysis.sel(depth=13.6, method='nearest')
    reanalysis = reanalysis.squeeze(dim='depth')
    
    ds = reanalysis.combine_first(analysis)
    ds.to_netcdf(Path_data+'Mean_velocities_eulerian.nc')
    #gridsize where to interpolate to
    Nbin=40
    #Interpolate to new grid:
    x = np.linspace(-65,-45,2*Nbin)
    y = np.linspace(55,65,Nbin)
    ds = ds.interp(longitude=x,  latitude=y, method='linear')
    
    #Reset index of dataframe and create arrays:
    df.reset_index(drop=True,inplace=True)
    u_res =np.zeros(len(df))
    v_res =np.zeros(len(df))
    
    #Calculate the residual velocities
    for i in range(len(df)):
        u_res[i] = df['ve'][i]/100 - ds.uo.sel(time=df['time'][i],longitude=df['lon'][i], latitude=df['lat'][i], method='nearest')
        v_res[i] = df['vn'][i]/100 - ds.vo.sel(time=df['time'][i],longitude=df['lon'][i], latitude=df['lat'][i], method='nearest')
    np.save(Path_data+'u_residual_eulerian.npy', u_res)
    np.save(Path_data+'v_residual_eulerian.npy', v_res)
    return ds, u_res, v_res
    
