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
Path_data = '/Users/tychobovenschen/Documents/MasterJaar2/Thesis/data/'
reanalysis_file = 'global-reanalysis-phy-001-030-monthly_1607861506758.nc'
analysis_file = 'global-analysis-forecast-phy-001-024-monthly_1607528507055.nc'
Nbin=40
# df = pd.read_
reanalysis = xr.open_dataset(Path_data+reanalysis_file)
analysis = xr.open_dataset(Path_data+analysis_file)

analysis = analysis.sel(depth=13.6, method='nearest')
reanalysis = reanalysis.squeeze(dim='depth')

ds = reanalysis.combine_first(analysis)


x = np.linspace(-65,-45,Nbin)
y = np.linspace(55,65,Nbin)



# data.reindex(longitude=x,latitude=y,method='nearest')
ds = ds.interp(longitude=x,  latitude=y, method='linear')


# 
# test1 = data.isel(time=50,depth=0).to_dataframe()