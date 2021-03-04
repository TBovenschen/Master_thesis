#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 15:58:22 2021

@author: tychobovenschen
"""
from plotting_functions import *
import xarray as xr
from datetime import timedelta

# ds = xr.open_dataset('/Users/tychobovenschen/Documents/MasterJaar2/Thesis/scratch/diff_model_WT.nc')

# ds = ds.sel(longitude=slice(-65,-45), latitude=slice(55,65)).isel(depth=0)
# ds.plot.scatter(hue='Age' ,x='lon', y='lat')


test = timedelta(hours=3)