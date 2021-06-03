#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 16:20:59 2021

@author: tychobovenschen
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm
from plotting_functions import plot_basicmap
ds = xr.open_dataset('/Users/tychobovenschen/Documents/MasterJaar2/Thesis/data/gebco_2020_n65.0_s55.0_w-65.0_e-45.0.nc')

#%%

plot_basicmap()
ds.elevation.plot(vmax=0,cmap=cm.batlow)
