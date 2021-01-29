#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:55:10 2021

@author: tychobovenschen
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
from binned_statistic import binned_statistic_2d_new
from scipy import stats
from datetime import datetime
from scipy import linalg
import tqdm
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator, LinearLocator)
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.collections import EllipseCollection


#Data paths:
Path_data = '/Users/tychobovenschen/Documents/MasterJaar2/Thesis/data/'

mean_vel_field = xr.open_dataset(Path_data+'Mean_velocities_eulerian_v2.nc')


land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['land'])



def plot_basic():
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-65,-45,55,65])
    ax.set_xticks(np.linspace(-65,-45,11), crs=ccrs.PlateCarree())
    ax.set_yticks(np.linspace(55,65,11), crs=ccrs.PlateCarree())
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['land'])

    ax.add_feature(land_50m,zorder=0)
    plt.annotate('Greenland',(-47,63),size=16)
    plt.annotate('Canada',(-64.5,57),size =16)
    ax.coastlines(resolution='50m')
    plt.ylabel('Degrees latitude', size=16)
    plt.xlabel('Degrees longitude', size=16)  
    return fig, ax
        
def plot_contour(X,Y,Z,zmin, zmax, title, cbarlabel, cmap='rainbow', grid=False):
    fig, ax = plot_basic()
    plt.title(title, size=24)
    if grid==True:
        ax.grid()
    plt.contourf(X,Y,Z, np.linspace(zmin,zmax,101),cmap=cmap,extend='both', corner_mask=False, transform=ccrs.PlateCarree())
    plt.colorbar(label=cbarlabel)
    
    
    
def plot_ellipse(eig_val,eig_vec,Nbin=20):
    fig, ax=plot_basic()
    x = np.linspace(-64.5,-45.5,Nbin)
    y = np.linspace(55.25,64.75,Nbin)
    X, Y = np.meshgrid(x, y)
    XY = np.column_stack((X.ravel(), Y.ravel()))
    #calculate largest and smalles eigenvalue
    index_major= abs(eig_val.labda).argmax(dim='i',skipna=False)
    index_minor= abs(eig_val.labda).argmin(dim='i',skipna=False)
    ells = EllipseCollection(eig_val.labda.isel(i=index_major)/8000,eig_val.labda.isel(i=index_minor)/8000,\
                         np.arctan2(eig_vec.mu.isel(i=index_minor,j=0),eig_vec.mu.isel(i=index_minor,j=1)).values/np.pi*180,units='x', offsets=XY,
                       transOffset=ax.transData, facecolors='None',edgecolors='tab:red', offset_position='screen')        
    bar = AnchoredSizeBar(ax.transData, size=1, label='8000 $m^2/s$', color = 'tab:red', loc=3, frameon=False)
    ax.add_artist(bar)
    ax.add_collection(ells)
    plt.show()