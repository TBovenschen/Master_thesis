#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:55:10 2021

@author: tychobovenschen
"""


#import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
#import matplotlib.ticker as mticker
import cartopy.feature as cfeature
from scipy import stats
#from datetime import datetime
#from scipy import linalg
#import tqdm
#from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
#                               AutoMinorLocator, LinearLocator)
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.collections import EllipseCollection


#Data paths:
Path_data = '/Users/tychobovenschen/Documents/MasterJaar2/Thesis/data/'

# mean_vel_field = xr.open_dataset(Path_data+'Mean_velocities_eulerian_v2.nc')


land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['land'])
def plot_basicmap():
    """A function for plotting a map of the labrador sea with land and labels"""
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([-65,-45,55,65])
    ax.set_xticks(np.linspace(-65,-45,11), crs=ccrs.PlateCarree())
    ax.set_yticks(np.linspace(55,65,11), crs=ccrs.PlateCarree())
    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                        edgecolor='face',
                                        facecolor=cfeature.COLORS['land'])

    ax.add_feature(land_50m,zorder=0)
    plt.annotate('Greenland',(-49,63),size=16)
    plt.annotate('Canada',(-64.5,57),size =16)
    ax.coastlines(resolution='50m')
    plt.ylabel('Degrees latitude', size=16)
    plt.xlabel('Degrees longitude', size=16)  
    return fig, ax

def plot_pcolormesh(X,Y,Z,zmin, zmax, title=None, cbarlabel=None, cmap='rainbow', grid=False):
    """A function for making a contourplot of the labrador sea"""
    fig, ax = plot_basicmap()
    plt.title(title, size=24)
    if grid==True:
        ax.grid()
    plt.pcolormesh(X,Y,Z, vmin=zmin, vmax=zmax, cmap=cmap, transform=ccrs.PlateCarree())
    plt.colorbar(label=cbarlabel)

def plot_angles(dangle,angle,dist):
    """A function for plotting the angles made by drifters in a histogram"""
    mean_angle = np.mean(np.concatenate(dangle))
    skew_angle = stats.skew(np.concatenate(dangle))
    countzero=np.zeros(len(angle))
    for i in range(len(angle)):
        countzero[i] = np.count_nonzero(dist[i]<100)
        # dangle[i] = dangle[i][~np.isnan(dangle[i])]
    plt.figure()
    plt.hist(np.concatenate(dangle)[~np.isnan(np.concatenate(dangle))],bins=200,range=[-180,180], stacked=True)
    plt.title('Difference in angles between consecutive data points',fontsize=16)
    plt.ylabel('Number of datapoints')
    plt.xlabel('Angles (degrees)')
    plt.text(-150,1500, 'Skewness = '+str(skew_angle)[:-14] +'\n'+ 'Mean angle = '+str(mean_angle)[:-14])
    # plt.xlim([-160,-150])
    plt.grid()
    plt.show()



        
def plot_contour(X,Y,Z,zmin, zmax, title=None, cbarlabel=None, cmap='rainbow', grid=False):
    """A function for making a contourplot of the labrador sea"""
    fig, ax = plot_basicmap()
    plt.title(title, size=24)
    if grid==True:
        ax.grid()
    plt.contourf(X,Y,Z, levels=np.linspace(zmin,zmax,101) ,cmap=cmap,extend='both', corner_mask=False, transform=ccrs.PlateCarree())
    plt.colorbar(label=cbarlabel)
    
    
    
def plot_ellipse(eig_val,eig_vec,Nbin=20):
    """A function for plotting ellipses of the diffusivity tensor"""
    fig, ax=plot_basicmap()
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
    plt.title('Anisotropy of the diffusivity tensor',size=24)
    ax.add_collection(ells)
    plt.show()