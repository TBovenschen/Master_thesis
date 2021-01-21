#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 16:55:40 2020

@author: tychobovenschen
"""
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import numpy as np
from matplotlib import ticker, cm
import matplotlib.pyplot as plt
import cartopy.mpl.ticker as cticker
import matplotlib.colors as colors

def plotonmap(X,Y,Z,zmin, zmax, title, cbarlabel, cmap='rainbow'):
    """A function for plotting on a map with projection PlateCarree"""
    plt.figure()
    ax1 = plt.axes(projection=ccrs.PlateCarree())   
    plt.contourf(X,Y,Z, np.linspace(zmin,zmax,101),cmap=cmap,extend='both', corner_mask=False, transform=ccrs.PlateCarree())
    plt.colorbar(label=cbarlabel)
    ax1.coastlines(resolution='50m')
    plt.title(title)
    plt.xlabel('Longitude (degrees')
    plt.ylabel('Latitude (degrees')
    plt.show()