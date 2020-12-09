#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:30:28 2020

@author: tychobovenschen
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import stats
import cartopy.crs as ccrs
from matplotlib import ticker, cm
import cartopy.feature as cfeature
from global_land_mask import globe
import cartopy.mpl.ticker as cticker
import matplotlib.colors as colors
import pandas as pd
from scipy.fft import rfft, rfftfreq
from datetime import datetime
from datetime import timedelta
import pickle
pi=np.pi

def calc_diff(df, Nbin):
    """
    A function to calculation the diffusivity per bin according to Visser.
    INPUT:
        df: a dataframe with all data
    OUTPUT:
        Mean_diff: The diffusivity average over the bins
        tau: The correlation time scale, average over the bins
    """
    #%% Calculate diffusivities according to Visser:

    #Assign parameters
    n=2 #number of dimensions
    dt = 3600*6  #timestep in seconds
    #number of bins (in both x and y-direction) to average angles
    
    phi = np.cos(df['dangle']/360*2*pi)
    
    Mean_phi, xedges, yedges,_ = stats.binned_statistic_2d(df['lon'],df['lat'], phi,statistic='mean',bins=Nbin)
    
    #%%
    #average angles and velocities over bins:
    Mean_angles, xedges, yedges,_ = stats.binned_statistic_2d(df['lon'],df['lat'], df['dangle'],statistic='mean',bins=Nbin)
    Mean_vel, xedges, yedges, binnumber = stats.binned_statistic_2d(df['lon'],df['lat'], df['speed']/100,statistic='mean',bins=Nbin, expand_binnumbers=True)
    
    # rearrange the binnumber in  an array with length of data (C-like ordering)
    binnumber_new = np.zeros(len(df))
    for j in range(len(df)):
        binnumber_new[j] = (binnumber[1,j]-1)*Nbin+binnumber[0,j]-1
    
    #Create arrays or residual velocity and D
    vel_res = np.zeros(len(df))
    D =  np.zeros(len(df))
    velocity = np.array(df['speed']/100) #total velocity
    tau = dt/(1-Mean_phi)
    #calculate the residual velocity (subtract the mean velocity of the grid cell):
    for i in range(len(df)):
        vel_res[i] = velocity[i] - np.reshape(Mean_vel,-1, order='F')[int(binnumber_new[i])]
    #Calculate phi, tau and D
    for i in range(len(vel_res)):
        D[i] = 1/n * vel_res[i]**2 *np.reshape(tau,-1,order='F')[int(binnumber_new[i])] #The diffusivity

    
    Mean_diff, xedges, yedges, _ = stats.binned_statistic_2d(df['lon'],df['lat'], D,statistic='mean',bins=Nbin)
    return Mean_diff, tau , vel_res