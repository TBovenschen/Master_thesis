#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:30:28 2020

@author: tychobovenschen
"""
#from IPython import get_ipython
#get_ipython().magic('reset -sf')
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import animation
from scipy import stats as stats
#import cartopy.crs as ccrs
#from matplotlib import ticker, cm
#import cartopy.feature as cfeature
#from global_land_mask import globe
#import cartopy.mpl.ticker as cticker
#import matplotlib.colors as colors
import pandas as pd
#from scipy.fft import rfft, rfftfreq
#from datetime import datetime
#from datetime import timedelta
from binned_statistic2 import binned_statistic_2d_new
#from reanalysisdata import reanalysis_meanvel
#import pickle
pi=np.pi

def calc_diff(df, Nbin, mean_method='eulerian'):
    """
    A function to calculation the diffusivity per bin according to Visser.
    INPUT:
        df: a dataframe with all data
        Nbin: numberes of bins in x and y direction
        mean_u: the mean velocity field in u
    OUTPUT:
        Mean_diff: The diffusivity average over the bins
        tau: The correlation time scale, average over the bins
    """
    #%% Calculate diffusivities according to Visser:
    Path_data = '/Users/tychobovenschen/Documents/MasterJaar2/Thesis/data/'
    df = pd.read_pickle(Path_data+'df.pkl')
    #Assign parameters
    n=2 #number of dimensions
    dt = 3600*6  #timestep in seconds
    #number of bins (in both x and y-direction) to average angles
    
    phi = np.cos(df.dangle/360*2*pi)
    Mean_phi, xedges, yedges,_ = stats.binned_statistic_2d(df['lon'],df['lat'], phi,statistic='mean',bins=Nbin)
    
    #%%
    #average angles and velocities over bins:
    Mean_angles, xedges, yedges,_ = stats.binned_statistic_2d(df['lon'],df['lat'], df['dangle'],statistic='mean',bins=Nbin)
    Mean_vel, xedges, yedges, binnumber = stats.binned_statistic_2d(df['lon'],df['lat'], df['speed']/100,statistic='mean',bins=Nbin, expand_binnumbers=True)

    binnumber_new = np.zeros(len(df))
    for j in range(len(df)):
        binnumber_new[j] = (binnumber[1,j]-1)*Nbin+binnumber[0,j]-1
    
    # #Create arrays or residual velocity and D
    D =  np.zeros(len(df))

    tau = dt/(1-Mean_phi)
    #     vel_res[i] = np.sqrt(u_res[i]**2+v_res[i]**2)
    #Calculate phi, tau and D
    if mean_method=='eulerian':
        u_res=pd.read_pickle(Path_data+'u_residual_eulerian.npy')
        v_res=pd.read_pickle(Path_data+'v_residual_eulerian.npy')
        vel_res = np.sqrt(u_res**2+v_res**2)
    for i in range(len(vel_res)):
        D[i] = 1/n * vel_res[i]**2 *np.reshape(tau,-1,order='F')[int(binnumber_new[i])] #The diffusivity

    Mean_diff, xedges, yedges, _ = binned_statistic_2d_new(df['lon'],df['lat'], D,statistic='nanmean',bins=Nbin)
    return Mean_diff, tau , vel_res