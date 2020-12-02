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


Path_data = '/Users/tychobovenschen/Documents/MasterJaar2/Thesis/data/'
File_data = 'interpolated_gld.20201120_024210.txt'

df = pd.read_pickle(Path_data+'df.pkl')
dangle = np.load(Path_data+'dangles.npy',allow_pickle=True)
data_split=np.load(Path_data+'data_split.npy',allow_pickle=True)
#%% Calculate diffusivities
u= [None]*len(dangle)
v= [None]*len(dangle)
vel= [None]*len(dangle)
phi = [None]*len(dangle)
tau = [None]*len(dangle)
D = [None]*len(dangle)
#Fill arrays for velocities
for i in range(len(data_split)):
    u[i] = data_split[i][1:-1,6]/100
    v[i] = data_split[i][1:-1,7]/100
    vel[i] = data_split
n=2 #number of dimensions
dt = 3600*6  #timestep in seconds

for i in range(len(dangle)):
    vel[i] = np.zeros(len(u[i]))
    for j in range(len(u[i])):
        vel[i][j] = np.sqrt(u[i][j]**2+v[i][j]**2) #The total velocity


for i in range(len(dangle)):
    phi = np.cos(dangle[i]/360*2*np.pi) # The cosine of the angle
    tau = dt/(1-phi)    #correlation time scale
    D[i] = 1/n * vel[i]**2 *dt/(1-phi) #The diffusivity


dangle_resh = [None]*len(dangle)
D_resh=[None]*len(dangle)
for i in range(len(dangle)):
    D_resh[i] = np.insert(D[i],0,np.nan)
    dangle_resh[i] = np.insert(dangle[i],0,np.nan)
    D_resh[i]=np.append(D_resh[i],np.nan)
    dangle_resh[i] = np.append(dangle_resh[i],np.nan)
D_resh = pd.DataFrame(np.concatenate(D_resh))
dangle_resh= pd.DataFrame(np.concatenate(dangle_resh))
df['dangle']=dangle_resh
df['Diff']= D_resh
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

 #%%   
Nbin=40
# Mean_diff, xedges, yedges, binnumber = stats.binned_statistic_2d(df['lon'],df['lat'], df['Diff'],statistic='mean',bins=Nbin)
Mean_angles, xedges, yedges, binnumber = stats.binned_statistic_2d(df['lon'],df['lat'], df['dangle'],statistic='mean',bins=Nbin)
Mean_vel, xedges, yedges, binnumber = stats.binned_statistic_2d(df['lon'],df['lat'], df['speed']/100,statistic='mean',bins=Nbin)

mean_diff = np.zeros(np.shape(Mean_angles))
for i in range(len(Mean_angles)):
    phi = np.cos(Mean_angles[i]/360*2*np.pi) # The cosine of the angle
    tau = dt/(1-phi)    #correlation time scale
    mean_diff[i,:] = 1/n * Mean_vel[i]**2 *dt/(1-phi) #The diffusivity

x = np.linspace(295,315,Nbin)
y = np.linspace(55,65,Nbin)
X,Y = np.meshgrid(x,y)