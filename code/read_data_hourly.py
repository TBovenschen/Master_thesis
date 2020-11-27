#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 12:44:54 2020

@author: tychobovenschen
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import datetime
from scipy import stats
import matplotlib.colors as colors
import pandas as pd
import cartopy.crs as ccrs
from matplotlib import ticker, cm
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from geopy import distance

Path_data = '/Users/tychobovenschen/Documents/MasterJaar2/Thesis/data/'

#Read the data
data = pd.read_csv(Path_data+'/Hourly_Data_Filtered.txt')
data_np = np.array(data) #Convert to numpy array
#split data for different buoys
data_split = np.split(data_np,np.where(np.diff(data_np[:,0])!=0)[0]+1)
#%%
colorss = ['r', 'b', 'y','g','k'] #Colors used for the trajectories
        
#Create plot for the trajectories
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,12), subplot_kw={'projection': ccrs.PlateCarree()})
for i in range(len(data_split)):
    ax.scatter(data_split[i][:,3],data_split[i][:,2],s=0.01,color=colorss[np.mod(i,len(colorss))])
ax.coastlines() 
ax.set_xticks([-65, -60, -55, -50, -45])
plt.title('Particle trajectories in Labrador Sea',fontsize=16)
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees')
ax.set_yticks([55, 60, 65])
plt.show()

#%%
i=144
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,12), subplot_kw={'projection': ccrs.PlateCarree()})
ax.scatter(data_split[i][:,3],data_split[i][:,2],s=5,color=colorss[np.mod(i,len(colorss))])
ax.coastlines() 
plt.xlim([-65,-45])
plt.ylim([55,65])
ax.set_xticks([-65, -60, -55, -50, -45])
plt.title('Particle trajectories in Labrador Sea',fontsize=16)
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees')
ax.set_yticks([55, 60, 65])
plt.show()
#%% Calculate angles:
#Create empty lists (for the different buoys)
dy= [None]*len(data_split)
dx= [None]*len(data_split)
angle= [None]*len(data_split)
dangle= [None]*len(data_split)

#For every buoy, calculate the dy and x and with that the angles at every timestep
for i in range(len(data_split)):
    dy[i]= np.zeros(len(data_split[i])-1)
    dx[i]= np.zeros(len(data_split[i])-1)
    angle[i]= np.zeros(len(data_split[i])-1)
    for j in range(len(data_split[i])-1):
        dy[i][j] = (data_split[i][j+1,2]-data_split[i][j,2])/360*40008e3
        dx[i][j] = (data_split[i][j+1,3]-data_split[i][j,3])/360*40075e3*np.cos(data_split[i][j,2]/360*2*np.pi)
        angle[i][j] = np.arctan(dy[i][j]/dx[i][j])/(2*np.pi)*360
#Calculate the difference of the angles between consecutive timestteps
for i in range(len(data_split)):
    dangle[i]= np.zeros(len(data_split[i])-2)
    for j in range(len(data_split[i])-2):        
        dangle[i][j]=angle[i][j+1]-angle[i][j]
        # dx[i][j] = distance.distance(((data_split[i][j+1,2],data_split[i][j,3]), (data_split[i][j,2],data_split[i][j,3]))).m
#%%
countzero=np.zeros(len(angle))
for i in range(len(angle)):
    countzero[i] = np.count_nonzero(dangle[i]==0)      
#%% Plot the angles  

plt.figure()
plt.hist(np.concatenate(dangle),bins=5000,stacked=True)
plt.title('Difference in angles between consecutive data points',fontsize=16)
plt.ylabel('Number of datapoints')
plt.ylim([0,1000])
plt.xlim([-30,30])
plt.xlabel('Angles (degrees)')
plt.grid()
plt.show()
#%%
means= np.zeros(len(dangle))
for i in range(len(dangle)):
    means[i] = np.mean(dangle[3])
plt.figure()
plt.hist(dangle[4],bins=200,range=[-50,30])
plt.show()
#%% Calculate diffusivities
u= [None]*len(dangle)
v= [None]*len(dangle)
vel= [None]*len(dangle)
phi = [None]*len(dangle)
tau = [None]*len(dangle)
D = [None]*len(dangle)
#Fill arrays for velocities
for i in range(len(data_split)):
    u[i] = data_split[i][1:-1,8]
    v[i] = data_split[i][1:-1,9]
n=2 #number of dimensions
dt = 3600  #timestep in seconds


for i in range(len(dangle)):
    vel[i] = np.sqrt(u[i]**2+v[i]**2) #The total velocity
    phi = np.cos(dangle[i]) # The cosine of the angle
    tau = dt/(1-phi)    #correlation time scale
    D[i] = 1/n * vel[i]**2 *dt/(1-phi) #The diffusivity

D_resh=[None]*len(dangle)
for i in range(len(dangle)):
    D_resh[i] = np.insert(D[i],0,np.nan)
    D[i] = np.append(D[i],np.nan)
D_resh = pd.DataFrame(np.concatenate(D_resh))

data['Diff']= D_resh
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
 #%%   
Nbin=20
Mean_diff, xedges, yedges, binnumber = stats.binned_statistic_2d(data['lon'],data['lat'], data['Diff'],statistic='mean',bins=Nbin)

x = np.linspace(295,315,Nbin)
y = np.linspace(55,65,Nbin)
X,Y = np.meshgrid(x,y)

plt.figure()
plt.contourf(X,Y,Mean_diff,locator=ticker.LogLocator())
plt.colorbar()
plt.title('Diffusivities')
plt.xlabel('Longitude (degrees')
plt.ylabel('Latitude (degrees')
plt.show()

