#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:44:35 2020

@author: tychobovenschen
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import cartopy.crs as ccrs
from matplotlib import ticker, cm
import cartopy.feature as cfeature
from global_land_mask import globe
import cartopy.mpl.ticker as cticker
import matplotlib.colors as colors
import pandas as pd
from scipy.fft import rfft, rfftfreq
import pickle
from calc_angle import calc_angle
from calc_diff import calc_diff
from scipy import stats
from binned_statistic import binned_statistic_2d_new
from datetime import datetime
from datetime import timedelta
from reanalysisdata import reanalysis_meanvel
from plotonmap import plotonmap
from plot_angles import plot_angles
import xarray as xr
import tqdm
from residual_vel_eul import calc_residual_vel_eul

pi=np.pi
#Data paths:
Path_data = '/Users/tychobovenschen/Documents/MasterJaar2/Thesis/data/'
File_data = 'interpolated_gld.20201120_024210.txt'

#read data
df  = pd.read_csv(Path_data+File_data,sep='\s+')
#Read all gps IDs with GPS:
Gps_ids = pd.read_csv(Path_data+'gps_ids.dat')

#Filter out buoys without GPS:
df = df[df['id'].isin(Gps_ids['ID'])]

del(Gps_ids) # delete the list of GPS ids again
#%%   Calculate time difference at every step
cnt=0
df['datetime'] = df['date']+' ' +df['time']
datetimes = np.zeros(len(df)).astype(datetime)
for i in df['datetime']:
    datetimes[cnt] = datetime.strptime(i,'%Y-%m-%d %H:%M:%S')
    cnt+=1
timedeltas = np.zeros(len(datetimes))
for i in range(len(datetimes)-1):
    timedeltas[i] = (datetimes[i+1]-datetimes[i]).total_seconds()/3600
#Add the time differences to the original dataframe:
df['timedeltas']=timedeltas
df['datetime']=datetimes

#Make a numpy array of the data
data = np.array(df) #convert to numpy array

#split data for different buoys (and same buoys with time gap)
data_split = np.split(data,np.where((data[:,13]!=6))[0]+1)

#Filter out buoys with only 1 data point in the region
cnt = 0
for i in range(len(data_split)):
    if len(data_split[cnt])<5:
        del data_split[cnt]
        cnt-=1
    cnt+=1
#Update the dataframe of all data:
df = pd.DataFrame(np.concatenate(data_split),columns=df.columns, dtype='object')
df=df.convert_dtypes()
del(timedeltas,data) #delete unused variables
#%% Plot the buoy trajectories

colorss = ['r', 'b', 'y','g','orange'] #Colors used for the trajectories
        
#Create plot for the trajectories
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,12), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([-65, -45, 55, 65])
for i in range(len(data_split)):
    ax.plot(data_split[i][:,4],data_split[i][:,3],lw=1,color=colorss[np.mod(i,len(colorss))],transform=ccrs.PlateCarree())
ax.coastlines(resolution='50m') 
ax.set_xticks([-65, -60, -55, -50, -45])
plt.title('Particle trajectories in Labrador Sea',fontsize=16)
plt.xlabel('Longitude (degrees)')
plt.grid()
plt.ylabel('Latitude (degrees')
ax.set_yticks([55, 60, 65])
plt.show()

#%%
#Calculate angles between different data points: (see function for documentation)
angle, dangle, dist = calc_angle(data_split)

#%% Plot the angles 

# plot_angles(dangle,angle,dist) 
#%%
#stack all dangles and add them to the dataframe of all data, remove nans and infs
dangle_resh = [None]*len(dangle)
dist_resh = [None]*len(dangle)
for i in range(len(dangle)):
    dangle_resh[i] = np.insert(dangle[i],0,np.nan)
    dangle_resh[i] = np.append(dangle_resh[i],np.nan)
    dist_resh[i] = np.insert(dist[i],0,np.nan)
dangle_resh= pd.DataFrame(np.concatenate(dangle_resh))
dist_resh= pd.DataFrame(np.concatenate(dist_resh))
df['dangle']=dangle_resh
df['dist']=dist_resh
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
# df.to_pickle(Path_data+'df.pkl')
#%% Calculate diffusivities according to Visser:

#Assign parameters
n=2 #number of dimensions
dt = 3600*6  #timestep in seconds
Nbin = 40 #number of bins (in both x and y-direction) to average angles

# phi = np.cos(df['dangle']/360*2*pi)
Mean_diff, tau, vel_res = calc_diff(df, Nbin,mean_method='eulerian') # function for calculating diffusion according to visser 



#%% Plot u
#Count data points per gridcell:
counts_cell, xedges, yedges, binnumber = stats.binned_statistic_2d(df['lon'],df['lat'], df['vn']/100,statistic='count',bins=Nbin, expand_binnumbers=True)

# Filter out grid cells with less than 10 data points
for i in range(40):
    for j in range(40):
        if counts_cell[i,j]<10:
            Mean_diff[i,j]=np.nan


#%%PLOTTING
#Define a grid
x = np.linspace(-65,-45,Nbin)
y = np.linspace(55,65,Nbin)
X,Y = np.meshgrid(x,y)

plotonmap(X, Y, np.swapaxes(tau,0,1)/3600/24, 0, 6, 'Tau', 'Days')
plotonmap(X,Y,np.swapaxes(Mean_diff,0,1),0,8000,title='Diffusivities unfiltered',cbarlabel='$m^2/s$')

u_res =np.load(Path_data+'u_residual_eulerian.npy')
u_res_mean, xedges, yedges, binnumber = binned_statistic_2d_new(df['lon'],df['lat'], u_res,statistic='nanmean',bins=Nbin, expand_binnumbers=True)

#%%

# #%%
# ############## DAVIS:#############


# cnt=0
# timelapse=15*4 #number of datapoints difference (days*4)
# for i in range(len(data_split)):
#     if timelapse>len(data_split[cnt]):
#         del(data_split[cnt])
#         continue
#     cnt+=1
# #Update the dataframe of all data:
# df_davis = pd.DataFrame(np.concatenate(data_split),columns=df.columns[0:14], dtype='object')
# df_davis =df_davis.convert_dtypes()

# #%%
# dy_davis= [None]*len(data_split)
# dx_davis= [None]*len(data_split)
# dist_davis= [None]*len(data_split)
# for i in range(len(data_split)):
#     dy_davis[i]= np.zeros(len(data_split[i])-timelapse)
#     dx_davis[i]= np.zeros(len(data_split[i])-timelapse)
#     dist_davis[i]= np.zeros(len(data_split[i])-timelapse)
#     for j in range(len(data_split[i])-timelapse):
#         dy_davis[i][j] = (data_split[i][j+timelapse,3]-data_split[i][j,3])/360*40008e3
#         dx_davis[i][j] = (data_split[i][j+timelapse,4]-data_split[i][j,4])/360*40075e3*np.cos(data_split[i][j,3]/360*2*np.pi)
#         dist_davis[i][j]= np.sqrt(dy_davis[i][j]**2+dx_davis[i][j]**2)
# nans = [np.nan]*timelapse
# for i in range(len(dist_davis)):
#     dist_davis[i] = np.insert(dist_davis[i],0,nans)
# dist_davis = np.concatenate(dist_davis)
# df_davis['dist']=dist_davis
# df_davis.dropna(inplace=True)
# #%%
# Mean_dist, xedges, yedges, binnumber_davis = stats.binned_statistic_2d(df_davis['lon'],df_davis['lat'], df_davis['dist'],statistic='mean',bins=Nbin, expand_binnumbers=True)
# Mean_vel, xedges, yedges, _ = stats.binned_statistic_2d(df_davis['lon'],df_davis['lat'], df_davis['speed']/100,statistic='mean',bins=Nbin, expand_binnumbers=True)
# Mean_u, xedges, yedges, _ = stats.binned_statistic_2d(df_davis['lon'],df_davis['lat'], df_davis['ve']/100,statistic='mean',bins=Nbin, expand_binnumbers=True)
# Mean_v, xedges, yedges, _ = stats.binned_statistic_2d(df_davis['lon'],df_davis['lat'], df_davis['vn']/100,statistic='mean',bins=Nbin, expand_binnumbers=True)

# # rearrange the binnumber in  an array with length of data (C-like ordering)
# binnumber_new = np.zeros(len(df_davis))
# for j in range(len(df_davis)):
#     binnumber_new[j] = (binnumber_davis[1,j]-1)*Nbin+binnumber_davis[0,j]-1

# dist_davis=dist_davis[np.isfinite(dist_davis)]
# dist_res = np.zeros(len(dist_davis))
# vel_res = np.zeros(len(dist_davis))
# velocity = np.zeros(len(dist_davis))
# for i in range(len(df_davis)):
#     dist_res[i] = dist_davis[i] - np.reshape(Mean_dist,-1, order='F')[int(binnumber_new[i])]
# velocity = np.array(df_davis['speed'])/100
# #calculate the residual velocity (subtract the mean velocity of the grid cell):
# for i in range(len(df_davis)):
#     vel_res[i] = velocity[i] - np.reshape(Mean_vel,-1, order='F')[int(binnumber_new[i])]    


# Diff_davis = -vel_res* dist_res
# Mean_Diff_davis, xedges, yedges, _ = stats.binned_statistic_2d(df_davis['lon'],df_davis['lat'], Diff_davis,statistic='mean',bins=Nbin, expand_binnumbers=True)







# #%% Plot v
# plt.figure()
# ax1 = plt.axes(projection=ccrs.PlateCarree())   
# plt.contourf(X,Y,np.swapaxes(Mean_v,0,1), np.linspace(-0.5,0.5,11), cmap='bwr',extend='both', corner_mask=False, transform=ccrs.PlateCarree())
# plt.colorbar()
# ax1.coastlines(resolution='50m')
# plt.title('v')
# plt.xlabel('Longitude (degrees')
# plt.ylabel('Latitude (degrees')
# plt.show()
# #%% Plot diffusions
# plt.figure()
# ax1 = plt.axes(projection=ccrs.PlateCarree())   
# plt.contourf(X,Y,np.swapaxes(Mean_Diff_davis,0,1), cmap='rainbow',extend='both', corner_mask=False, transform=ccrs.PlateCarree())
# plt.colorbar()
# ax1.coastlines(resolution='50m')
# plt.title('Diffusion davis mean')
# plt.xlabel('Longitude (degrees')
# plt.ylabel('Latitude (degrees')
# plt.show()


#%% Fourier analysis

# eulerian_vel = np.mean(u_mean_eul, axis=1)
# eulerian_vel = np.mean(eulerian_vel, axis=1)

# plt.figure()
# plt.acorr(u_mean_eul[:,18,15], maxlags=20)
# #%%
# # f_cor = 2*7.2921e-5*np.sin(60/360*2*pi)
# # T_cor = 2*pi/f_cor/3600
# angle_alles = eulerian_vel
# T = 6*3600
# N = len(angle_alles)
# angles_fft = rfft(angle_alles)

# xf = rfftfreq(len(angle_alles),T)


# plt.figure()
# # plt.vlines([1/T_cor,1/12.4206],0,0.02,linestyles='dashed')
# plt.plot(xf[10:]*3600, 2.0/N * np.abs(angles_fft)[10:])
# # plt.xscale('log')
# # plt.xlim(50e-3,10**-1)
# # plt.ylim(0,0.001)
# plt.grid()
# plt.title('Fourier spectrum of velocity')
# # plt.ylim(0,0.01)
# # plt.xlim([0,0.0005])
# plt.xlabel('f($h^{-1}$)')
# plt.show()