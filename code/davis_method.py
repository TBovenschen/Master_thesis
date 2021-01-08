#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 10:58:02 2021

@author: tychobovenschen
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import pickle
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
#Open the file with the buoy data:
df = pd.read_pickle(Path_data+'df.pkl')
df.reset_index(drop=True,inplace=True)

#Open file with mean flow displacements:
meanflowdisp=xr.open_dataset(Path_data+'meanflowdisp.nc') #Hourly trajectory data of particles release

df['mf_lon_dist']=(meanflowdisp.lon[:,15*24]-meanflowdisp.lon[:,0])*1.11e4 *np.cos(meanflowdisp.lat[:,0]*np.pi/180)
df['mf_lat_dist']=(meanflowdisp.lat[:,15*24]-meanflowdisp.lat[:,0])*1.11e4

mean_vel_field = xr.open_dataset(Path_data+'Mean_velocities_eulerian_v2.nc')

datastart=73000
dataend=datastart+60
plt.figure()
xr.plot.pcolormesh(mean_vel_field.isel(time=0).uo)
plt.scatter(meanflowdisp.lon[datastart:dataend,0:15*24+1],meanflowdisp.lat[datastart:dataend,0:15*24+1],s=1)
plt.plot(df.lon[datastart:dataend],df.lat[datastart:dataend],color='r')
plt.scatter(df.lon[datastart],df.lat[datastart],color='y',s=20)

#%%
#Make a numpy array of the data
data = np.array(df) #convert to numpy array

#split data for different buoys (and same buoys with time gap)
# data_split = np.split(data,np.where((data[:,13]!=6))[0]+1)# #%%
data_split=np.load('data_split_test.npz')
# ############## DAVIS:#############


cnt=0
timelapse=15*4 #number of datapoints difference (days*4)
for i in range(len(data_split)):
    if timelapse>len(data_split[cnt]):
        del(data_split[cnt])
        continue
    cnt+=1
#Update the dataframe of all data:
df_davis = pd.DataFrame(np.concatenate(data_split),columns=df.columns[0:18], dtype='object')
df_davis =df_davis.convert_dtypes()

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