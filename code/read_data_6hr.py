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
import pickle
from calc_angle import calc_angle
from datetime import datetime
from datetime import timedelta
pi=np.pi

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
#%%   
df['timedeltas']=timedeltas


data = np.array(df) #convert to numpy array

#split data for different buoys (and same buoys with time gap)
# data_split = np.split(data,np.where((np.diff(data[:,0])!=0))[0]+1)
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
del(timedeltas,datetimes,data) #delete unused variables
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
i=81
fig = plt.figure()
ax=plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-65, -45, 55, 65])
ax.plot(data_split[i][:,4],data_split[i][:,3],lw=2,color=colorss[np.mod(i,len(colorss))],transform=ccrs.PlateCarree())
ax.scatter(data_split[i][0,4],data_split[i][0,3],transform=ccrs.PlateCarree())
ax.grid()
ax.coastlines() 
plt.xlim([-65,-45])
plt.ylim([55,65])
ax.set_xticks([-65, -60, -55, -50, -45])
plt.title('Particle trajectories in Labrador Sea',fontsize=16)
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees')
# for j, txt in enumerate(angle[i]):
#     ax.annotate(int(txt), (data_split[i][j,4],data_split[i][j,3]))
# for j, txt in enumerate(dangle[i]):
#     ax.annotate(int(txt), (data_split[i][j+1,4],data_split[i][j+1,3]-0.05))
ax.set_yticks([55, 60, 65])
plt.show()



#%%
#Calculate angles between different data points: (see function for documentation)
angle, dangle, dist = calc_angle(data_split)
#%% Plot the angles  
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

#%%
#stack all dangles and add them to the dataframe of all data, remove nans and infs
dangle_resh = [None]*len(dangle)
for i in range(len(dangle)):
    dangle_resh[i] = np.insert(dangle[i],0,np.nan)
    dangle_resh[i] = np.append(dangle_resh[i],np.nan)
dangle_resh= pd.DataFrame(np.concatenate(dangle_resh))
df['dangle']=dangle_resh
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)


#%% Calculate diffusivities according to Visser:
u= [None]*len(dangle)
v= [None]*len(dangle)
vel= [None]*len(dangle)

#Fill arrays for velocities
# for i in range(len(data_split)):
#     u[i] = data_split[i][1:-1,6]/100 #velocity in x-direction (m/s)
#     v[i] = data_split[i][1:-1,7]/100 #velocity in y-direction (m/s)
#     # vel[i] = data_split[i][1:-1,8]/100 #total velocity (m/s)

n=2 #number of dimensions
dt = 3600*6  #timestep in seconds
Nbin = 20 #number of bins to average angles



#average angles and velocities over bins:
Mean_angles, xedges, yedges,_ = stats.binned_statistic_2d(df['lon'],df['lat'], df['dangle'],statistic='mean',bins=Nbin)
Mean_vel, xedges, yedges, binnumber = stats.binned_statistic_2d(df['lon'],df['lat'], df['speed']/100,statistic='mean',bins=Nbin, expand_binnumbers=True)

# rearrange the binnumber in  an array with length of data
binnumber_new = np.zeros(len(df))
for j in range(len(df)):
    binnumber_new[j] = (binnumber[1,j]-1)*Nbin+binnumber[0,j]

#Create arrays or residual velocity, phi, tau and D
vel_res = np.zeros(len(df))
phi =  np.zeros(len(df))
tau  =  np.zeros(len(df))
D =  np.zeros(len(df))
velocity = np.array(df['speed']/100) #total velocity
#calculate the residual velocity (subtract the mean velocity of the grid cell):
for i in range(len(df)):
    vel_res[i] = velocity[i] - np.reshape(Mean_vel,-1, order='C')[int(binnumber_new[i])]

#Calculate phi, tau and D
for i in range(len(vel_res)):
    phi[i] = np.cos(np.reshape(Mean_angles,-1, order='C')[int(binnumber_new[i])]/360*2*np.pi) # The cosine of the angle
    tau[i] = dt/(1-phi[i])    #correlation time scale
    D[i] = 1/n * vel_res[i]**2 *dt/(1-phi[i]) #The diffusivity
np.nan_to_num(vel_res,copy=False)
Mean_vel_res, xedges, yedges, _ = stats.binned_statistic_2d(df['lon'],df['lat'], vel_res,statistic='mean',bins=Nbin, expand_binnumbers=True)
np.nan_to_num(D,copy=False)

Mean_diff, xedges, yedges, _ = stats.binned_statistic_2d(df['lon'],df['lat'], D,statistic='mean',bins=Nbin)

#%%
x = np.linspace(-65,-45,Nbin)
y = np.linspace(55,65,Nbin)
X,Y = np.meshgrid(x,y)

plt.figure()
ax1 = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(X,Y,np.swapaxes(Mean_vel,0 ,1), np.linspace(0,0.4,11),cmap='rainbow',extend='both', corner_mask=False, transform=ccrs.PlateCarree())
plt.colorbar()
ax1.coastlines(resolution='50m')
plt.title('Mean Velocities')
plt.xlabel('Longitude (degrees')
plt.ylabel('Latitude (degrees')
plt.show()

#%%
np.nan_to_num(tau,copy=False)
Mean_tau, xedges, yedges, _ = stats.binned_statistic_2d(df['lon'],df['lat'], tau,statistic='mean',bins=Nbin)

plt.figure()
ax1 = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(X,Y,np.swapaxes(Mean_tau,0 ,1),locator=ticker.LogLocator(),cmap='rainbow',extend='both', transform=ccrs.PlateCarree())
plt.colorbar()
ax1.coastlines(resolution='50m')
plt.title('Mean tau')
plt.xlabel('Longitude (degrees')
plt.ylabel('Latitude (degrees')
plt.show()


#%%
plt.figure()
ax1 = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(X,Y,np.swapaxes(Mean_angles, 0, 1),np.linspace(-15,15,21), cmap='bwr', corner_mask=True,extend='both', transform=ccrs.PlateCarree())
plt.colorbar()
ax1.coastlines(resolution='50m')
plt.title('Angles')
plt.xlabel('Longitude (degrees')
plt.ylabel('Latitude (degrees')
plt.show()
#%%
plt.figure()
ax1 = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(X,Y,np.swapaxes(Mean_diff,0,1),locator=ticker.LogLocator(), cmap='rainbow',extend='both', transform=ccrs.PlateCarree())
plt.colorbar()
ax1.coastlines(resolution='50m')
plt.title('Diffusivities')
plt.xlabel('Longitude (degrees')
plt.ylabel('Latitude (degrees')
plt.show()


#%% Fourier analysis
f_cor = 2*7.2921e-5*np.sin(60/360*2*pi)
T_cor = 2*pi/f_cor/3600
angle_alles = np.array(df['speed'])
T = 6*3600
N = len(angle_alles)
angles_fft = rfft(angle_alles)

xf = rfftfreq(len(angle_alles),T)


plt.figure()
plt.vlines([1/T_cor,1/12.4206],0,0.01,linestyles='dashed')
plt.plot(xf[10:]*3600, 2.0/N * np.abs(angles_fft)[10:])
# plt.xscale('log')
plt.grid()
plt.title('Fourier spectrum of velocity')
# plt.ylim(0,0.01)
# plt.xlim([0,0.0005])
plt.xlabel('f($h^{-1}$)')
plt.show()