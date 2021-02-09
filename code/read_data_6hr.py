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
import scipy.stats as stats
import cartopy.crs as ccrs
import pandas as pd
#import pickle
from calc_angle import calc_angle
from calc_diff import calc_diff
# from binned_statistic import binned_statistic_2d_new
from binned_statistic2 import binned_statistic_2d_new
from datetime import datetime
#from datetime import timedelta
#from reanalysisdata import reanalysis_meanvel
import xarray as xr
#import tqdm
#from residual_vel_eul import calc_residual_vel_eul
from plotting_functions import *
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
fig,ax=plot_basicmap()
for i in range(len(data_split)):
    ax.plot(data_split[i][:,4],data_split[i][:,3],lw=1,color=colorss[np.mod(i,len(colorss))],transform=ccrs.PlateCarree())
plt.title('Drifter trajectories in Labrador Sea',fontsize=16)

#%%
#Calculate angles between different data points: (see function for documentation)
angle, dangle, dist = calc_angle(data_split)

#%% Plot the angles 

plot_angles(dangle,angle,dist) 
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

# ds, u_res, v_res = calc_residual_vel_eul(df)
#%%
Path_data = '/Users/tychobovenschen/Documents/MasterJaar2/Thesis/data/'
reanalysis_file = 'global-reanalysis-phy-001-030-monthly_1612428165441.nc'
analysis_file = 'global-analysis-forecast-phy-001-024-monthly_1612428303941.nc'


reanalysis = xr.open_dataset(Path_data+reanalysis_file)
analysis = xr.open_dataset(Path_data+analysis_file)


analysis = analysis.sel(depth=13.6, method='nearest')
reanalysis = reanalysis.squeeze(dim='depth')

ds = reanalysis.combine_first(analysis)
ds.to_netcdf(Path_data+'Mean_velocities_eulerian.nc')

# filter out grid cells where the ice concentration is more then 10%:
# ds = ds.where((ds.siconc<0.1) | (xr.ufuncs.isnan(ds.siconc)))


#gridsize where to interpolate to
Nbin=40
#Interpolate to new grid:
x = np.linspace(-65,-45,Nbin)
y = np.linspace(55,65,Nbin)
ds = ds.interp(longitude=x,  latitude=y, method='linear')

#Reset index of dataframe and create arrays:
df.reset_index(drop=True,inplace=True)
u_res =np.zeros(len(df))
v_res =np.zeros(len(df))

#Calculate the residual velocities
# for i in range(len(df)):
u_res = df['ve']/100 - ds.uo.sel(time=xr.DataArray(df['datetime']),longitude=xr.DataArray(df['lon']), latitude=xr.DataArray(df['lat']), method='nearest')
v_res = df['vn']/100 - ds.vo.sel(time=xr.DataArray(df['datetime']),longitude=xr.DataArray(df['lon']), latitude=xr.DataArray(df['lat']), method='nearest')
# np.save(Path_data+'u_residual_eulerian.npy', u_res, allow_pickle=False)
# np.save(Path_data+'v_residual_eulerian.npy', v_res, allow_pickle=False)
u_res.to_pickle(Path_data+'u_residual_eulerian.npy')
v_res.to_pickle(Path_data+'v_residual_eulerian.npy')

print(np.count_nonzero(~np.isfinite(u_res)))

# u_res=pd.read_pickle(Path_data+'u_residual_eulerian.npy')

#%% Calculate diffusivities according to Visser:

#Assign parameters
n=2 #number of dimensions
dt = 3600*6  #timestep in seconds
Nbin = 20 #number of bins (in both x and y-direction) to average angles

# phi = np.cos(df['dangle']/360*2*pi)
Mean_diff, tau, vel_res = calc_diff(df, Nbin,mean_method='eulerian') # function for calculating diffusion according to visser 

Mean_diff_visser = np.swapaxes(Mean_diff,0,1)

#%%PLOTTING
#Define a grid
x = np.linspace(-65,-45,Nbin)
y = np.linspace(55,65,Nbin)
X,Y = np.meshgrid(x,y)

plot_contour(X, Y, np.swapaxes(tau,0,1)/3600/24, 0, 4, title='Tau', cbarlabel= 'Days')
plot_contour(X,Y,np.swapaxes(Mean_diff,0,1),0,8000, title='Diffusivities Visser method',cbarlabel='$m^2/s$',cmap='rainbow')

# u_res =np.load(Path_data+'u_residual_eulerian.npy')
# u_res_mean, xedges, yedges, binnumber = binned_statistic_2d_new(df['lon'],df['lat'], u_res,statistic='nanmean',bins=Nbin, expand_binnumbers=True)



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