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
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
import cartopy.feature as cfeature
from binned_statistic import binned_statistic_2d_new
from scipy import stats
from datetime import datetime
from scipy import linalg
import tqdm
from plotting_functions import *


"""A script for calculating the diffusivity tensor according to the method of
From the diffusivity the symmetric part is taken and from this the eigenvalues and vectors
are calculated"""

pi=np.pi
#Data paths:
Path_data = '/Users/tychobovenschen/Documents/MasterJaar2/Thesis/data/'
File_data = 'interpolated_gld.20201120_024210.txt'

#Open the file with the buoy data:
df = pd.read_pickle(Path_data+'df.pkl')
df.reset_index(drop=True,inplace=True)

#Open file with mean flow displacements:
meanflowdisp=xr.open_dataset(Path_data+'meanflowdisp.nc') #Hourly trajectory data of particles release

#Add the mean flow distance to the dataframe (and convert to meters):
df['mf_lon_dist']=(meanflowdisp.lon[:,15*24]-meanflowdisp.lon[:,0])*1.11e5 *np.cos(meanflowdisp.lat[:,0]*np.pi/180)
df['mf_lat_dist']=(meanflowdisp.lat[:,15*24]-meanflowdisp.lat[:,0])*1.11e5

#Load the residual velocities and add them to the dataframe:
u_res =np.load(Path_data+'u_residual_eulerian.npy')
v_res =np.load(Path_data+'v_residual_eulerian.npy')
df['u_res']=u_res
df['v_res']=v_res

#Drop all unnecssary columns:
df.drop(['speed', 'varlat','varlon', 'vart','dangle','dist'],axis=1,inplace=True)

####Split the dataframe on the places where timegap >6 hours (different trajectories)#####
# cnt=0 #a counter for the loop
datetimes = np.zeros(len(df)).astype(datetime) #create array for the datetimes
for cnt,i in enumerate(df['datetime']):
    datetimes[cnt] = datetime.strptime(i,'%Y-%m-%d %H:%M:%S')
    # cnt+=1
#Add a column to the data with the time difference between data points:
for i in range(len(datetimes)-1):
    df['timedeltas'][i] = (datetimes[i+1]-datetimes[i]).total_seconds()/3600

#Split the different trajectories (when dt is not 6 hours)
data_np = np.array(df) #make numpy array
data_split = np.split(data_np,np.where((data_np[:,9])!=6)[0]+1)


##### PLOTTING ######
mean_vel_field = xr.open_dataset(Path_data+'Mean_velocities_eulerian_v2.nc')

datastart=73400
dataend=datastart+60
plt.figure()
xr.plot.pcolormesh(mean_vel_field.isel(time=0).uo)
plt.title('Trajectories')
plt.scatter(meanflowdisp.lon[datastart:dataend,0:15*24+1],meanflowdisp.lat[datastart:dataend,0:15*24+1],s=1,label='Virtual parcels trajectories')
plt.plot(df.lon[datastart:dataend],df.lat[datastart:dataend],lw=3,color='r',label='Buoy trajectory')
plt.scatter(df.lon[datastart],df.lat[datastart],color='y',s=20)
plt.legend()
plt.plot()
#%%


# ############## DAVIS:#############

# Filter out trajectories with less datapoints than 15 days:
cnt=0
timelapse=15*4 #number of datapoints difference (days*4)
for i in range(len(data_split)):
    if timelapse>len(data_split[cnt]):
        del(data_split[cnt])
        continue
    cnt+=1
#Update the dataframe of all data:
df_davis = pd.DataFrame(np.concatenate(data_split),columns=df.columns[:], dtype='object')
df_davis =df_davis.convert_dtypes()

#%%

########### CALCULATE RESIDUAL DISTANCES:
dy_tot= [None]*len(data_split) #Create lists of arrays
dx_tot= [None]*len(data_split)
dy_res= [None]*len(data_split)
dx_res= [None]*len(data_split)
for i in range(len(data_split)):
    dy_tot[i]= np.zeros(len(data_split[i])-timelapse) #Create array for each trajectory
    dx_tot[i]= np.zeros(len(data_split[i])-timelapse)
    dy_res[i]= np.zeros(len(data_split[i])-timelapse)
    dx_res[i]= np.zeros(len(data_split[i])-timelapse)
    for j in range(len(data_split[i])-timelapse):  # Calculate distance for 15 days
        #Total distances:
        dy_tot[i][j] = (data_split[i][j+timelapse,3]-data_split[i][j,3])/360*40008e3
        dx_tot[i][j] = (data_split[i][j+timelapse,4]-data_split[i][j,4])/360*40075e3*np.cos(data_split[i][j,3]/360*2*np.pi)
        # Residual distances (total distance - mean flow distance)
        dy_res[i][j] = dy_tot[i][j] - data_split[i][j,13]
        dx_res[i][j] = dx_tot[i][j] - data_split[i][j,12]        

# Add again to the dataframe and drop nans:
nans = [np.nan]*timelapse
for i in range(len(dx_res)):
    dx_res[i] = np.insert(dx_res[i],0,nans)
    dy_res[i] = np.insert(dy_res[i],0,nans)
df_davis['dy_res']=np.concatenate(dy_res)
df_davis['dx_res']=np.concatenate(dx_res)
df_davis.dropna(inplace=True)
#%% ######## Calculate the diffusivities ############
df_davis.reset_index(drop=True,inplace=True)

D_11 = -df_davis['u_res'] * df_davis['mf_lon_dist']
D_12 = -df_davis['v_res'] * df_davis['mf_lon_dist']
D_21 = -df_davis['u_res'] * df_davis['mf_lat_dist']
D_22 = -df_davis['v_res'] * df_davis['mf_lat_dist']






# D_xr  =xr.Dataset({'k11':(['x','y','time'],D_11),
#                       'k12':(['x','y','time'],D_12),
#                       'k21':(['x','y','time'],D_21),
#                       'k22':(['x','y','time'],D_22)},                     
#                   coords={
#                 "lon": (["x","y"],df_davis.lon),
#                   "lat": (["x","y"],df_davis.lat),
#                   "time": (['time'],df_davis.datetime)},)
 #%%
#Define number of grid cells in x- and y direction
Nbin=20

#Take the binned average of every component of the diffusivity tensor:
D_11_bin,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D_11, statistic='nanmean',bins=Nbin, range=([-65,-45],[55,65]), expand_binnumbers=False)
D_12_bin,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D_12, statistic='nanmean',bins=Nbin, range=([-65,-45],[55,65]), expand_binnumbers=True)
D_21_bin,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D_21, statistic='nanmean',bins=Nbin, range=([-65,-45],[55,65]), expand_binnumbers=True)
D_22_bin,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D_22, statistic='nanmean',bins=Nbin, range=([-65,-45],[55,65]), expand_binnumbers=True)
counts,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D_11, statistic='count',bins=Nbin, range=([-65,-45],[55,65]), expand_binnumbers=True)

D_11_bin=np.swapaxes(D_11_bin, 0, 1)
D_12_bin=np.swapaxes(D_12_bin, 0, 1)
D_21_bin=np.swapaxes(D_21_bin, 0, 1)
D_22_bin=np.swapaxes(D_22_bin, 0, 1)
counts=np.swapaxes(counts, 0, 1)

#Define grid
lon=np.linspace(-64.5, -45.5, Nbin)
lat = np.linspace(55.25, 64.25, Nbin)
lon, lat = np.meshgrid(lon, lat)

x = np.linspace(-64.5, -45.5,Nbin)
y = np.linspace(55.25, 64.75,Nbin)
X, Y = np.meshgrid(x, y)

#%% Add the diffusivity components to 1 xarray dataset
k_matrix =xr.Dataset({'k11':(['x','y'],D_11_bin),
                      'k12':(['x','y'],D_12_bin),
                      'k21':(['x','y'],D_21_bin),
                      'k22':(['x','y'],D_22_bin)},                     
                  coords={
                "lon": (["x","y"],lon),
                  "lat": (["x","y"],lat)},)

#Symmetric part:
k_S =xr.Dataset({'k11':(['x','y'],D_11_bin),
                      'k12':(['x','y'],(D_12_bin+D_21_bin)/2),
                      'k21':(['x','y'],(D_21_bin+D_12_bin)/2),
                      'k22':(['x','y'],D_22_bin)},                     
                  coords={
                "lon": (["x","y"],lon),
                  "lat": (["x","y"],lat)},)

#Same for the number of counts per grid cell
counts_xr =xr.Dataset({'counts':(['x','y'],counts)},                     
                  coords={
                "lon": (["x","y"],lon),
                  "lat": (["x","y"],lat)},)
# Filter out grid cells with less then 50 data points
for i in range(Nbin):
    for j in range(Nbin):
        if counts_xr.counts[i,j]<50:
            k_S.k11[i,j] = np.nan
            k_S.k12[i,j] = np.nan
            k_S.k21[i,j] = np.nan
            k_S.k22[i,j] = np.nan
#%%
#Create arrays for eigen values and eigenvectors:
eig_val = np.zeros((len(lat),len(lon),2))
eig_vec = np.zeros((len(lat),len(lon),2,2))

#Calculate the eigenvalues and eigenvectors for every bin:
for i in range(Nbin):
    for j in range(Nbin):
        try:
            eig_val[i,j,:], eig_vec[i,j,:,:] = linalg.eig(k_S.isel(x=i,y=j).\
                to_array().values.reshape(2,2),check_finite=True)
        except (ValueError): #If there are nan values in the diffusivity: fill eigenvalue and eigenvectors with nans
            eig_val[i,j,:]=[np.nan, np.nan]
            eig_vec[i,j,:,:]=([np.nan, np.nan],[np.nan, np.nan])
            continue


#Make an xarray dataset of the eigenvalues and eigenvectors:
eig_val =xr.Dataset({'labda':(['lon','lat','i'],eig_val)},
                  coords={
                "lon": (["lon"],x),
                  "lat": (["lat"],y)},)
eig_vec =xr.Dataset({'mu':(['lon','lat','i','j'],eig_vec)},
                  coords={
                "lon": (["lon"],x),
                  "lat": (["lat"],y)},
                  attrs={
                      "title": 'Eigen vectors per grid cell'})

#%%
#calculate largest and smalles eigenvalue
index_major= abs(eig_val.labda).argmax(dim='i',skipna=False)
index_minor= abs(eig_val.labda).argmin(dim='i',skipna=False)

plt.figure()
ax=plt.axes(projection=ccrs.PlateCarree())
xr.plot.pcolormesh(np.abs(eig_val.labda.isel(i=index_major))/np.abs(eig_val.labda.isel(i=index_minor)),x="lon",y="lat",\
                  vmin=0000,vmax=100,cmap='rainbow',levels=50,transform=ccrs.PlateCarree())
ax.coastlines(resolution='50m')

#%%
plt.figure()
ax=plt.axes(projection=ccrs.PlateCarree())
xr.plot.pcolormesh(k_S.k11,x="lon",y="lat",\
                  vmin=000,vmax=2000,cmap='rainbow',levels=50)
plt.xlim(-65,-45)
plt.ylim(55,65)
ax.coastlines(resolution='50m')
#%%
plt.figure()
ax=plt.axes(projection=ccrs.PlateCarree())

eig_val.labda.attrs['units']='$m^2/s$'
xr.plot.contourf(eig_val.labda.isel(i=index_major),x="lon",y="lat",\
                 vmin=-10000,corner_mask=False,vmax=10000,cmap='coolwarm',levels=50)
plt.title('Major principle component')
plt.xlim(-65,-45)

ax.coastlines(resolution='50m')


plt.figure()
ax=plt.axes(projection=ccrs.PlateCarree())

xr.plot.contourf(eig_val.labda.isel(i=index_minor),x="lon",y="lat",\
                 vmin=-2000,corner_mask=False,vmax=2000,cmap='coolwarm',levels=51)
ax.coastlines(resolution='50m')
plt.title('Minor principle component')
#%%
plt.figure()
ax=plt.axes(projection=ccrs.PlateCarree())

xr.plot.contourf(np.abs(eig_val.labda.isel(i=index_major))/np.abs(eig_val.labda.isel(i=index_minor)),x="lon",y="lat",\
                 vmin=0,vmax=40,cmap='viridis',levels=51)
ax.coastlines(resolution='50m')
plt.title('Anisotropy (major/minor component')





#%% Plotting of the ellipses

plot_ellipse(eig_val, eig_vec)


#%%

# ds 
# ds.sel(lat=60)
# bath = xr.open_dataset(Path_data+'gebco_2020_n65.0_s55.0_w-65.0_e-45.0.nc')

# # bath = bath.interp(lon=x, lat=y)
# bath_df = bath.sel(lon=df_davis.lon.to_xarray(),lat=df_davis.lat.to_xarray(), method='nearest')

# bath_binned,bin_edges,_ = stats.binned_statistic(bath_df.elevation.values, np.abs(D_11), bins=200)

# plt.scatter(bin_edges[:-1],bath_binned)

