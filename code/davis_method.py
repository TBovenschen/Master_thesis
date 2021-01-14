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
import cartopy.crs as ccrs
from scipy import stats
from binned_statistic import binned_statistic_2d_new
from datetime import datetime
from datetime import timedelta
from reanalysisdata import reanalysis_meanvel
from plotonmap import plotonmap
from plot_angles import plot_angles
import xarray as xr
import scipy
import tqdm
from residual_vel_eul import calc_residual_vel_eul


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
#Add the mean flow distance to the dataframe:
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
cnt=0 #a counter for the loop
datetimes = np.zeros(len(df)).astype(datetime) #create array for the datetimes
for i in df['datetime']:
    datetimes[cnt] = datetime.strptime(i,'%Y-%m-%d %H:%M:%S')
    cnt+=1
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
        # Residual distances (total-mean flow distance)
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
D = np.zeros((len(df_davis),2,2)) #create diffusivity tensor (2 dimensional for each datapoint)
indicesj=['u_res', 'v_res']
indicesk=['mf_lon_dist', 'mf_lat_dist']
for i in range(len(df_davis)):
    for j in range(2):
        for k in range(2) :
            D[i,j,k] = -df_davis[indicesj[j]][i]*df_davis[indicesk[k]][i]


 #%%
#Define number of grid cells in x- and y direction
Nbin=40 

#Take the binned average of every component of the diffusivity tensor:
D_11,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,0,0],statistic='nanmean',bins=Nbin, expand_binnumbers=True)
D_12,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,0,1],statistic='nanmean',bins=Nbin, expand_binnumbers=True)
D_21,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,1,0],statistic='nanmean',bins=Nbin, expand_binnumbers=True)
D_22,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,1,1],statistic='nanmean',bins=Nbin, expand_binnumbers=True)
#%%

#Stack the different components into 1 4 dimensional array.

data_xr = np.dstack((D_11,D_12))
data_xr2 = np.dstack((D_21,D_22))
data_xr3 = np.swapaxes(np.stack((data_xr,data_xr2),axis=3),0,1)

#Define grid
lon=np.linspace(-65,-45,Nbin)
lat = np.linspace(55,65,Nbin)
lon, lat = np.meshgrid(lon,lat)


#%%
#Create xarray dataset:
k_matrix =xr.Dataset({'k':(['x','y','i','j'],data_xr3)},
                 coords={
                "lon": (["x","y"],lon),
                 "lat": (["x","y"],lat)},)
#Transposed dataset
k_matrix_T=k_matrix.swap_dims({'i':'j','j':'i'}).transpose()

#The symmetric part:
k_S= (k_matrix+k_matrix_T)/2

#%%
#Create arrays for eigen values and eigenvectors:
eig_val = np.zeros((len(lat),len(lon),2))
eig_vec = np.zeros((len(lat),len(lon),2,2))

for i in range(40):
    for j in range(40):
        try:
            eig_val[i,j,:], eig_vec[i,j,:,:] = scipy.linalg.eig(k_S.k.isel(x=i,y=j),check_finite=True)
        except (ValueError):
            eig_val[i,j,:]=[np.nan, np.nan]
            continue
eig_val =xr.Dataset({'labda':(['x','y','i'],eig_val)},
                 coords={
                "lon": (["x","y"],lon),
                 "lat": (["x","y"],lat)},)
eig_vec =xr.Dataset({'mu':(['x','y','i','j'],eig_vec)},
                 coords={
                "lon": (["x","y"],lon),
                 "lat": (["x","y"],lat)},
                 attrs={
                     "title": 'Eigen vectors per grid cell'})

#%%
#calculate largest and smalles eigenvalue
index_major= abs(eig_val.labda).argmax(dim='i',skipna=False)
index_minor= abs(eig_val.labda).argmin(dim='i',skipna=False)

plt.figure()
ax=plt.axes(projection=ccrs.PlateCarree())
xr.plot.contourf(np.abs(eig_val.labda.isel(i=index_major)),x="lon",y="lat",vmin=0000,corner_mask=False,vmax=12000,cmap='rainbow',levels=50)
ax.coastlines(resolution='50m')

