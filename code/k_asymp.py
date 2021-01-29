
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
meanflowdisp=xr.open_dataset(Path_data+'meanflowdisp2.nc') #Hourly trajectory data of particles release


mfd_lon_time=np.zeros((len(df),60*24))
mfd_lat_time=np.zeros((len(df),60*24))

for i in tqdm.tqdm(range(60*24),position=0,leave=True):
    mfd_lon_time[:,i]=(meanflowdisp.lon[:,i]-meanflowdisp.lon[:,0])*1.11e5 *np.cos(meanflowdisp.lat[:,0]*np.pi/180)
    mfd_lat_time[:,i]=(meanflowdisp.lat[:,i]-meanflowdisp.lat[:,0])*1.11e5
    
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
#%%
#Split the different trajectories (when dt is not 6 hours)
data_np = np.array(df) #make numpy array
data_split = np.split(data_np,np.where((data_np[:,9])!=6)[0]+1)


mfd_lat_time_split = np.split(mfd_lat_time,np.where((data_np[:,9])!=6)[0]+1)
mfd_lon_time_split = np.split(mfd_lat_time,np.where((data_np[:,9])!=6)[0]+1)





#%%


# ############## DAVIS:#############

# Filter out trajectories with less datapoints than 15 days:
cnt=0
timelapse=60*4 #number of datapoints difference (days*4)
for i in range(len(data_split)):
    if timelapse>len(data_split[cnt]):
        del(data_split[cnt],mfd_lat_time_split[cnt],mfd_lon_time_split[cnt])
        continue
    cnt+=1
#Update the dataframe of all data:
df_davis = pd.DataFrame(np.concatenate(data_split),columns=df.columns[:], dtype='object')
df_davis =df_davis.convert_dtypes()
mfd_lat_time=np.concatenate(mfd_lat_time_split)
mfd_lon_time=np.concatenate(mfd_lon_time_split)

#%%

########### CALCULATE RESIDUAL DISTANCES:
dy_tot= [None]*len(data_split) #Create lists of arrays
dx_tot= [None]*len(data_split)
dy_res= [None]*len(data_split)
dx_res= [None]*len(data_split)
timelapse=60*4
for i in tqdm.tqdm(range(len(data_split)),position=0,leave=True):
    dy_tot[i]= np.zeros((len(data_split[i])-timelapse,timelapse)) #Create array for each trajectory
    dx_tot[i]= np.zeros((len(data_split[i])-timelapse,timelapse)) 
    dy_res[i]= np.zeros((len(data_split[i])-timelapse,timelapse)) 
    dx_res[i]= np.zeros((len(data_split[i])-timelapse,timelapse)) 
    for j in range(len(data_split[i])-timelapse):  # Calculate distance for 15 days
        for t in range(timelapse):
            #Total distances:
            dy_tot[i][j,t] = (data_split[i][j+t,3]-data_split[i][j,3])/360*40008e3
            dx_tot[i][j,t] = (data_split[i][j+t,4]-data_split[i][j,4])/360*40075e3*np.cos(data_split[i][j,3]/360*2*np.pi)
            # Residual distances (total-mean flow distance)
            dy_res[i][j,t] = dy_tot[i][j,t] - mfd_lat_time_split[i][j,t*6]
            dx_res[i][j,t] = dx_tot[i][j,t] - mfd_lon_time_split[i][j,t*6]      
#%%
# Add again to the dataframe and drop nans:
nans2d = np.zeros((timelapse,timelapse))*np.nan
nans = np.zeros((timelapse,12))*np.nan
for i in range(len(dx_res)):
    dx_res[i] = np.insert(dx_res[i],0,nans2d, axis=0)
    dy_res[i] = np.insert(dy_res[i],0,nans2d, axis=0)
mfd_lat_time=pd.DataFrame(np.concatenate(dy_res))
mfd_lon_time=pd.DataFrame(np.concatenate(dx_res))

mfd_lat_time.dropna(inplace=True)
mfd_lon_time.dropna(inplace=True)

df_davis=df_davis.reindex(mfd_lat_time.index)
df_davis.dropna(inplace=True)
#%% ######## Calculate the diffusivities ############
df_davis.reset_index(drop=True,inplace=True)
mfd_lat_time.reset_index(drop=True,inplace=True)
mfd_lon_time.reset_index(drop=True,inplace=True)
D = np.zeros((len(df_davis),timelapse,2,2)) #create diffusivity tensor (2 dimensional for each datapoint)
indicesj=['u_res', 'v_res']
indicesk=[np.array(mfd_lon_time), np.array(mfd_lat_time)]
#%%
for i in tqdm.tqdm(range(len(df_davis)),position=0,leave=True):
    for t in range(timelapse):
        for j in range(2):
            for k in range(2) :
                D[i,t,j,k] = -df_davis[indicesj[j]][i]*indicesk[k][i,t]


 #%%
#Define number of grid cells in x- and y direction
Nbin=20 
D_11 = np.zeros((Nbin,Nbin,timelapse))
D_12 = np.zeros((Nbin,Nbin,timelapse))
D_21 = np.zeros((Nbin,Nbin,timelapse))
D_22 = np.zeros((Nbin,Nbin,timelapse))
#Take the binned average of every component of the diffusivity tensor:
for t in tqdm.tqdm(range(timelapse),position=0):
    D_11[:,:,t],  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,t,0,0],statistic='nanmean',bins=Nbin, expand_binnumbers=True)
    D_12[:,:,t],  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,t,0,1],statistic='nanmean',bins=Nbin, expand_binnumbers=True)
    D_21[:,:,t],  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,t,1,0],statistic='nanmean',bins=Nbin, expand_binnumbers=True)
    D_22[:,:,t],  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,t,1,1],statistic='nanmean',bins=Nbin, expand_binnumbers=True)
#%%
    
D_11=np.swapaxes(D_11,0,1)
D_12=np.swapaxes(D_12,0,1)
D_21=np.swapaxes(D_21,0,1)
D_22=np.swapaxes(D_22,0,1)
    


#%%
#Define grid
lon=np.linspace(-65,-45,Nbin)
lat = np.linspace(55,65,Nbin)
lon, lat = np.meshgrid(lon,lat)


#%%
k_matrix =xr.Dataset({'k11':(['x','y','time'],D_11),
                      'k12':(['x','y','time'],D_12),
                      'k21':(['x','y','time'],D_21),
                      'k22':(['x','y','time'],D_22)},                     
                  coords={
                "lon": (["x","y"],lon),
                  "lat": (["x","y"],lat),
                  "time": (['time'],np.arange(timelapse))},)

#Symmetric part:
k_S =xr.Dataset({'k11':(['x','y','time'],D_11),
                      'k12':(['x','y','time'],(D_12+D_21)/2),
                      'k21':(['x','y','time'],(D_21+D_12)/2),
                      'k22':(['x','y','time'],D_22)},                     
                  coords={
                "lon": (["x","y"],lon),
                  "lat": (["x","y"],lat),
                  "time": (['time'],np.arange(timelapse))},)



#%%
#Create arrays for eigen values and eigenvectors:
eig_val = np.zeros((len(lat),len(lon),timelapse,2))
eig_vec = np.zeros((len(lat),len(lon),timelapse,2,2))

for i in range(Nbin):
    for j in range(Nbin):
        for t in range(timelapse):
            try:
                eig_val[i,j,t,:], eig_vec[i,j,:,:] = scipy.linalg.eig(k_S.isel(x=j,y=i,time=t).\
                    to_array().values.reshape(2,2),check_finite=True)
            except (ValueError):
                eig_val[i,j,t,:]=[np.nan, np.nan]
                eig_vec[i,j,t,:,:]=([np.nan, np.nan],[np.nan, np.nan])
                continue



eig_val =xr.Dataset({'labda':(['x','y','t','i'],eig_val)},
                  coords={
                "lon": (["x","y"],lon),
                  "lat": (["x","y"],lat),
                  "t": (['t'],np.arange(timelapse))},)
eig_vec =xr.Dataset({'mu':(['x','y','t','i','j'],eig_vec)},
                  coords={
                "lon": (["x","y"],lon),
                  "lat": (["x","y"],lat),
                  "t": (['t'],np.arange(timelapse))},
                  attrs={
                      "title": 'Eigen vectors per grid cell'})

#%%

eig_val = xr.open_dataset(Path_data+'eig_val_60.nc')
k_S = xr.open_dataset(Path_data+'k_S.nc')

#calculate largest and smalles eigenvalue
index_major= abs(eig_val.labda).argmax(dim='i',skipna=False)
index_minor= abs(eig_val.labda).argmin(dim='i',skipna=False)

# plt.figure()
# ax=plt.axes(projection=ccrs.PlateCarree())
# xr.plot.contourf(np.abs(eig_val.labda.isel(i=index_major)),x="lon",y="lat",\
#                  vmin=0000,corner_mask=False,vmax=12000,cmap='rainbow',levels=50)
# ax.coastlines(resolution='50m')
def norm(data):
    return (data)/(np.max(data)-np.min(data))
plt.figure()
xr.plot.line(np.abs(eig_val.labda.isel(i=index_major).mean(dim=('x','y'))),x='t')
plt.xlabel('time (days)')
plt.xticks(np.linspace(0,241,11),(np.linspace(0,60,11,dtype=int)))

# normalized = np.zeros((240,20,20))
# for k in tqdm.tqdm(range(20),position=0):
#     for j in range(20):
#         normalized[:,k,j] = norm(np.abs(eig_val.labda.isel(i=index_major).isel(x=j,y=k)))
# plt.figure()
# plt.plot(np.linspace(0,240,240),np.nanmean(normalized,axis=(1,2)))

#%%

plt.figure()
# for i in range(5):
xr.plot.line(np.mean(norm(np.abs(eig_val.labda.isel(i=index_minor).isel(x=slice(0,20),y=9))),axis=(0)),x='t')
plt.xlabel('time (days)')
plt.xticks(np.linspace(0,241,11),(np.linspace(0,60,11,dtype=int)))
plt.ylabel('k')
plt.title('Major principle component of k')
# dy_tot[i][j] = (data_split[i][j+timelapse,3]-data_split[i][j,3])/360*40008e3
# dx_tot[i][j] = (data_split[i][j+timelapse,4]-data_split[i][j,4])/360*40075e3*np.cos(data_split[i][j,3]/360*2*np.pi)
# # Residual distances (total-mean flow distance)
# dy_res[i][j] = dy_tot[i][j] - data_split[i][j,13]
# dx_res[i][j] = dx_tot[i][j] - data_split[i][j,12]     

########### INFLUENCE OF INTEGRAL TIME SCALE:

df['mf_lon_dist']=(meanflowdisp.lon[:,15*24]-meanflowdisp.lon[:,0])*1.11e5 *np.cos(meanflowdisp.lat[:,0]*np.pi/180)
df['mf_lat_dist']=(meanflowdisp.lat[:,15*24]-meanflowdisp.lat[:,0])*1.11e5

#%%
buoynr=1
ITS=30
dx_tot_time = np.zeros(ITS)
dy_tot_time = np.zeros(ITS)
dx_res_time = np.zeros(ITS)
dy_res_time = np.zeros(ITS)
dx_mean_time = np.zeros(ITS)
dy_mean_time = np.zeros(ITS)



for i in range(ITS):
    dx_tot_time[i] = (data_split[buoynr][i*4,4]-data_split[buoynr][0,4])/\
        360*40075e3*np.cos(data_split[buoynr][0,3]/360*2*np.pi)
    dy_tot_time[i] = (data_split[buoynr][i*4,3]-data_split[buoynr][0,3])/360*40008e3
    dx_mean_time[i] = meanflowdisp.lon[buoynr,i*24]-meanflowdisp.lon[buoynr,0]*1.11e5 *np.cos(meanflowdisp.lat[buoynr,0]*np.pi/180)
    dy_mean_time[i] = meanflowdisp.lat[buoynr,i*24]-meanflowdisp.lat[buoynr,0]*1.11e5 
    dy_res_time[i]= dy_tot_time[i] - dy_mean_time[i]
    dx_res_time[i]= dx_tot_time[i] - dx_mean_time[i]

k_time_x =-dx_res_time*u_res[0]
k_time_y =-dy_res_time*v_res[0]


plt.figure()
plt.plot(np.arange(ITS),k_time_y)


#%%
