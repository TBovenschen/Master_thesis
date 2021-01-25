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
from binned_statistic import binned_statistic_2d_new
from datetime import datetime
import scipy
import tqdm


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
D = np.zeros((len(df_davis),2,2)) #create diffusivity tensor (2 dimensional for each datapoint)
indicesj=['u_res', 'v_res']
indicesk=['mf_lon_dist', 'mf_lat_dist']
for i in range(len(df_davis)):
    for j in range(2):
        for k in range(2) :
            D[i,j,k] = -df_davis[indicesj[j]][i]*df_davis[indicesk[k]][i]


 #%%
#Define number of grid cells in x- and y direction
Nbin=20 

#Take the binned average of every component of the diffusivity tensor:
D_11,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,0,0],statistic='nanmean',bins=Nbin, expand_binnumbers=True)
D_12,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,0,1],statistic='nanmean',bins=Nbin, expand_binnumbers=True)
D_21,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,1,0],statistic='nanmean',bins=Nbin, expand_binnumbers=True)
D_22,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,1,1],statistic='nanmean',bins=Nbin, expand_binnumbers=True)
counts,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,0,0],statistic='count',bins=Nbin, expand_binnumbers=True)

D_11=np.swapaxes(D_11,0,1)
D_12=np.swapaxes(D_12,0,1)
D_21=np.swapaxes(D_21,0,1)
D_22=np.swapaxes(D_22,0,1)
counts=np.swapaxes(counts,0,1)
#%%




#Define grid
lon=np.linspace(-65,-45,Nbin)
lat = np.linspace(55,65,Nbin)
lon, lat = np.meshgrid(lon,lat)


#%%
k_matrix =xr.Dataset({'k11':(['x','y'],D_11),
                      'k12':(['x','y'],D_12),
                      'k21':(['x','y'],D_21),
                      'k22':(['x','y'],D_22)},                     
                  coords={
                "lon": (["x","y"],lon),
                  "lat": (["x","y"],lat)},)

#Symmetric part:
k_S =xr.Dataset({'k11':(['x','y'],D_11),
                      'k12':(['x','y'],(D_12+D_21)/2),
                      'k21':(['x','y'],(D_21+D_12)/2),
                      'k22':(['x','y'],D_22)},                     
                  coords={
                "lon": (["x","y"],lon),
                  "lat": (["x","y"],lat)},)
counts_xr =xr.Dataset({'counts':(['x','y'],counts)},                     
                  coords={
                "lon": (["x","y"],lon),
                  "lat": (["x","y"],lat)},)

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
            eig_val[i,j,:], eig_vec[i,j,:,:] = scipy.linalg.eig(k_S.isel(x=i,y=j).\
                to_array().values.reshape(2,2),check_finite=True)
        except (ValueError): #If there are nan values in the diffusivity: fill eigenvalue and eigenvectors with nans
            eig_val[i,j,:]=[np.nan, np.nan]
            eig_vec[i,j,:,:]=([np.nan, np.nan],[np.nan, np.nan])
            continue


#Make an xarray dataset of the eigenvalues and eigenvectors:
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
xr.plot.contourf(np.abs(eig_val.labda.isel(i=index_minor)),x="lon",y="lat",\
                  vmin=0000,corner_mask=False,vmax=2000,cmap='rainbow',levels=50)
ax.coastlines(resolution='50m')

#%%
plt.figure()
ax=plt.axes(projection=ccrs.PlateCarree())
xr.plot.pcolormesh(k_S.k11,x="lon",y="lat",\
                  vmin=000,vmax=2000,cmap='rainbow',levels=50)
ax.coastlines(resolution='50m')
#%%
plt.figure()
ax=plt.axes(projection=ccrs.PlateCarree())

eig_val.labda.attrs['units']='$m^2/s$'
xr.plot.contourf(eig_val.labda.isel(i=index_major),x="lon",y="lat",\
                 vmin=-10000,corner_mask=False,vmax=10000,cmap='coolwarm',levels=50)
plt.title('Major principle component')
ax.coastlines(resolution='50m')


plt.figure()
ax=plt.axes(projection=ccrs.PlateCarree())

xr.plot.contourf(eig_val.labda.isel(i=index_minor),x="lon",y="lat",\
                 vmin=-2000,corner_mask=False,vmax=2000,cmap='coolwarm',levels=51)
ax.coastlines(resolution='50m')
plt.title('Minor principle component')


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

#%%

plt.figure()
plt.plot(np.arange(ITS),k_time_y)
from matplotlib.collections import EllipseCollection
x = np.linspace(-65,-45,Nbin)
y = np.linspace(55,65,Nbin)
X, Y = np.meshgrid(x, y)

XY = np.column_stack((X.ravel(), Y.ravel()))


fig, ax = plt.subplots()
ax = plt.axes(projection=ccrs.PlateCarree()) 

ells = EllipseCollection(eig_val.labda.isel(i=index_major)/6000,eig_val.labda.isel(i=index_minor)/6000,\
                         np.arctan2(eig_vec.mu.isel(i=index_minor,j=0),eig_vec.mu.isel(i=index_minor,j=1)).values/pi*180,units='x', offsets=XY,
                       transOffset=ax.transData, facecolors='None',edgecolors='tab:blue')
# ax.autoscale_view()
# ells.set_array((counts).ravel())

ax.coastlines(resolution='50m')
ax.add_collection(ells)
ax.autoscale()