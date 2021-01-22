#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:47:59 2021

@author: tychobovenschen
"""

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
import pandas as pd
from binned_statistic import binned_statistic_2d_new
from datetime import datetime
from scipy import linalg


"""A script for calculating the diffusivity tensor according to the method of Davis(1991)
From the diffusivity the symmetric part is taken and from this the eigenvalues and vectors
are calculated"""

#Data paths:
Path_data = '/Users/tychobovenschen/Documents/MasterJaar2/Thesis/data/'
File_data = 'interpolated_gld.20201120_024210.txt'

######## DATA PROCESSING ##########:
"""This part divides the dataset into trajectories, according to the time difference
    between different data points (when it is not exactly 6 hours, we have a new trajectory
    The data is then put in a list of numpy arrays"""
    
#Open the file with the buoy data: 
df = pd.read_pickle(Path_data+'df.pkl')
df.reset_index(drop=True,inplace=True)

#Open file with mean flow displacements:
meanflowdisp=xr.open_dataset(Path_data+'meanflowdisp.nc') #Hourly trajectory data of particles release

#Add the mean flow distance to the dataframe (and convert to meters):
df['mf_lon_dist']=(meanflowdisp.lon[:,15*24]-meanflowdisp.lon[:,0])/360*40075e3 *np.cos(meanflowdisp.lat[:,0]*np.pi/180)
df['mf_lat_dist']=(meanflowdisp.lat[:,15*24]-meanflowdisp.lat[:,0])/360*40008e3

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

# Filter out trajectories with less datapoints than 15 days:
cnt=0
timelapse=15*4 #number of datapoints difference (days*4)
for i in range(len(data_split)):
    if timelapse>len(data_split[cnt]):
        del(data_split[cnt])
        continue
    cnt+=1
    
#Add all the filtered trajectories together again in a dataframe:
df_davis = pd.DataFrame(np.concatenate(data_split),columns=df.columns[:], dtype='object')
df_davis =df_davis.convert_dtypes()

########### CALCULATE RESIDUAL DISTANCES: ##########
"""This part subtracts the distance traveled caused by the mean flow from the total observed distance
    travelled in 15 days, for every data point"""
    
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
        #Total distances (converted to meters):
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
"""Here, for every datapoint, a diffusivity tensor is calculated, after this
    the components of the tensor are averaged over the grid cells and the symmetric
    part of the tensor is taken"""
df_davis.reset_index(drop=True,inplace=True) #reset the index of the dataframe
D = np.zeros((len(df_davis),2,2)) #create diffusivity tensor (2 dimensional for each datapoint)

#The indices of the 2 dimensional tensor of the diffusivity:
indicesj=['u_res', 'v_res'] 
indicesk=['mf_lon_dist', 'mf_lat_dist']

# Calculate the 2 dimensional tensor for the diffusivity for every datapoint:
for i in range(len(df_davis)):
    for j in range(2):
        for k in range(2) :
            D[i, j, k] = -df_davis[indicesj[j]][i]*df_davis[indicesk[k]][i]


#Define number of grid cells in x- and y direction
Nbin=20 

#Take the binned average of every component of the diffusivity tensor:
D_11,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,0,0],statistic='nanmean',bins=Nbin, expand_binnumbers=True)
D_12,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,0,1],statistic='nanmean',bins=Nbin, expand_binnumbers=True)
D_21,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,1,0],statistic='nanmean',bins=Nbin, expand_binnumbers=True)
D_22,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(df_davis['lon'],df_davis['lat'], D[:,1,1],statistic='nanmean',bins=Nbin, expand_binnumbers=True)

#Define grid
lon=np.linspace(-65, -45, Nbin)
lat = np.linspace(55, 65, Nbin)
lon, lat = np.meshgrid(lon, lat)

#Add the diffusivity components into one dataset, taking only the symmetric part:
k_S =xr.Dataset({'k11':(['x','y'],D_11),
                      'k12':(['x','y'],(D_12+D_21)/2),
                      'k21':(['x','y'],(D_21+D_12)/2),
                      'k22':(['x','y'],D_22)},                     
                  coords={
                "lon": (["x","y"],lon),
                  "lat": (["x","y"],lat)},)



#%%##### EIGENVALUES AND EIGENVECTORS: #######
"""For every grid cell the eigen values and eigenvectors are calculated"""
#Create arrays for eigen values and eigenvectors:
eig_val = np.zeros((len(lat), len(lon), 2))
eig_vec = np.zeros((len(lat), len(lon), 2, 2))

#Calculate the eigenvalues and eigenvectors for every bin:
for i in range(Nbin):
    for j in range(Nbin):
        try:
            eig_val[i,j,:], eig_vec[i,j,:,:] = linalg.eig(k_S.isel(x=j, y=i).\
                to_array().values.reshape(2,2), check_finite=True)
        except (ValueError): 
        #If there are nan values in the diffusivity a ValueError is raised: in this case:
            #fill eigenvalue and eigenvectors with nans
            eig_val[i,j,:] = [np.nan, np.nan]
            eig_vec[i,j,:,:] = ([np.nan, np.nan], [np.nan, np.nan])
            continue


#Make an xarray dataset of the eigenvalues and eigenvectors:
eig_val =xr.Dataset({'labda':(['x','y','i'],eig_val)},
                  coords={
                "lon": (["x","y"],lon),
                  "lat": (["x","y"],lat)},
                  attrs={
                      "title":"Eigen values per grid cell"})

eig_vec =xr.Dataset({'mu':(['x','y','i','j'],eig_vec)},
                  coords={
                "lon": (["x","y"],lon),
                  "lat": (["x","y"],lat)},
                  attrs={
                      "title": 'Eigen vectors per grid cell'})

