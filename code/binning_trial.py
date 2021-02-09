
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

ds = df.to_xarray()
# ds =ds.expand_dims(dim='binnum')

Nbin=20

# bins = xr.Dataset({'lat':(['binnumb', 'index']),
#                    'lon':(['binnumb', 'index'])}
#  

#%%

latbins = np.linspace(55,65,Nbin+1)
lonbins = np.linspace(-65,-45,Nbin+1)
latnumbers = np.digitize(df.lat, latbins, right=True)
lonnumbers = np.digitize(df.lon, lonbins, right=True)
df['lonbin']=lonnumbers
df['latbin']=latnumbers


df_groups = df.groupby([df.lonbin,df.latbin])

ar = np.dtype({'names':['een', 'twee'], 'formats':[np.float, np.float]})    
#%%
# times = np.arange(6)
# ds = xr.Dataset({
#     'bin%i' %i: xr.DataArray(
#                 data   = df.loc[((df.lon>bin_xedges[i]) & (df.lon<bin_xedges[i+1]) & (df.lat>bin_yedges[i]) & (df.lat<bin_yedges[i+1])),:],   # enter data here
#                 dims   = ['index'],
#                 coords = {'index': 'index'},
#                 attrs  = {
#                     '_FillValue': -999.9,
#                     'units'     : 'W/m2'})})


# #%%
# df1 = pd.DataFrame(np.random.randn(6, 4),
#                     index=list('abcdef'),
#                     columns=list('ABCD'))

# df1.loc[df1['A'] > 0, :]
# ds = xr.Dataset(df)
# coordx = 



# ds = ds.expand_dims(dim=({'lat_bin':Nbin, 'lon_bin':Nbin}))

# ds = ds.expand_dims(dim={'lat_bin':ds.latbin, 'lon_bin':ds.lonbin})

#%%
bin_xedges = np.linspace(-65,-45,Nbin+1)
bin_yedges = np.linspace(55,65,Nbin+1)


df.loc[(df.lon>bin_xedges[i]) & (df.lon<bin_xedges[i+1]) & (df.lat>bin_yedges[i]) & (df.lat<bin_yedges[i+1]),:]      
# df.set_index('latbin', inplace=True)
# multi = pd.MultiIndex.from_frame(df)
        
# bins = xr.Dataset(
#     data_vars=dict(
#         lat=(["binnumb"], df.lat),
#         precipitation=(["x", "y", "time"], precipitation),
    
#     'binnumb':(['lat', 'lon'])
#                   )


# # bins = np.zeros((Nbin,Nbin))

# for i in range(Nbin):
#     for j in range(Nbin):   
#         bins[i,j] = np.where(((df.lon>bin_xedges[i]) & (df.lon<bin_xedges[i+1]) & (df.lat>bin_yedges[i]) & (df.lat<bin_yedges[i+1])))         




