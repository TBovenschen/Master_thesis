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
from datetime import datetime
from datetime import timedelta

path_data = '/Users/tychobovenschen/Documents/MasterJaar2/Thesis/data/'
file_data = 'interpolated_gld.20201120_024210.txt'
#read data
df  = pd.read_csv(path_data+file_data,sep='\s+')
#Read all gps IDs with GPS:
gps_ids = pd.read_csv(path_data+'gps_ids.dat')

#Filter out buoys without GPS:
df = df[df['id'].isin(gps_ids['ID'])]
land = globe.is_land(df['lat'],df['lon'])
df = df[~land]
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

#split data for different buoys
# data_split = np.split(data,np.where((np.diff(data[:,0])!=0))[0]+1)
data_split = np.split(data,np.where((data[:,13]!=6))[0]+1)

#Filter out buoys with only 1 data point in the region
cnt = 0
for i in range(len(data_split)):
    if len(data_split[cnt])<5:
        del data_split[cnt]
        cnt-=1
    cnt+=1


#%% Plot the buoy trajectories


colorss = ['r', 'b', 'y','g','k'] #Colors used for the trajectories
        
#Create plot for the trajectories
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,12), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([295, 315, 55, 65])
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
i=77
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,12), subplot_kw={'projection': ccrs.PlateCarree()})
ax.scatter(data_split[i][:,4],data_split[i][:,3],s=5,color=colorss[np.mod(i,len(colorss))])
ax.coastlines() 
plt.xlim([-65,-45])
plt.ylim([55,65])
ax.set_xticks([-65, -60, -55, -50, -45])
plt.title('Particle trajectories in Labrador Sea',fontsize=16)
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees')
ax.set_yticks([55, 60, 65])
plt.show()
# groups=df.groupby('id')
#%% Calculate angles:
#Create empty lists (for the different buoys)
dy= [None]*len(data_split)
dx= [None]*len(data_split)
angle= [None]*len(data_split)
dangle= [None]*len(data_split)
dist= [None]*len(data_split)

#For every buoy, calculate the dy and x and with that the angles at every timestep
for i in range(len(data_split)):
    dy[i]= np.zeros(len(data_split[i])-1)
    dx[i]= np.zeros(len(data_split[i])-1)
    angle[i]= np.zeros(len(data_split[i])-1)
    dist[i]= np.zeros(len(data_split[i])-1)
    for j in range(len(data_split[i])-1):
        dy[i][j] = (data_split[i][j+1,3]-data_split[i][j,3])/360*40008e3
        dx[i][j] = (data_split[i][j+1,4]-data_split[i][j,4])/360*40075e3*np.cos(data_split[i][j,3]/360*2*np.pi)
        angle[i][j] = np.arctan(dy[i][j]/dx[i][j])/(2*np.pi)*360
        dist[i][j]= np.sqrt(dy[i][j]**2+dx[i][j]**2)
#Calculate the difference of the angles between consecutive timestteps
for i in range(len(data_split)):
    dangle[i]= np.zeros(len(data_split[i])-2)
    for j in range(len(data_split[i])-2):        
        dangle[i][j]=angle[i][j+1]-angle[i][j]
        # dx[i][j] = distance.distance(((data_split[i][j+1,2],data_split[i][j,3]), (data_split[i][j,2],data_split[i][j,3]))).m
# np.save('Data/angles.npy',dangle)        
#%% Plot the angles  

plt.figure()
plt.hist(np.concatenate(dangle),bins=300, stacked=True)
plt.title('Difference in angles between consecutive data points',fontsize=16)
plt.ylabel('Number of datapoints')
plt.xlabel('Angles (degrees)')
plt.grid()
plt.show()
#%%


countzero=np.zeros(len(angle))
for i in range(len(angle)):
    countzero[i] = np.count_nonzero(dist[i]>20000)
#%% Calculate diffusivities
u= [None]*len(dangle)
v= [None]*len(dangle)
vel= [None]*len(dangle)
phi = [None]*len(dangle)
tau = [None]*len(dangle)
D = [None]*len(dangle)
#Fill arrays for velocities
for i in range(len(data_split)):
    u[i] = data_split[i][1:-1,8]/100
    v[i] = data_split[i][1:-1,9]/100
n=2 #number of dimensions
dt = 3600*6  #timestep in seconds

for i in range(len(dangle)):
    vel[i] = np.zeros(len(u[i]))
    for j in range(len(u[i])):
        vel[i][j] = np.sqrt(u[i][j]**2+v[i][j]**2) #The total velocity


for i in range(len(dangle)):
    phi = np.cos(dangle[i]/360*2*np.pi) # The cosine of the angle
    tau = dt/(1-phi)    #correlation time scale
    D[i] = 1/n * vel[i]**2 *dt/(1-phi) #The diffusivity

D_resh=[None]*len(dangle)
for i in range(len(dangle)):
    D_resh[i] = np.insert(D[i],0,np.nan)
    D[i] = np.append(D[i],np.nan)
D_resh = pd.DataFrame(np.concatenate(D_resh))

df['Diff']= D_resh
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

 #%%   
Nbin=20
Mean_diff, xedges, yedges, binnumber = stats.binned_statistic_2d(df['lon'],df['lat'], df['Diff'],statistic='mean',bins=Nbin)

x = np.linspace(295,315,Nbin)
y = np.linspace(55,65,Nbin)
X,Y = np.meshgrid(x,y)

plt.figure()
plt.contourf(X,Y,Mean_diff,locator=ticker.LogLocator())
plt.colorbar()
plt.title('Diffusivities')
plt.xlabel('Longitude (degrees')
plt.ylabel('Latitude (degrees')
plt.show()

#%%    
# fig=plt.figure(1)
# ax = plt.axes(xlim=(-65,-45), ylim=(55, 65))
# line, = ax.plot([], [], lw=2)
# time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
# cnt = 0
# line_lw, = ax.plot([], [], lw=2) #create object to store the data
# i = 16

# def init(): #function for initial frame (empty)
#     # line.set_data([], [])
#     line_lw.set_data([], [])
#     return line_lw,

# def anim(i): #function for the ,:])
#     global data_split
#     line_lw.set_data(data_split[16][i,4],data_split[16][i,3])
#     return line_lw,
#     # line.set_data(x, C_e[i

# #ax.set_title('$\Phi$ for timestep {:.2f} '.format(t)+'$\Lambda$={:.1f} and '.format(Lambda)+'$r$={:.4f}'.format(r))
# animat = animation.FuncAnimation(fig,anim, init_func=init, interval=100, blit=True)