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

df = pd.DataFrame(np.concatenate(data_split),columns=df.columns, dtype='object')
df=df.convert_dtypes()
#%% Plot the buoy trajectories


colorss = ['r', 'b', 'y','g','orange'] #Colors used for the trajectories
        
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
i=81
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,12),subplot_kw={'projection': ccrs.AzimuthalEquidistant(-55,60)})
ax.plot(data_split[i][:,4],data_split[i][:,3],lw=2,color=colorss[np.mod(i,len(colorss))],transform=ccrs.AzimuthalEquidistant(-55,60))
ax.scatter(data_split[i][0,4],data_split[i][0,3],transform=ccrs.AzimuthalEquidistant(-55,60))
ax.coastlines(transform=ccrs.AzimuthalEquidistant(-55,60)) 
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
        # angle[i][j] = np.arctan(dy[i][j]/dx[i][j])/(2*np.pi)*360
        angle[i][j] = np.arctan2(dy[i][j],dx[i][j])/(2*np.pi)*360
        # if (dx[i][j]<0) & (dy[i][j]<0):
        #     angle[i][j]=angle[i][j]-180
        # if (dx[i][j]<0) & (dy[i][j]>0):
        #     angle[i][j]=angle[i][j]+180
        dist[i][j]= np.sqrt(dy[i][j]**2+dx[i][j]**2)
#Calculate the difference of the angles between consecutive timestteps
for i in range(len(data_split)):
    dangle[i]= np.zeros(len(data_split[i])-2)
    for j in range(len(data_split[i])-2):        
        dangle[i][j]=angle[i][j+1]-angle[i][j]
        if (dangle[i][j]>180):
            dangle[i][j]=dangle[i][j]-360
        if  (dangle[i][j]<-180):
            dangle[i][j]=dangle[i][j]+360
        # dx[i][j] = distance.distance(((data_split[i][j+1,2],data_split[i][j,3]), (data_split[i][j,2],data_split[i][j,3]))).m
np.save(Path_data+'dangles.npy',dangle)       
np.save(Path_data+'data_split.npy', data_split) 
#%% Plot the angles  
mean_angle = np.mean(np.concatenate(dangle))
skew_angle = stats.skew(np.concatenate(dangle))
countzero=np.zeros(len(angle))
for i in range(len(angle)):
    countzero[i] = np.count_nonzero(dist[i]<100)
    # dangle[i] = dangle[i][~np.isnan(dangle[i])]
plt.figure()
plt.hist(np.concatenate(dangle)[~np.isnan(np.concatenate(dangle))],bins=100,range=[-50,50], stacked=True)
plt.title('Difference in angles between consecutive data points',fontsize=16)
plt.ylabel('Number of datapoints')
plt.xlabel('Angles (degrees)')
plt.text(-150,1500, 'Skewness = '+str(skew_angle)[:-14] +'\n'+ 'Mean angle = '+str(mean_angle)[:-14])
# plt.xlim([-160,-150])
plt.grid()
plt.show()


print('mean=', mean_angle)
print('skewness',skew_angle)
print('kurtosis=', stats.kurtosis(np.concatenate(dangle)))


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


dangle_resh = [None]*len(dangle)
D_resh=[None]*len(dangle)
for i in range(len(dangle)):
    D_resh[i] = np.insert(D[i],0,np.nan)
    dangle_resh[i] = np.insert(dangle[i],0,np.nan)
    D_resh[i]=np.append(D_resh[i],np.nan)
    dangle_resh[i] = np.append(dangle_resh[i],np.nan)
D_resh = pd.DataFrame(np.concatenate(D_resh))
dangle_resh= pd.DataFrame(np.concatenate(dangle_resh))
df['dangle']=dangle_resh
df['Diff']= D_resh
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df.to_pickle(Path_data+'df.pkl')

 #%%   
Nbin=40
# Mean_diff, xedges, yedges, binnumber = stats.binned_statistic_2d(df['lon'],df['lat'], df['Diff'],statistic='mean',bins=Nbin)
Mean_angles, xedges, yedges, binnumber = stats.binned_statistic_2d(df['lon'],df['lat'], df['dangle'],statistic='mean',bins=Nbin)
Mean_vel, xedges, yedges, binnumber = stats.binned_statistic_2d(df['lon'],df['lat'], df['speed']/100,statistic='mean',bins=Nbin)

mean_diff = np.zeros(np.shape(Mean_angles))
for i in range(len(Mean_angles)):
    phi = np.cos(Mean_angles[i]/360*2*np.pi) # The cosine of the angle
    tau = dt/(1-phi)    #correlation time scale
    mean_diff[i,:] = 1/n * Mean_vel[i]**2 *dt/(1-phi) #The diffusivity

x = np.linspace(295,315,Nbin)
y = np.linspace(55,65,Nbin)
X,Y = np.meshgrid(x,y)
#%%
plt.figure()
plt.contourf(X,Y,Mean_vel,cmap='rainbow')
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
plt.figure()
plt.hist(df['Diff'],bins=200,range=[0,100])
plt.show()

#%%
f_cor = 2*7.2921e-5*np.sin(60/360*2*pi)
T_cor = 2*pi/f_cor/3600
angle_alles = np.array(df['ve'])
T = 6*3600
N = len(angle_alles)
angles_fft = rfft(angle_alles)

xf = rfftfreq(len(angle_alles),T)


plt.figure()
plt.vlines([1/T_cor,1/12.4206],0,0.01,linestyles='dashed')
plt.plot(xf[10:]*3600, 2.0/N * np.abs(angles_fft)[10:])
plt.xscale('log')
plt.grid()
plt.title('Fourier spectrum of angles')
# plt.ylim(0,0.01)
# plt.xlim([0,0.0005])
plt.xlabel('f($h^{-1}$)')
plt.show()