#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:32:29 2021

@author: tychobovenschen
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.collections import EllipseCollection
from scipy import linalg
from binned_statistic2 import binned_statistic_2d_new

    
def k_davis(output):
    u_res = output.u[:,-1]

    dx_res = ((output.x[:,-1]-output.x[:,0]))
    v_res = (output.v[:,-1])

    dy_res = ((output.y[:,-1]-output.y[:,0]))
    """Function for calculating the diffusivity according to the method of Davis(1991)"""
    #Calculate diffusivity for every data point
    D_11 = - u_res * dx_res 
    D_12 = - u_res * dy_res 
    D_21 = - v_res * dx_res
    D_22 = - v_res * dy_res

    Nbin = 20
    #Bin the diffusivities in different grid cells
    D_11_bin,  xedges, yedges,binnumber_davis = binned_statistic_2d_new(output.x[:,-1],\
                    output.y[:,-1],D_11.astype('float'), statistic='nanmean',range=([0,20],[0,20]),bins=Nbin)
    D_12_bin,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(output.x[:,-1],\
                    output.y[:,-1], D_12.astype('float'), statistic='nanmean',range=([0,20],[0,20]),bins=Nbin, expand_binnumbers=True)
    D_21_bin,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(output.x[:,-1],\
                    output.y[:,-1], D_21.astype('float'), statistic='nanmean',range=([0,20],[0,20]),bins=Nbin, expand_binnumbers=True)
    D_22_bin,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(output.x[:,-1],\
                    output.y[:,-1], D_22.astype('float'), statistic='nanmean',range=([0,20],[0,20]),bins=Nbin, expand_binnumbers=True)
    print(D_22_bin)


    D_11_bin=np.swapaxes(D_11_bin, 0, 1)
    D_12_bin=np.swapaxes(D_12_bin, 0, 1)
    D_21_bin=np.swapaxes(D_21_bin, 0, 1)
    D_22_bin=np.swapaxes(D_22_bin, 0, 1)

    #Define grid
    
    x = np.linspace(0, 20,Nbin)
    y = np.linspace(0, 20,Nbin)
    X, Y = np.meshgrid(x, y)

        #%% Add the diffusivity components to 1 xarray dataset

    #Symmetric part:
    k_S =xr.Dataset({'k11':(['lat','lon'],D_11_bin),
                          'k12':(['lat','lon'],(D_12_bin+D_21_bin)/2),
                          'k21':(['lat','lon'],(D_21_bin+D_12_bin)/2),
                          'k22':(['lat','lon'],D_22_bin)},                     
                      coords={
                    "lon": (["lon"],x),
                      "lat": (["lat"],y)},)


    #Create arrays for eigen values and eigenvectors:
    eig_val = np.zeros((len(y),len(x),2))
    eig_vec = np.zeros((len(y),len(x),2,2))

    #Calculate the eigenvalues and eigenvectors for every bin:
    for i in range(Nbin):
        for j in range(Nbin):
            try:
                eig_val[i,j,:], eig_vec[i,j,:,:] = linalg.eig(k_S.isel(lat=i,lon=j).\
                    to_array().values.reshape(2,2),check_finite=True)
            except (ValueError): #If there are nan values in the diffusivity: fill eigenvalue and eigenvectors with nans
                eig_val[i,j,:]=[np.nan, np.nan]
                eig_vec[i,j,:,:]=([np.nan, np.nan],[np.nan, np.nan])
                continue


    #Make an xarray dataset of the eigenvalues and eigenvectors:
    eig_val =xr.Dataset({'labda':(['lat','lon','i'],eig_val)},
                      coords={
                    "lon": (["lon"],x),
                      "lat": (["lat"],y)},)
    eig_vec =xr.Dataset({'mu':(['lat','lon','i','j'],eig_vec)},
                      coords={
                    "lon": (["lon"],x),
                      "lat": (["lat"],y)},
                      attrs={
                          "title": 'Eigen vectors per grid cell'})

    return k_S, eig_val, eig_vec
#%%

#Parameters:
Ntraj=10 #Number of particles released (in x and in y direction) so Ntraj**2 is total
Nobs =1000  #Number of data points per particle
dt = 0.1    #Delta time
x = np.linspace(0,200,200)  #X-grid
y = np.linspace(0,200,200) #Y-grid

t = np.linspace(0,Nobs*dt, Nobs)    #Time grid

#FIELDSET (not necessary for analytical trajectories)
BC =xr.Dataset({'u':(['y','x', 'time'],np.zeros((len(y),len(x), Nobs))),
                      'v':(['y','x', 'time'],np.zeros((len(y),len(x), Nobs)))},                     
                  coords={
                "x": (["x"],x),
                  "y": (["y"],y),
                  "time": (["time"],np.linspace(0,Nobs,Nobs)),},)




tides1 =xr.Dataset({'u':(['y','x', 'time'],np.zeros((len(y),len(x), Nobs))),
                      'v':(['y','x', 'time'],np.zeros((len(y),len(x), Nobs)))},                     
                  coords={
                "x": (["x"],x),
                  "y": (["y"],y),
                  "time": (["time"],np.linspace(0,Nobs,Nobs)),},)
tides2 =xr.Dataset({'u':(['y','x', 'time'],np.zeros((len(y),len(x), Nobs))),
                      'v':(['y','x', 'time'],np.zeros((len(y),len(x), Nobs)))},                     
                  coords={
                "x": (["x"],x),
                  "y": (["y"],y),
                  "time": (["time"],np.linspace(0,Nobs,Nobs)),},)
for i in range(Nobs):
    BC['v'][:,:,i] = BC.v[:,:,i] +( 2 - x * 0.04)

tides1['u'] = tides1.u + 2*np.sin(0.1*tides1.time)
tides2['u'] = tides1.u + 2*np.sin(0.2*tides2.time)

#%%
#PARTICLE SET:
pset =xr.Dataset({'x':(['traj','obs'], np.nan *np.zeros((Ntraj**2,Nobs))),
                      'y':(['traj','obs'],np.nan *np.zeros((Ntraj**2,Nobs))),
                      'u':(['traj','obs'],np.zeros((Ntraj**2,Nobs))),
                      'v':(['traj','obs'],np.zeros((Ntraj**2,Nobs)))})
releaseX = np.linspace(0,20,Ntraj)
releaseY = np.linspace(0,20,Ntraj)
releaseX, releaseY = np.meshgrid(releaseX, releaseY)
releaseX = np.ravel(releaseX)
releaseY = np.ravel(releaseY)

pset['x'][:,0]=releaseX
pset['y'][:,0]=releaseY










# fieldset = tides+BC
# for i in range(len(pset.obs)-1):
    # pset['x'][:,i+1] = pset.x[:,i]+fieldset.u.sel(x=pset.x[:,i], y = pset.y[:,i], time=i, method='nearest') *dt
    # pset['y'][:,i+1] = pset.y[:,i]+fieldset.v.sel(x=pset.x[:,i], y = pset.y[:,i], time=i,  method='nearest') *dt

#%%
#1 tidal component:
# for i in range(len(pset.traj)):
    # pset['x'][i]= releaseX[i]-20*np.cos(0.1*pset.obs) + 20
    # pset['y'][i]= releaseY[i]+(2 - 0.04*releaseX[i] -0.8)*pset.obs + 8 *np.sin(0.1*pset.obs)
    
#2 tidal components:
   #u = 2 sin(0.1t) + 2 sin(0.2t)
   # v = 2-0.04x
# for i in range(len(pset.traj)):
    # pset['x'][i]= releaseX[i]-20*np.cos(0.1*pset.obs)  - 10*np.cos(0.2*pset.obs) + 30
    # pset['y'][i]= releaseY[i]+(2 - 0.04*(releaseX[i] + 30))*pset.obs + 8*np.sin(0.1*pset.obs) + 4 * np.sin(0.2*pset.obs)
# pset['u'] =pset.u + fieldset.u.sel(x=pset.x, y =pset.y,time=pset.obs,  method='nearest').values

#Tidal components in different directioins:
#u = 2 sin(0.1t)
# v = 2-0.04x + cos(0.2t)
for i in range(len(pset.traj)):
    pset['x'][i]= releaseX[i]-20*np.cos(0.1*pset.obs) + 20
    pset['y'][i]= releaseY[i]+(2 - 0.04*(releaseX[i] + 20))*pset.obs + 8*np.sin(0.1*pset.obs) + 5 * np.sin(0.2*pset.obs)

#Eddy field:
A=1
u0=0
v0=0
X,Y = np.meshgrid(x,y)
u = A * np.cos(0.1 * X) * np.sin(0.1 * Y) + u0
v = -A * np.sin(0.1*X) * np.cos(0.1* Y) + v0

x0 = releaseX
y0= releaseY
t= pset.obs/10
# (-100+np.cos( np.sin(t*x0[i])) + np.cos(np.sin(t*y0[i]))-np.sqrt(400* np.cos(np.sin(t*x0[i]))+(100-np.cos(np.sin(t*x0[i]))-np.cos(np.sin(t*y0[i])))**2/(2*np.cos(np.sin(t)))))


# 1/2*(100/(np.cos(np.sin(t)))+x0+y0+ (np.sqrt(400* np.cos(np.sin(t*x0))+
#                                              (100-np.cos(np.sin(t*x0[i]))-np.cos(np.sin(t*y0[i])))**2))/(2*np.cos(np.sin(t))))
for i in range(len(pset.traj)):
    pset['x'][i]= (-100+np.cos( np.sin(t*x0[i])) + np.cos(np.sin(t*y0[i]))+
                   np.sqrt(400* np.cos(np.sin(t*x0[i]))+(100-np.cos(np.sin(t*x0[i]))-np.cos(np.sin(t*y0[i])))**2/(2*np.cos(np.sin(t)))))
    pset['y'][i]= 1/2*(100/(np.cos(np.sin(t)))+x0[i]+y0[i]+ (np.sqrt(400* np.cos(np.sin(t*x0[i]))+
                                             (100-np.cos(np.sin(t*x0[i]))-np.cos(np.sin(t*y0[i])))**2))/(2*np.cos(np.sin(t))))


#%%
plt.figure()
plt.contourf(v, cmap='coolwarm')
#%%
k_S, eig_val, eig_vec = k_davis(pset)

#%%
from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots()
xdata, ydata = [], []
ln = ax.scatter(xdata,ydata,sizes=[5])
# plt.contourf(BC.x, BC.y, BC.v.isel(time=0),cmap='coolwarm', levels=100,zorder=0)

plt.xlim(-200,100)
plt.ylim(-200,1000)
annotation = ax.annotate('Time=', xy=(75, 20))
annotation.set_animated(True)
def init():
    return ln, annotation

def update(i):
    annotation.set_text('Time='+str(i))
    ln.set_offsets(np.c_[pset.x.isel(obs=i), pset.y.isel(obs=i)])
    return ln, annotation

ani = FuncAnimation(fig, update, frames=1000,
                    init_func=init, blit=True, interval = 100)
plt.colorbar(label='Meridional velocity')

plt.show()



#%%
scale=10
X, Y = np.meshgrid(x,y)
XY = np.column_stack((X.ravel(), Y.ravel()))
#calculate largest and smalles eigenvalue
fig =plt.figure()
ax = plt.axes()
plt.xlim(0,50)
plt.ylim(-100,100)
index_major= abs(eig_val.labda).argmax(dim='i',skipna=False)
index_minor= abs(eig_val.labda).argmin(dim='i',skipna=False)
ells = EllipseCollection(eig_val.labda.isel(i=index_major)/scale,eig_val.labda.isel(i=index_minor)/scale,\
                      np.arctan2(eig_vec.mu.isel(i=index_minor,j=0),eig_vec.mu.isel(i=index_minor,j=1)).values/np.pi*180, units='x', offsets=XY, facecolors='None',edgecolors='tab:red',transOffset=ax.transData, offset_position='screen')        
# bar = AnchoredSizeBar(ax.transData, size=1, label='8000 $m^2/s$', color = 'tab:red', loc=3, frameon=False)
# ax.add_artist(bar)
# ells.set_linewidth(5)
# signs= np.sign(eig_val.labda.isel(i=index_major)).stack(z=['lat', 'lon'])
# colors = ['yellow','red', 'blue']
# ax.autoscale_view()

# ells.set_array([colors[i] for i in test])
# ells.set_cmap('coolwarm')
# ells.set_edgecolor(np.array([colors[i] for i in signs.fillna(0).values.astype(int)]))

ax.add_collection(ells)
plt.show()

# from plotting_functions import *
# plot_ellipse(eig_val, eig_vec, scale=1)