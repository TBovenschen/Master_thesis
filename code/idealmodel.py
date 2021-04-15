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
    u_res = output.u[:,-2]

    dx_res = ((output.x[:,-2]-output.x[:,0]))
    v_res = (output.v[:,-2])

    dy_res = ((output.y[:,-2]-output.y[:,0]))
    """Function for calculating the diffusivity according to the method of Davis(1991)"""
    #Calculate diffusivity for every data point
    D_11 = - u_res * dx_res 
    D_12 = - u_res * dy_res 
    D_21 = - v_res * dx_res
    D_22 = - v_res * dy_res

    Nbin = 20
    #Bin the diffusivities in different grid cells
    D_11_bin,  xedges, yedges,binnumber_davis = binned_statistic_2d_new(output.x[:,-1],\
                    output.y[:,-1],D_11.astype('float'), statistic='nanmean',range=([-200,200],[-200,200]),bins=Nbin)
    D_12_bin,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(output.x[:,-1],\
                    output.y[:,-1], D_12.astype('float'), statistic='nanmean',range=([-200,200],[-200,200]),bins=Nbin, expand_binnumbers=True)
    D_21_bin,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(output.x[:,-1],\
                    output.y[:,-1], D_21.astype('float'), statistic='nanmean',range=([-200,200],[-200,200]),bins=Nbin, expand_binnumbers=True)
    D_22_bin,  xedges, yedges, binnumber_davis = binned_statistic_2d_new(output.x[:,-1],\
                    output.y[:,-1], D_22.astype('float'), statistic='nanmean',range=([-200,200],[-200,200]),bins=Nbin, expand_binnumbers=True)
    print(D_22_bin)


    D_11_bin=np.swapaxes(D_11_bin, 0, 1)
    D_12_bin=np.swapaxes(D_12_bin, 0, 1)
    D_21_bin=np.swapaxes(D_21_bin, 0, 1)
    D_22_bin=np.swapaxes(D_22_bin, 0, 1)

    #Define grid
    
    x = np.linspace(0, 20,Nbin)
    y = np.linspace(0, 20,Nbin)
    X, Y = np.meshgrid(x, y)

        # Add the diffusivity components to 1 xarray dataset

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
Ntraj=30 #Number of particles released (in x and in y direction) so Ntraj**2 is total
dt = 1   #Delta time
lengthtime = 1000
Nobs =int(lengthtime/dt) #Number of data points per particle
x = np.linspace(-500,500,200)  #X-grid
y = np.linspace(-500,500,200) #Y-grid

t = np.linspace(0,Nobs*dt, Nobs)    #Time grid

X,Y,T = np.meshgrid(x,y,t)
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

tides1['u'] = (('y','x','time'),2*np.sin(0.1*T))
tides2['u'] = (('y','x','time'),2*np.sin(0.2*T))

#%%#PARTICLE SET:
pset =xr.Dataset({'x':(['traj','obs'], np.nan *np.zeros((Ntraj**2,Nobs))),
                      'y':(['traj','obs'],np.nan *np.zeros((Ntraj**2,Nobs))),
                      'u':(['traj','obs'],np.zeros((Ntraj**2,Nobs))),
                      'v':(['traj','obs'],np.zeros((Ntraj**2,Nobs)))})
releaseX = np.linspace(-15,16,Ntraj)
releaseY = np.linspace(-15,16,Ntraj)
releaseX, releaseY = np.meshgrid(releaseX, releaseY)
releaseX = np.ravel(releaseX)
releaseY = np.ravel(releaseY)

pset['x'][:,0]=releaseX
pset['y'][:,0]=releaseY

fieldset = tides1+BC
# for i in range(len(pset.obs)-1):
    # pset['x'][:,i+1] = pset.x[:,i]+fieldset.u.sel(x=pset.x[:,i], y = pset.y[:,i], time=i, method='nearest') *dt
    # pset['y'][:,i+1] = pset.y[:,i]+fieldset.v.sel(x=pset.x[:,i], y = pset.y[:,i], time=i,  method='nearest') *dt

#%% EXECUTION


################### 1 tidal component: ###########################################
    #u = 2 sin(0.1t)
    # v = 2-0.04x
# for i in range(len(pset.traj)):
#     pset['x'][i]= releaseX[i]-20*np.cos(0.1*t) + 20
#     pset['y'][i]= releaseY[i]+(2 - 0.04*releaseX[i] -0.8)*t + 8 *np.sin(0.1*t)
    
# ################# 2 tidal components: ################################
#    u = 2 sin(0.1t) + 2 sin(0.2t)
#     v = 2-0.04x
# for i in range(len(pset.traj)):
#     pset['x'][i]= releaseX[i]-20*np.cos(0.1*pset.obs)  - 10*np.cos(0.2*pset.obs) + 30
#     pset['y'][i]= releaseY[i]+(2 - 0.04*(releaseX[i] + 30))*pset.obs + 8*np.sin(0.1*pset.obs) + 4 * np.sin(0.2*pset.obs)
# pset['u'] =pset.u + fieldset.u.sel(x=pset.x, y =pset.y,time=pset.obs,  method='nearest').values

#################### Tidal components in different directioins: ##################
#u = 2 sin(0.1t)
# v = 2-0.04x + sin(0.1t)
# for i in range(len(pset.traj)):
#     pset['x'][i]= releaseX[i]-20*np.cos(0.1*pset.obs) + 20
#     pset['y'][i]= releaseY[i]+(2 - 0.04*(releaseX[i] + 20))*pset.obs + 8 *np.sin(0.1*pset.obs) - 20 * np.cos(0.2*pset.obs) + 20






#################### Eddy field: ##################################################
    
#FIELDSET (not necessary for analytical trajectories)
eddies =xr.Dataset({'u':(['y','x', 'time'],np.zeros((len(y),len(x), Nobs))),
                      'v':(['y','x', 'time'],np.zeros((len(y),len(x), Nobs)))},                     
                  coords={
                "x": (["x"],x),
                  "y": (["y"],y),
                  "time": (["time"],np.linspace(0,Nobs*dt,Nobs)),},)
    
A=1
u0=0
v0=0
dt = 1
t=np.linspace(0,Nobs*dt,Nobs)

X,Y,T = np.meshgrid(x,y,t)
eddies['u'] = (('y','x','time'),(A * np.cos(0.1 * X) * np.sin(0.1 * Y) + 1*np.sin(0.1*T)))
eddies['v'] = (('y','x','time'),-A * np.sin(0.1*X) * np.cos(0.1* Y) + v0)
    
# eddies['u'] = (('y','x','time'),(A * np.sin(0.1 * X) * np.sin(0.1 * Y)))
# eddies['v'] = (('y','x','time'),A * np.sin(0.1*X) * np.sin(0.1* Y) + v0)


pset.x[:,0] = releaseX #Set initial position
pset.y[:,0]= releaseY   #Set initial y-position

for t in range(Nobs-1):
    # pset.u[:,t] = eddies.u.sel(x=pset.x[:,t], y=pset.y[:,t], method='nearest') #Sample the zonal velocity
    pset.u[:,t] = A * np.cos(0.1 * pset.x[:,t]) * np.sin(0.1 * pset.y[:,t])+np.sin(0.1*t)+ u0 #Sample the zonal velocity
    pset.v[:,t] = -A * np.sin(0.1 * pset.x[:,t]) * np.cos(0.1 * pset.y[:,t]) + v0 #Sample the zonal velocity
    pset.x[:,t+1] = pset.u[:,t]*dt+pset.x[:,t]    #Euler forward integration
    pset.y[:,t+1] = pset.v[:,t]*dt+pset.y[:,t] 

################### EDDY + SHEAR + TIDE ######
# #FIELDSET (not necessary for analytical trajectories)
# eddies =xr.Dataset({'u':(['y','x', 'time'],np.zeros((len(y),len(x), Nobs))),
#                       'v':(['y','x', 'time'],np.zeros((len(y),len(x), Nobs)))},                     
#                   coords={
#                 "x": (["x"],x),
#                   "y": (["y"],y),
#                   "time": (["time"],np.linspace(0,Nobs*dt,Nobs)),},)
    
# A=1
# u0=0
# v0=0
# dt = 1
# t=np.linspace(0,Nobs*dt,Nobs)

# X,Y,T = np.meshgrid(x,y,t)
# eddies['u'] = (('y','x','time'),(A * np.cos(0.1 * X) * np.sin(0.1 * Y) + 1*np.sin(0.1*T)))
# eddies['v'] = (('y','x','time'),-A * np.sin(0.1*X) * np.cos(0.1* Y) + 0.01*X+v0)

# # eddies['u'] = (('y','x','time'),(A * np.sin(0.1 * X) * np.sin(0.1 * Y)))
# # eddies['v'] = (('y','x','time'),A * np.sin(0.1*X) * np.sin(0.1* Y) + v0)


# pset.x[:,0] = releaseX #Set initial position
# pset.y[:,0]= releaseY   #Set initial y-position

# for t in range(Nobs-1):
#     # pset.u[:,t] = eddies.u.sel(x=pset.x[:,t], y=pset.y[:,t], method='nearest') #Sample the zonal velocity
#     pset.u[:,t] = A * np.cos(0.1 * pset.x[:,t]) * np.sin(0.1 * pset.y[:,t])+np.sin(0.1*t) #Sample the zonal velocity
#     pset.v[:,t] = -A * np.sin(0.1 * pset.x[:,t]) * np.cos(0.1 * pset.y[:,t])+ 0.01*pset.x[:,t]  + v0 #Sample the zonal velocity
#     pset.x[:,t+1] = pset.u[:,t]*dt+pset.x[:,t]    #Euler forward integration
#     pset.y[:,t+1] = pset.v[:,t]*dt+pset.y[:,t] 

    #%%
# plt.figure()
# plt.contourf(v, cmap='coolwarm')


#%%ANIMATE
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
fig, ax = plt.subplots()
xdata, ydata = [], []
ln = ax.scatter(xdata,ydata,sizes=[5],zorder=1)
# cntr = ax.pcolormesh(x, y, eddies.v.isel(time=0),cmap='coolwarm',vmin=-2, vmax=2,zorder=0, shading='flat')
# 
plt.xlim(-500,500)
plt.ylim(-500,500)
annotation = ax.annotate('Time=', xy=(-55, 50))
annotation.set_animated(True)
# plt.colorbar(cntr,label='zonal velocity')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Eddies + tides + shear ')
def init():
    return ln, annotation, #cntr,

def update(i):
    annotation.set_text('Time='+str(i))
    ln.set_offsets(np.c_[pset.x.isel(obs=i), pset.y.isel(obs=i)])
    # cntr[0] = ax.pcolormesh(eddies.x, eddies.y, eddies.u.isel(time=i),cmap='coolwarm',vmin=-2, vmax=2,zorder=0)
    # cntr.set_array(eddies.v.isel(time=i).values[:-1,:-1].ravel())
    return ln, annotation, #cntr,



ani = FuncAnimation(fig, update, frames=1000,
                    init_func=init, blit=True, interval = 1)

# ani.save('/Users/tychobovenschen/Documents/MasterJaar2/Thesis/plots/eddies_tides_shear.mp4', writer=writer)
# plt.show()


#%%
k_S, eig_val, eig_vec = k_davis(pset)
#%%
scale=10
X, Y = np.meshgrid(x,y)
XY = np.column_stack((X.ravel(), Y.ravel()))
#calculate largest and smalles eigenvalue
fig =plt.figure()
ax = plt.axes()
plt.xlim(-200,200)
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

#%%Calculate variancex


plt.figure()
s = np.var(pset.x, axis=0)
diff = 1/2*(s[1:]-s[:-1])/dt
s.plot()
plt.xlabel('time', size =14)
plt.title('Variance of x-displacement of tide+eddy simulation', size = 16)
plt.ylabel('$\sigma^2$', size =14)

plt.figure()
diff.plot()
# plt.ylim(-1,1)
plt.xlabel('time', size =14)
plt.title('Diffusivity in x-direction of tide+eddy simulation', size = 16)
plt.ylabel('$k$', size =14)