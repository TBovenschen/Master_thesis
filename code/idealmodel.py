#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:08:42 2021

@author: tychobovenschen
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
def idealmodel(Ntraj=30, dt=1, lengthtime=1000, shear=True, tides=True, eddies=True, sh=0.01, A=1, theta=0, phi=0):
    """"A model where particles are advected on a simple velocity field. The velocity field 
    can consist of 3 different components that can all be turned on or off separately 
    (shear, tides, eddies). Furthermore a number of parameters can be changed:
        shear,tides,eddies: True is on, False is off
        
        Ntraj       = number of particles (default=30)
        dt          = timestep (default = 1)
        lengthtime  = the length of the run (default=1000)
        sh          = A parameter that defines the strenght of the shear
                                (the higher, the more shear in the zonal direction)
        A          = The amplitude of the tides relative to the eddies (default = 1)
        phi         = The phase difference between the tides and eddies (default = 0)
        theta       = The angle of the tides relative to the x-axis (default is 0)
        """
    Nobs =int(lengthtime/dt) #Number of data points per particle
    x = np.linspace(-500,500,200)  #X-grid

    y = np.linspace(-500,500,200) #Y-grid
    
    t = np.linspace(0,Nobs*dt, Nobs)    #Time grid
    X,Y,T = np.meshgrid(x,y,t)
    theta = theta/180*np.pi

######PARTICLE SET:
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
    
    ####### EXECUTION #######
    #u-components:
    def u_tide(t):
        return A*np.cos(theta) * np.sin(0.1*t + phi*np.pi/180)
    def u_eddy(t):
        return np.cos(0.1 * pset.x[:,t]) * np.sin(0.1 * pset.y[:,t])
    #v-components:
    def v_tide(t):
        return A*np.sin(theta) * np.sin(0.1*t + phi*np.pi/180)
    def v_eddy(t):
        return -np.sin(0.1 * pset.x[:,t]) * np.cos(0.1 * pset.y[:,t])
    def v_shear(t):
        return sh*pset.x[:,t]
    
    if (shear==True) & (tides==True) & (eddies==False):  #Tides + SHEAR
        simu_type='ST'
        for t in range(Nobs-1):
            pset.u[:,t] = u_tide(t) #Sample the zonal velocity
            pset.v[:,t] = v_tide(t) + v_shear(t)             # Sample the meridional velocity
            pset.x[:,t+1] = pset.u[:,t]*dt+pset.x[:,t]    #Euler forward integration
            pset.y[:,t+1] = pset.v[:,t]*dt+pset.y[:,t] 
          
    elif (eddies==True) & (tides==True) & (shear==False):  #EDDIES + TIDES
        A=1
        simu_type='TE'
        for t in range(Nobs-1):
            pset.u[:,t] = u_tide(t) + u_eddy(t) #Sample the zonal velocity
            pset.v[:,t] = v_tide(t) + v_eddy(t)             # Sample the meridional velocity
            pset.x[:,t+1] = pset.u[:,t]*dt+pset.x[:,t]    #Euler forward integration
            pset.y[:,t+1] = pset.v[:,t]*dt+pset.y[:,t] 
    elif (eddies==False) & (tides==True) & (shear==False):  #TIDES
        A=1
        simu_type='T'
        for t in range(Nobs-1):
            pset.u[:,t] = u_tide(t) #Sample the zonal velocity
            pset.v[:,t] = v_tide(t)             # Sample the meridional velocity
            pset.x[:,t+1] = pset.u[:,t]*dt+pset.x[:,t]    #Euler forward integration
            pset.y[:,t+1] = pset.v[:,t]*dt+pset.y[:,t] 
            
    elif (eddies==True) & (shear==True) & (tides==False): # EDDIES + SHEAR
        A=1  # amplitude of eddies
        simu_type='SE'
        for t in range(Nobs-1):
            pset.u[:,t] = u_eddy(t) #Sample the zonal velocity
            pset.v[:,t] = v_eddy(t) + v_shear(t) #Sample the meridional velocity
            pset.x[:,t+1] = pset.u[:,t]*dt+pset.x[:,t]    #Euler forward integration
            pset.y[:,t+1] = pset.v[:,t]*dt+pset.y[:,t] 
    
    elif (eddies==True) & (tides==True) & (shear==True):  ### EDDIES + SHEAR + TIDES
        A=1  # amplitude of eddies
        simu_type='STE'
        for t in range(Nobs-1):
            pset.u[:,t] = u_tide(t) + u_eddy(t) #Sample the zonal velocity
            pset.v[:,t] = v_eddy(t) + v_shear(t) + v_tide(t)#Sample the meridional velocity
            pset.x[:,t+1] = pset.u[:,t]*dt+pset.x[:,t]    #Euler forward integration
            pset.y[:,t+1] = pset.v[:,t]*dt+pset.y[:,t] 
    else:
        print('Invalid configuration')
    pset.attrs['simu_type']=simu_type
    pset.attrs['theta'] = theta/np.pi*180
    pset.attrs['phi'] = phi
    pset.attrs['sh'] = sh
    pset.attrs['A'] = A
    return pset

  
        



pset = idealmodel(eddies=True, tides=True, shear=True, theta=0, lengthtime=1000,sh=0.01)
#%%ANIMATE
x = np.linspace(-500,500,200)  #X-grid
dt=1
y = np.linspace(-500,500,200) #Y-grid
Nobs = 1000
eddies =xr.Dataset({'u':(['y','x', 'time'],np.zeros((len(y),len(x), Nobs))),
                      'v':(['y','x', 'time'],np.zeros((len(y),len(x), Nobs)))},                     
                  coords={
                "x": (["x"],x),
                  "y": (["y"],y),
                  "time": (["time"],np.linspace(0,Nobs*dt,Nobs)),},)
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
fig, ax = plt.subplots()
xdata, ydata = [], []
ln = ax.scatter(xdata,ydata,sizes=[5],zorder=1)
cntr = ax.pcolormesh(x, y, eddies.v.isel(time=0),cmap='coolwarm',vmin=-2, vmax=2,zorder=0, shading='flat')
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
        #%%Calculate variancex
dt=1
pset = idealmodel(eddies=True, tides=True, shear=True, theta=60, lengthtime=2000,sh=0.01)
plt.figure()
for i in range(9):
    pset = idealmodel(eddies=True, tides=True, shear=True, theta=i*10, lengthtime=2000,sh=0.01)
    sx = np.var(pset.x, axis=0)
    plt.plot(sx, label=str(pset.theta))
plt.legend()
plt.title()
plt.show()
def plotandsave(pset, save=False):
    direc = 'Documents/MasterJaar2/Thesis/plots/idealmodel/simu_%s_%sdeg_phase%s/' %(pset.simu_type,int(pset.theta),pset.phi)
    if save==True:
        try:
            os.mkdir(direc)
        except(FileExistsError):
            print('Directory already existed, files are overwritten')
    plt.figure()
    sx = np.var(pset.x, axis=0)
    diffx = 1/2*(sx[1:]-sx[:-1])/dt
    sx.plot()
    plt.xlabel('time', size =14)
    plt.title('Variance of x-displacement of %s simulation' %pset.simu_type, size = 16)
    plt.ylabel('$\sigma^2$', size =14)
    if save==True:
        plt.savefig(direc+'sx.png')
    plt.figure()
    diffx.plot()
    # plt.ylim(-1,1)
    plt.xlabel('time', size =14)
    plt.title('Diffusivity in x-direction of %s simulation' %pset.simu_type, size = 16)
    plt.ylabel('$k$', size =14)
    if save==True:
        plt.savefig(direc+'diffx.png')

    plt.figure()
    sy = np.var(pset.y, axis=0)
    diffy = 1/2*(sy[1:]-sy[:-1])/dt
    sy.plot()
    plt.xlabel('time', size =14)
    plt.title('Variance of y-displacement of %s simulation' %pset.simu_type, size = 16)
    plt.ylabel('$\sigma^2$', size =14)
    if save==True:
        plt.savefig(direc+'sy.png')
    plt.figure()
    diffy.plot()
    # plt.ylim(-1,1)
    plt.xlabel('time', size =14)
    plt.title('Diffusivity in y-direction of %s simulation' %pset.simu_type, size = 16)
    plt.ylabel('$k$', size =14)
    if save==True:
            plt.savefig(direc+'diffy.png')
    plt.figure()
    plt.plot(sy/sx)
    plt.title('Anisotropy of variance in %s simulation' %pset.simu_type, size=16)
    if save==True:
            pset.to_netcdf(direc+'pset.nc')
    
plotandsave(pset)