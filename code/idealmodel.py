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
import tqdm
import matplotlib.pyplot as plt
def idealmodel(Ntraj=30, dt=1, lengthtime=1000, sh=0.01, At=1,Ae=1, theta=0, phi=0, per=1,v0=0, relsize=1, method='Euler'):
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
    releaseX = np.linspace(-relsize,relsize,Ntraj)
    releaseY = np.linspace(-relsize,relsize,Ntraj)
    releaseX, releaseY = np.meshgrid(releaseX, releaseY)
    releaseX = np.ravel(releaseX)
    releaseY = np.ravel(releaseY)
    
    pset['x'][:,0]=releaseX
    pset['y'][:,0]=releaseY
    
    ####### EXECUTION #######
    #u-components:
    def u_tide(x,y,t):
        return At*np.cos(theta) * np.sin(per*t + phi*np.pi/180)
    def u_eddy(x,y,t):
        return Ae*np.cos(x) * np.sin(y) 
    #v-components:
    def v_tide(x,y,t):
        return At*np.sin(theta) * np.sin(per*t + phi*np.pi/180)
    def v_eddy(x,y,t):
        return -Ae*np.sin(x) * np.cos(y)
    def v_shear(x,y,t):
        return sh * x + v0
    def u_total(x,y,t):
        return u_tide(x,y,t)+u_eddy(x,y,t)
    def v_total(x,y,t):
        return v_tide(x,y,t)+v_eddy(x,y,t)+v_shear(x,y,t)
    #### EULER FORWARD####
    if method == 'Euler':
        for i in range(Nobs-1):
            t=i*dt
            pset.u[:,i] = u_total(pset.x[:,i], pset.y[:,i],t) #Sample the zonal velocity
            pset.v[:,i] = v_total(pset.x[:,i], pset.y[:,i],t)#Sample the meridional velocity
            pset.x[:,i+1] = pset.u[:,i]*dt+pset.x[:,i]    #Euler forward integration
            pset.y[:,i+1] = pset.v[:,i]*dt+pset.y[:,i] 
    elif method == 'RK4':
        for i in range(Nobs-1):
            t=i*dt
            k1 = u_total(pset.x[:,i], pset.y[:,i], t)
            l1 = v_total(pset.x[:,i], pset.y[:,i], t)
            k2 = u_total(pset.x[:,i]+dt*k1/2 ,pset.y[:,i]+dt*l1/2, t+dt/2)
            l2 = v_total(pset.x[:,i]+dt*k1/2 ,pset.y[:,i]+dt*l1/2, t+dt/2)
            k3 = u_total(pset.x[:,i]+dt*k2/2 ,pset.y[:,i]+dt*l2/2, t+dt/2)
            l3 = v_total(pset.x[:,i]+dt*k2/2 ,pset.y[:,i]+dt*l2/2, t+dt/2)
            k4 = u_total(pset.x[:,i]+dt*k3, pset.y[:,i]+dt*l3, t+dt)
            l4 = v_total(pset.x[:,i]+dt*k3, pset.y[:,i]+dt*l3, t+dt)
            k = (k1 + 2*k2 + 2*k3 + k4)/6
            l = (l1 + 2*l2 + 2*l3 + l4)/6
            pset.x[:,i+1] = pset.x[:,i] + dt*k
            pset.y[:,i+1] = pset.y[:,i] + dt*l
    else:
        print('Method must be either "Euler" or "RK4"')
    pset.attrs['theta'] = theta/np.pi*180
    pset.attrs['phi'] = phi
    pset.attrs['sh'] = sh
    pset.attrs['At'] = At
    pset.attrs['Ae'] = Ae
    
    
    # def u_rk4(x,y,t):
    #     return At*np.cos(theta) * np.sin(per*t + phi*np.pi/180) + Ae*np.cos(0.1 * x) * np.sin(0.1 * y)
    # def v_rk4(x,y,t):
    #     return -Ae*np.sin(0.1 * x) * np.cos(0.1 * y)
    
    # for t in range(Nobs-1):
    #     k1 = u_total(pset.x[:,t], pset.y[:,t], t)
    #     l1 = v_total(pset.x[:,t], pset.y[:,t], t)
    #     k2 = u_total(pset.x[:,t]+dt*k1/2 ,pset.y[:,t]+dt*l1/2, t+dt/2)
    #     l2 = v_total(pset.x[:,t]+dt*k1/2 ,pset.y[:,t]+dt*l1/2, t+dt/2)
    #     k3 = u_total(pset.x[:,t]+dt*k2/2 ,pset.y[:,t]+dt*l2/2, t+dt/2)
    #     l3 = v_total(pset.x[:,t]+dt*k2/2 ,pset.y[:,t]+dt*l2/2, t+dt/2)
    #     k4 = u_total(pset.x[:,t]+dt*k3, pset.y[:,t]+dt*l3, t+dt)
    #     l4 = v_total(pset.x[:,t]+dt*k3, pset.y[:,t]+dt*l3, t+dt)
    #     k = (k1 + 2*k2 + 2*k3 + k4)/6
    #     l = (l1 + 2*l2 + 2*l3 + l4)/6
    #     pset.x[:,t+1] = pset.x[:,t] + dt*k
    #     pset.y[:,t+1] = pset.y[:,t] + dt*l
    return pset

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
        
