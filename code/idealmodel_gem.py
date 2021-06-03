#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 10:08:42 2021

@author: tychobovenschen
"""
import numpy as np
import xarray as xr
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
    
    return pset
#%%        

for i in range(19):
    for j in range(19):
        if j==0:
            pset1 = idealmodel(At=i*0.2, Ae=1, phi=0,dt=1, lengthtime=100,sh=0,theta=0, relsize=10,v0=0,Ntraj=30, method='RK4')
        if j!=0:
            pset1=xr.concat((pset1,idealmodel(At=i*0.2, Ae=1, phi=0,dt=1, lengthtime=100,sh=0,theta=5*j, per=1, relsize=10,v0=0,Ntraj=30, method='RK4')),dim='theta')
    if i ==0:
        pset = pset1
    else:
        pset=xr.concat((pset,pset1),dim='A')