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
def idealmodel(Ntraj=30, dt=1, lengthtime=1000, sh=0.01, At=1,Ae=1, theta=0, phi=0, per=0.1, method='Euler'):
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
    def u_tide(x,y,t):
        return At*np.cos(theta) * np.sin(per*t + phi*np.pi/180)
    def u_eddy(x,y,t):
        return Ae*np.cos(0.1 * x) * np.sin(0.1 * y)
    #v-components:
    def v_tide(x,y,t):
        return At*np.sin(theta) * np.sin(per*t + phi*np.pi/180)
    def v_eddy(x,y,t):
        return -Ae*np.sin(0.1 *x) * np.cos(0.1 * y)
    def v_shear(x,y,t):
        return sh * x
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
        
########## RK4 ############


# pset = idealmodel(theta=0, lengthtime=1000,sh=0.01)
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



ani = FuncAnimation(fig, update, frames=2000,
                    init_func=init, blit=True, interval = 100)

# ani.save('/Users/tychobovenschen/Documents/MasterJaar2/Thesis/plots/eddies_tides_shear.mp4', writer=writer)
        #%%Calculate variancex
dt=1
for i in tqdm.tqdm(range(11)):
    if i==0:
        pset = idealmodel(lengthtime=1000, method='Euler')
    if i!=0:
        pset=xr.concat((pset,idealmodel(lengthtime=1000, theta=i*10, method='Euler')),dim='A')
#%%
sx=np.var(pset.x,axis=1)
sy=np.var(pset.y,axis=1)

diffx = 1/2*(sx[:,1:]-sx[:,:-1])/dt

plt.figure()
for i in range(9):
    sx.isel(A=i).plot(label=r'c='+str(i*0.02)[:5])
plt.xlabel('time')
plt.ylabel('variance of x')
plt.title(r'The variance of x for an ETS field with varying tidal period', size=22)
plt.legend()
plt.show()

plt.figure()
for i in range(9):
    (sx/sy).isel(A=i).plot(label=r'theta='+str(i*10)[:5])
plt.xlabel('time')
plt.ylabel('$s_x/s_y$')
plt.title(r'The anisotropy of the variance for an ETS field with varying shear', size=22)
plt.legend()
plt.show()


plt.figure()
for i in range(9):
    sy.isel(A=i).plot(label=r'A='+str(i*0.4)[:3])
plt.xlabel('time')
plt.ylabel('diffusivity')
plt.title(r'The diffusivity of x for an ETS field with varying tidal amplitude', size=22)
plt.legend()
plt.show()


#%%

plt.rcParams.update({'font.size': 16})


fig= plt.figure()
ax=plt.axes()
for i in range(10):
    pset = idealmodel(eddies=True, tides=True, shear=True, theta=0,phi=i*10, lengthtime=2000,sh=0.01)
    sx = np.var(pset.x, axis=0)
    ax.plot(sx, label=r'$\theta=$'+str(int(pset.theta))+'$^{\circ}$')
ax.legend()
ax.set_title(r'The variance in the x-direction for an ETS field with varying $\theta$', size=22)
plt.xlabel('time',size=18)
plt.ylabel('variance',size=18)
fig.show()

method='Euler'
dt = 1
pset = idealmodel(At=1,theta=0, Ae=1, phi=0,dt=1, lengthtime=2000,sh=0.01, method='RK4')

sy=np.var(pset.y,axis=0)
sx=np.var(pset.x,axis=0)

plt.plot(np.linspace(0,1000, len(sx)),sx, label=method+str(dt))
plt.plot(np.linspace(0,1000, len(sx)),sx/sy, label=method+str(dt))

plt.legend()
plotandsave(pset)

