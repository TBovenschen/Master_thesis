#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import matplotlib.pyplot as plt
import cartopy 
import numpy as np
import xarray as xr
from datetime import datetime
from datetime import timedelta
from glob import glob
from parcels import (grid, Field, FieldSet, VectorField, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4, ErrorCode, ParticleFile, Variable, plotTrajectoriesFile)


#Open the file with the buoy data:
file = open('df.pkl','rb')
df = pd.read_pickle('df.pkl')


# In[2]:


"""
Function for easily setting fieldsets from Global Ocean Physical Reanalysis data in Parcels.
"""
def create(startDate, months, lonRange=(-65, -45), latRange=(55,65), **kwargs):
    """
    Creates a parcels.FieldSet object from hydrodynamic data in a netCDF file.
    
    Parameters
    ----------
    startDate : str, int
        String or int indicating the first date to load (YYYYMM)
    months : int
        Number of months to load    

        
    Returns
    ----------
    parcels.FieldSet    
    """
    readDir = "/data/oceanparcels/input_data/CMEMS/GLOBAL_REANALYSIS_PHY_001_030_monthly/"
    fieldFiles = sorted(glob(readDir + "*.nc"))
    startFile = glob(readDir + f"mercatorglorys12v1_gl12_mean_{startDate}.nc")
    assert len(startFile) == 1, "No file found for this `start_date`."
    startFileIndex = fieldFiles.index(startFile[0])
    endFileIndex = startFileIndex + months
    if endFileIndex >= len(fieldFiles) - 1:
        days = len(fieldFiles) - startFileIndex -1
        endFileIndex = len(fieldFiles) - 1
        warnings.warn("\n Timespan of simulation exceeds the amount of data that is available. "                      +"Reducing the amount of `days` to " + str(months) +".")
    selectedFiles = fieldFiles[startFileIndex:endFileIndex]
    variables = {'U' : 'uo',
                 'V' : 'vo'}
    dimensions = {'U': {'time' : 'time',
                        'lat' : 'latitude',
                        'lon' : 'longitude',
                       'depth': 'depth'},
                  'V': {'time' : 'time',
                        'lat' : 'latitude',
                        'lon' : 'longitude',
                       'depth': 'depth'}}
    mesh = fieldFiles[0]
    filenames = {'U' : {'lon' : mesh, 
                        'lat' : mesh, 
                        'data' : selectedFiles},
                 'V' : {'lon' : mesh, 
                        'lat' : mesh, 
                        'data' : selectedFiles}}  
    
    ds = xr.open_dataset(fieldFiles[0])
    
    minLonIdx = np.searchsorted(ds.longitude, lonRange[0]) 
    maxLonIdx = np.searchsorted(ds.longitude, lonRange[1])
    minLatIdx = np.searchsorted(ds.latitude, latRange[0]) 
    maxLatIdx = np.searchsorted(ds.latitude, latRange[1])
    
    indices = {'lon' : range(minLonIdx, maxLonIdx),
               'lat' : range(minLatIdx, maxLatIdx),
               'depth' : [9]}
    fieldset = FieldSet.from_netcdf(selectedFiles, 
                                    variables, 
                                    dimensions, 
                                    indices = indices,
                                    allow_time_extrapolation = True,
                                   )

    
    fieldset.computeTimeChunk(fieldset.U.grid.time[0], 1)
    
    fieldset.landMask = np.isnan(ds.uo[0, 0, minLatIdx:maxLatIdx, minLonIdx:maxLonIdx].data)
    ds.close()
    return fieldset


# In[3]:


#Create fieldset with the function above
fieldset=create(201301,50)


# In[4]:


#Plot the fieldset at a certain time
data = fieldset.V.data

plt.pcolormesh(data[1,:,:], cmap='bwr')


# In[5]:


#PARTICLE SET
pset = ParticleSet.from_list(fieldset=fieldset,   # the fields on which the particles are advected
                             pclass=JITParticle,  # the type of particles (JITParticle or ScipyParticle)
                             lon=df.lon[:2950], # a vector of release longitudes 
                             lat=df.lat[:2950],    # a vector of release latitudes
                              time=pd.to_datetime(df.datetime[:2950]))


# In[6]:


print(pset)


# In[7]:


#Execute the kernel and write to file
output_file = pset.ParticleFile(name="meanflowdisp1.nc", outputdt=3600) # the file name and the time step of the outputs
pset.execute(AdvectionRK4,                 # the kernel (which defines how particles move)
             runtime=3600*24*5000,    # the total length of the run
             dt=600,      # the timestep of the kernel
             output_file=output_file)
output_file.close()


# In[9]:



# In[10]:


#Open output file
ds = xr.open_dataset('meanflowdisp1.nc')

# In[13]:


#plt.pcolormesh(np.linspace(-65,-45,240),np.linspace(55,65,120),fieldset.U.data[1,:,:], vmin=-0.01, vmax=0.01,cmap='bwr')
plt.scatter(ds.lon[1,:],ds.lat[1,:],s=1)
plt.xlim([-53,-48])
plt.ylim([55,60])


# In[15]:


# In[16]:



# In[ ]:




