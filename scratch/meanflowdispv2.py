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
def create(startDate, months, lonRange=(-65, -44), latRange=(54,65), **kwargs):
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
        months = len(fieldFiles) - startFileIndex -1
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


def deleteparticle(particle,fieldset,time):
    """ This function deletes particles as they exit the domain and prints a message about their attributes at that moment
    """
    
    #print('Particle '+str(particle.id)+' has died at t = '+str(time)+' at lon, lat, depth = '+str(particle.lon)+','+str(particle.lat)+', '+str(particle.depth))
    particle.delete()

# In[3]:


#Create fieldset with the function above
fieldset=create(201206,60)


# In[4]:


#Plot the fieldset at a certain time
data = fieldset.V.data

plt.pcolormesh(data[1,:,:], cmap='bwr')


# In[5]:
class BuoyParticle(JITParticle):
    Age=Variable('Age',initial=0)

#PARTICLE SET
pset = ParticleSet.from_list(fieldset=fieldset,   # the fields on which the particles are advected
                             pclass=BuoyParticle,  # the type of particles (JITParticle or ScipyParticle)
                             lon=df.lon[:29000], # a vector of release longitudes 
                             lat=df.lat[:29000],    # a vector of release latitudes
                              time=pd.to_datetime(df.datetime[:29000]))


# In[6]:


#print(pset)


# In[7]:
def aging(particle, fieldset,time):
    particle.Age+=particle.dt
    if particle.Age>3600*24*10:
        particle.delete()


age_kernel = pset.Kernel(aging)
   

#Execute the kernel and write to file
output_file = pset.ParticleFile(name="meanflowdisp1.nc", outputdt=3600) # the file name and the time step of the outputs
pset.execute(AdvectionRK4+age_kernel,                 # the kernel (which defines how particles move)
             runtime=3600*24*200,    # the total length of the run
             dt=600,      # the timestep of the kernel
             recovery = {ErrorCode.ErrorOutOfBounds:deleteparticle}, #delete particle when it gets out of the domain
             output_file=output_file)
output_file.close()







