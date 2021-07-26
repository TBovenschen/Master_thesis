# Thesis
 
The code for my thesis on my local computer. Scripts are used to calculate diffusivities from observational data as well as from Parcels model data. Furthermore, there is a script for an idealized model where particles are advected on a velocity field with tides, eddies and shear. 
**
File descriptions:**

binnedstatistic2.py : An adapted version of the scipy binnedstatistics module

calc_angle.py       : A function for calculating angles of consecutive drifter segments

calc_diff.py        : A function to calculate the diffusivities from trajectory data according to the method of Visser(2008)

davis_method.py     : A script to calculate the diffusivities from trajectory data according to the method of Davis(1991)

idealmodel.py       : A model where particles can be advected on an idealized velocity field consisting of tides, eddies and/or shear

plot_bathymetry, plotting_functions.py and quiverplot.py: Scripts used for plotting the results

read_data6hr.py     : A script to load and process drifter data. 

reanalysisdata.py   : A script to load and process monthly mean velocity reanalysis data from CMEMS.

residual_vel_eul.py : A function to calculate the residual velocities from the drifter velocities and the monthly means.
