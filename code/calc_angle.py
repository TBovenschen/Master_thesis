#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:10:37 2020

@author: tychobovenschen
"""
import numpy as np

def calc_angle(data_split):
    """Function to calculate the angles of different data points in buoy trajectories:
    INPUT: 
    data_split:   A list of numpy arrays where every array is 1 buoy trajectory
                    data_split columns: (ID, date, time, lat , lon, ve, vn, speed, varlat, varlon, vart, datetime, timedelta)
    OUTPUT: 
    angle:          A list of arrays with the angles that a buoy makes with the x-direction 
                            (length of every array is length the arrays in data_split -1)
    dangle:         A list of arrays with the difference in between consectuive angles of buoys
                            length of every array is length the arrays in data_split -2)
    dist:           A list of arrays with distance between data points"""

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
            angle[i][j] = np.arctan2(dy[i][j],dx[i][j])/(2*np.pi)*360
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
    return angle, dangle, dist