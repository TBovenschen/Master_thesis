#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 17:24:39 2020

@author: tychobovenschen
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def plot_angles(dangle,angle,dist):
    mean_angle = np.mean(np.concatenate(dangle))
    skew_angle = stats.skew(np.concatenate(dangle))
    countzero=np.zeros(len(angle))
    for i in range(len(angle)):
        countzero[i] = np.count_nonzero(dist[i]<100)
        # dangle[i] = dangle[i][~np.isnan(dangle[i])]
    plt.figure()
    plt.hist(np.concatenate(dangle)[~np.isnan(np.concatenate(dangle))],bins=200,range=[-180,180], stacked=True)
    plt.title('Difference in angles between consecutive data points',fontsize=16)
    plt.ylabel('Number of datapoints')
    plt.xlabel('Angles (degrees)')
    plt.text(-150,1500, 'Skewness = '+str(skew_angle)[:-14] +'\n'+ 'Mean angle = '+str(mean_angle)[:-14])
    # plt.xlim([-160,-150])
    plt.grid()
    plt.show()
