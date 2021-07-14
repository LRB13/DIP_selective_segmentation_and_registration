# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 20:23:11 2020

@author: burro
"""

from skimage.filters import threshold_multiotsu
#image = data.camera()
import numpy as np


def RS_Fitting_f2(img,c1):
    """ defines f2 in RS_Fitting
    """
    thresholds = threshold_multiotsu(img,classes=3)
    if c1 <= thresholds[0]:
        # ii)
        gamma1 = c1
        gamma2 = thresholds[0]-c1
    elif c1 <= thresholds[1]:
        #i)
        gamma1 = c1 - thresholds[0]
        gamma2 = thresholds[1]-c1
    else:
        gamma1 = c1 - thresholds[1]
        gamma2 = 1-c1
    
    n,m = img.shape
    f2 = np.zeros([n,m])
    
    for i in range(0,n):
        for j in range(0,m):
            if c1-gamma1 <= img[i,j] and img[i,j] <= c1:
                f2[i,j] = 1 + (img[i,j]-c1)/gamma1
            elif c1 < img[i,j] and img[i,j] <= c1 + gamma2:
                f2[i,j] = 1 - (img[i,j]-c1)/gamma2
        
    
    return f2

def RS_Fitting(img,c1,lambda3=1):
    f1 = (img-c1)**2
    f2 = RS_Fitting_f2(img,c1)
    fit = f1 - lambda3*f2
    return fit
                
    
        