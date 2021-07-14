# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 17:51:48 2020

@author: burro
"""

import numpy as np
import cv2

def make_mask(img,z)
    pts = np.round(np.array(z,dtype='int32'))

    mask = np.zeros((img.shape[0], img.shape[1]))

    cv2.fillConvexPoly(mask, pts, 1)
    mask = mask.astype(np.bool)