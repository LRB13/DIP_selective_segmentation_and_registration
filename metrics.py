# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 11:33:32 2020

@author: burro
"""


import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K


def dice_coef_np(y_true, y_pred):
    smooth=1
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
####################################
def dice_coef(y_true, y_pred):
    smooth=1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def PSNR(T,R):
    return tf.image.psnr(T,R,1)
    
def rel_ssd(T,R,movedT):
    #inputs of shape [batch,n,m,1]
    SSD0 = np.sum(np.square(T - R))
    SSD = np.sum(np.square(movedT - R))
    relSSD = SSD/SSD0
    return relSSD
    
def NCC(movedT,R):
    #inputs of shape [batch,n,m,1]
    TuR = movedT*R
    TuTu = movedT*movedT
    RR = R*R
    t1 = np.sum(TuR)
    t2 = np.sqrt( np.sum( TuTu ) )
    t3 = np.sqrt( np.sum( RR ) )
    return t1/(t2*t3)

def normalised_grad(im):
    #assume input of [n,m]
    x,y = np.gradient(im)
    denom = np.sqrt( np.sum( np.square(x) + np.square(y) + 1e-9) )
    xn = x/denom
    yn = y/denom
    return xn,yn

def DNGF(T,R):
    Tx,Ty = normalised_grad(T)
    Rx,Ry = normalised_grad(R)
    arg = np.square(Tx*Rx + Ty*Ry)
    d = np.sum ( 1 - arg )
    return d
    
def NGF(T,R,movedT):
    dngf0 = DNGF(T[0,:,:,0],R[0,:,:,0])
    dngf1 = DNGF(movedT[0,:,:,0],R[0,:,:,0])
    return dngf1/(dngf0+1e-9)



def SSIM(movedT,R):
    #normalise movedT and R
    movedT = (movedT - np.min(movedT))/(np.max(movedT) - np.min(movedT))
    R = (R - np.min(R))/(np.max(R) - np.min(R))
    
    return tf.image.ssim(tf.convert_to_tensor(movedT,dtype='float32'),tf.convert_to_tensor(R,dtype='float32'),1)

def min_det(disp_tensor):
    u1 = disp_tensor[0,:,:,0]
    u2 = disp_tensor[0,:,:,1]
    u1x,u1y = np.gradient(u1)
    u2x,u2y = np.gradient(u2)
    det = u1x*u2y - u2x*u1y
    qmin = np.min(det)
    return qmin 
    
def all_similarities(T,R,movedT,disp_tensor):
    psnrval = PSNR(movedT,R).numpy()
    rel_ssdval = rel_ssd(T,R,movedT)
    NCCval = NCC(movedT,R)
    ngf_val = NGF(T,R,movedT)
    ssimval = SSIM(movedT,R).numpy()
    minDet = min_det(disp_tensor)
    return psnrval, rel_ssdval, NCCval, ngf_val, ssimval, minDet
    