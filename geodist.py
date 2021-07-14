# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 11:30:49 2019

@author: burro
"""

import numpy as np
#import cv2
import matplotlib.pyplot as plt
import tkinter #conda install tk
import matplotlib
from skimage.restoration import denoise_nl_means, estimate_sigma


def get_input(im,n):
    #matplotlib.use('TkAgg')
    matplotlib.use('Qt4Agg')
    plt.imshow(im,cmap='gray')
    z = plt.ginput(n)
    #plt.show()
    plt.close()
    x=np.zeros(len(z))
    y=np.zeros(len(z))
    
    for i in range(0,len(z)):
        cord = z[i]
        x[i]=np.round(cord[0])
        y[i]=np.round(cord[1])
            
    return x,y

def call_geodist(im,n=3,x=None,y=None):
    #call this function. If x,y need user input, do not define x or y in call.
    # if x and y are already known, input here.
    # n is the number of marker points
    if x:
        x = np.int(np.round(x))
        y = np.int(np.round(y))
    else:
        x,y = get_input(im,n)
        
    sigma_est = np.mean(estimate_sigma(im))
    ims = denoise_nl_means(im,h=0.6*sigma_est,sigma=5*sigma_est) #vary sigma if geodesic does not look good

    
    return calculate_geodist(ims,x,y)
        


def calculate_geodist(im,x,y):
    
    ##### get mask
    #traditional x axis on bottom
    mask = np.zeros(np.shape(im))
    #breakpoint()
    if isinstance(x,int):
        mask[y,x] = 1
    else:
        for i in range(0,len(x)):
            xi = np.int(x[i])
            yi = np.int(y[i])
            mask[yi,xi] = 1
    
    ##############
    #ims = anisodenoise(im)
    ims = im
    gx,gy = np.gradient(ims)
    grad = np.sqrt(np.square(gx)+np.square(gy))
    
    #eucDist = timesweep(np.ones(im.shape),mask)
    
    beta = 1000
    epsi = 1e-3
    theta = 0
    f = epsi + beta*grad #+ theta*eucDist
    D = timesweep(f,mask)
    
    return D
    

def anisodenoise(im):
    im = im-np.min(im)
    im=im/np.max(im)
    gx,gy = np.gradient(im)
    grad = np.sqrt(np.square(gx)+np.square(gy))
    g = 1/(1+1000*grad)
    n,m=np.shape(im)
    A = np.zeros(np.shape(im))
    B = A
    C = A
    D = A
    
    mu = 1e-3 #smoothing term
    tau = 5e-4 #fitting term
    iters = 100
    
    hx = 1/(n-1); hy = 1/(m-1)
    cx = mu/(np.square(hx)); cy = mu/(np.square(hy))

    A[:,:] = g[:,:]
    A[0:n-2,:] = (1/2)*(g[1:n-1,:] + A[0:n-2,:])

    B[:,:] = g[:,:]
    B[1:n-1,:] = (1/2)*(g[0:n-2,:] + B[1:n-1,:])

    C[:,:] = g[:,:]
    C[:,0:m-2] = (1/2)*(g[:,1:m-1] + C[:,0:m-2])
    
    D[:,:] = g[:,:]
    D[:,1:m-1] = (1/2)*(g[:,0:m-2] + D[:,1:m-1])
    
    A = cx*A; B = cx*B
    C = cy*C; D = cy*D
    
    u = im
    num = np.zeros([n,m])
    newu=u

    denom = A + B + C + D
    denom = denom + tau


    for k in range(0,iters):
        oldu = u
        for i in range(0,n):
            for j in range(0,m):
                if i==0 and j==0:
                    num[i,j] = A[i,j]*u[i+1,j] + B[i,j]*u[i+1,j] + C[i,j]*u[i,j+1] + D[i,j]*u[i,j+1]
                    newu[i,j] = num[i,j]/denom[i,j]
                elif i==n-1 and j==0:
                    num[i,j] = A[i,j]*u[i-1,j] + B[i,j]*u[i-1,j] + C[i,j]*u[i,j+1] + D[i,j]*u[i,j+1]
                    newu[i,j] = num[i,j]/denom[i,j]
                elif i==0 and j==m-1:
                    num[i,j] = A[i,j]*u[i+1,j] + B[i,j]*u[i+1,j] + C[i,j]*u[i,j-1] + D[i,j]*u[i,j-1]
                    newu[i,j] = num[i,j]/denom[i,j]
                elif i==n-1 and j==m-1:
                    num[i,j] = A[i,j]*u[i-1,j] + B[i,j]*u[i-1,j] + C[i,j]*u[i,j-1] + D[i,j]*u[i,j-1]
                    newu[i,j] = num[i,j]/denom[i,j]
                elif i==0:
                    num[i,j] = A[i,j]*u[i+1,j] + B[i,j]*u[i+1,j] + C[i,j]*u[i,j+1] + D[i,j]*u[i,j-1]
                    newu[i,j] = num[i,j]/denom[i,j]
                elif i==n-1:
                    num[i,j] = A[i,j]*u[i-1,j] + B[i,j]*u[i-1,j] + C[i,j]*u[i,j+1] + D[i,j]*u[i,j-1]
                    newu[i,j] = num[i,j]/denom[i,j]
                elif j==0:
                    num[i,j] = A[i,j]*u[i+1,j] + B[i,j]*u[i-1,j] + C[i,j]*u[i,j+1] + D[i,j]*u[i,j+1]
                    newu[i,j] = num[i,j]/denom[i,j]
                elif j==m-1:
                    num[i,j] = A[i,j]*u[i+1,j] + B[i,j]*u[i-1,j] + C[i,j]*u[i,j-1] + D[i,j]*u[i,j-1]
                    newu[i,j] = num[i,j]/denom[i,j]
                else:
                    num[i,j] = A[i,j]*u[i+1,j] + B[i,j]*u[i-1,j] + C[i,j]*u[i,j+1] + D[i,j]*u[i,j-1]
                    newu[i,j] = num[i,j]/denom[i,j]
        u=newu
    
    return u






    

def timesweep(f,mask):
    n,m = np.shape(f)
    inf = 1e6
    T = 1000
    u = inf*(1-mask)
    
    res0=[] #add stuff to this with res0.append(R)
    stop = .0005
    R = 10*stop
    
    count = 0
    xx=np.linspace(0,n-1,n)
    yy=np.linspace(0,m-1,m)
    orderx = [xx,np.flip(xx),np.flip(xx),xx]
    ordery = [yy,yy,np.flip(yy),np.flip(yy)]
    
    h=0.005
    
    for iter in range(0,20):
        count+=1
        order = np.mod(count-1,4)+1 #1,2,3,4,1,2,3,4,1,2,3,4,1,....
        x=orderx[order-1]
        y=ordery[order-1]
        
        oldu = u
        
        for i in x:
            for j in y:
                i=int(i)
                j=int(j)
                if i==0:
                    a = u[i+1,j]
                elif i==n-1:
                    a = u[i-1,j]
                else:
                    a = np.minimum(u[i-1,j],u[i+1,j])
                
                if j==0:
                    b = u[i,j+1]
                elif j==m-1:
                    b = u[i,j-1]
                else:
                    b = np.minimum(u[i,j-1],u[i,j+1])
                
                fh = f[i,j]*h
                if np.minimum(a,b) < T:
                    cond = np.abs(a-b)
                    if cond >= fh:
                        uBar = np.minimum(a,b) + fh
                    else:
                        uBar = (1/2)*(a+b+np.sqrt(2*np.square(fh) - np.square(a-b)))
                    u[i,j] = np.minimum(u[i,j],uBar)
        
        R = np.linalg.norm(oldu-u)
        res0.append(R)
        if len(res0) == 500:
            print("Did not converge after {} iterations".format(count))
                    
    
    D = (u-np.min(u))/(np.max(u)-np.min(u))
    print("Geodesic distance converged at iter = {}".format(count))
    return D
                    
                        
