import os
if not os.path.isdir('Results'):
    os.mkdir('Results')
import sys
import numpy as np
import tensorflow
import tensorflow as tf


physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


from matplotlib import pyplot as plt
import random
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, ReLU, LeakyReLU
from tensorflow.keras.layers import concatenate, Input, BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from scipy.io import loadmat

init = tf.initializers.RandomNormal(mean=0.0, stddev=0.05) #for normal distribution initialisation of wieghts
var = tf.Variable(init(shape=[256,256]))



#LOAD DATA
data = loadmat('ID5_slice80.mat')
T = data['T']
img_size = T.shape[0]
R = data['R']
gt_T = data['maskT']
gt_R = data['maskR']
gd = data['gd']
mask = data['mask']

A1 = np.sum( mask*T )/np.sum(mask)
C1 = A1




#Get geodesic if geodesic not available
if not 'gd' in locals():
    import geodist
    gd = geodist.call_geodist(T,n=3,x=None,y=None)
#PARAMETERS

lambdaP1 = 2 #Fidelity on T
lambdaP2 = .1 #Fidelity on R
xiP = 30 #Distance constraint
DSSDHP =50 #Sum of squared differences (T-R)
#################################################################################
input_filters1 = 32
input_filters2 = 32
lr1 = 0.001
lr2 = 0.001


#LOSSES

def net1loss(y_true,y_pred):
    disp_tensor = y_pred[:,:,:,0:2]
    moved_T = y_pred[:,:,:,2:3]
    seg = y_pred[:,:,:,3:4]
    moved_seg = y_pred[:,:,:,4:5]
    R = y_pred[:,:,:,5:6]
    T = y_pred[:,:,:,6:7]
    moved_R = y_pred[:,:,:,7:8]
    

    CVFit1 = lambdaP1*K.sum(fit_T*seg)
    
    gdConstr = xiP*K.sum( gd*seg )

    CVFit2 = lambdaP2*K.sum(fit_R*moved_seg)
    
    return CVFit1 + gdConstr + CVFit2   
    
def net2loss(y_true,y_pred):
    disp_tensor = y_pred[:,:,:,0:2]
    moved_T = y_pred[:,:,:,2:3]
    seg = y_pred[:,:,:,3:4]
    moved_seg = y_pred[:,:,:,4:5]
    R = y_pred[:,:,:,5:6]
    T = y_pred[:,:,:,6:7]
    
    CVFit2 = lambdaP2*K.sum(fit_R*moved_seg)

    DSSD = DSSDHP*K.sum(K.square(moved_T - R))   
    
    return CVFit2 + DSSD
    
### 
from RS_Fitting import RS_Fitting
fit_T = RS_Fitting(T,A1,lambda3=1)
fit_R = RS_Fitting(R,C1,lambda3=1)

gd = gd[np.newaxis,:,:,np.newaxis]
R = R[np.newaxis,:,:,np.newaxis]
T = T[np.newaxis,:,:,np.newaxis]
fit_T = fit_T[np.newaxis,:,:,np.newaxis]
fit_R = fit_R[np.newaxis,:,:,np.newaxis]


import nets
net1 = nets.vxm_UNet1(img_size,input_filters1)
net2 = nets.vxm_UNet2_v2(img_size,input_filters2)


net1.compile(optimizer = Adam(lr=lr1,decay=0.0001),loss=net1loss,experimental_run_tf_function=False)
net2.compile(optimizer = Adam(lr=lr2,decay=0.0001),loss=net2loss,experimental_run_tf_function=False)

callbacks = [tensorflow.keras.callbacks.EarlyStopping(monitor='loss')]

disp_tensor = np.zeros([T.shape[0], T.shape[1], T.shape[2], 2])

z1 = np.random.uniform(0,1/10,[1,T.shape[1],T.shape[2],input_filters1])
z2 = np.random.uniform(0,1/10,[1,T.shape[1],T.shape[2],input_filters2])


net1_inputs = [z1,T,R,disp_tensor]
net1_outputs = net1.predict(net1_inputs)
seg_tensor = net1_outputs[:,:,:,3:4]

net2_inputs = [z2,T,R,seg_tensor]
net2_outputs = net2.predict(net2_inputs)

epoch_num=1250
batch_size = 1
iters=1

epochVaryNoise = 0


##

import metrics
    

logs = np.array([['diceT','diceR','psnr','epoch']])



for epoch in range(epoch_num):
    if epoch%20 == 0:
        print(epoch)
    

    z1a = np.random.normal(0,1/100,z1.shape)
    z2a = np.random.normal(0,1/8,z2.shape)


    net1_inputs = [z1+z1a,T,R,disp_tensor]
    net1.fit(x=net1_inputs,y=net1_outputs,epochs=iters,batch_size=batch_size,callbacks=callbacks)
    net1_outputs = net1.predict(net1_inputs)
    

    net2_inputs = [z2+z2a,T,R,seg_tensor]
    net2_outputs = net2.predict(net2_inputs)
    net2.fit(x=net2_inputs,y=net2_outputs,epochs=iters, batch_size=batch_size,callbacks=callbacks)
    net2_outputs = net2.predict(net2_inputs)
    
    seg_tensor = net1_outputs[:,:,:,3:4] 
    disp_tensor = net2_outputs[:,:,:,0:2]
    
    
    if epoch%50 == 0:     
        print(epoch)
        
        net1_inputs = [z1,T,R,disp_tensor]
        net1_outputs = net1.predict(net1_inputs)
        seg_tensor = net1_outputs[:,:,:,3:4]
        
        net2_inputs = [z2,T,R,seg_tensor]
        net2_outputs = net2.predict(net2_inputs)
        
        disp_tensor = net2_outputs[:,:,:,0:2]            
        moved_T = net2_outputs[:,:,:,2:3]
        seg = net2_outputs[:,:,:,3:4]
        moved_seg = net2_outputs[:,:,:,4:5]
        R = net2_outputs[:,:,:,5:6]
        
        diceT = metrics.dice_coef_np(seg[0,:,:,0],gt_T)
        diceR = metrics.dice_coef_np(moved_seg[0,:,:,0],gt_R)
        psnr_TR = metrics.PSNR(moved_T,R).numpy()
        logsAppend = np.array([[diceT,diceR,psnr_TR,epoch]])
        logs = np.concatenate((logs,logsAppend))
        
        
        
        BWSeg = seg
        BWSeg[seg>.5]=1
        BWSeg[seg<=.5]=0

        
        BWSegMoved = moved_seg
        BWSegMoved[moved_seg>.5]=1
        BWSegMoved[moved_seg<=.5]=0
        
        
        fig = plt.figure(figsize=(12,12))
        plt.subplot(2,3,1)
        plt.imshow(T[0,:,:,0],cmap='gray')  
        plt.title('T')
        plt.subplot(2,3,2)
        plt.imshow(R[0,:,:,0],cmap='gray')
        plt.title('R')
        plt.subplot(2,3,3)
        plt.imshow(moved_T[0,:,:,0],cmap='gray')
        plt.title('T(x+u), psnr = ' + str(psnr_TR))
        plt.subplot(2,3,4)
        plt.imshow(seg[0,:,:,0])
        plt.title('phi, dice=' + str(diceT))
        plt.subplot(2,3,5)
        plt.imshow(moved_seg[0,:,:,0])
        plt.title('phi moved, dice=' + str(diceR))
        plt.subplot(2,3,6)
        plt.imshow(R[0,:,:,0],cmap='gray')
        plt.contour(BWSegMoved[0,:,:,0],colors='red')
        plt.title('overlaid')
        fig.savefig('Results/out'+str(epoch)+'.png')
        plt.close('all')
        

    
net1_inputs = [z1,T,R,disp_tensor]
net1_outputs = net1.predict(net1_inputs)
seg_tensor = net1_outputs[:,:,:,3:4]

net2_inputs = [z2,T,R,seg_tensor]
net2_outputs = net2.predict(net2_inputs)

disp_tensor = net2_outputs[:,:,:,0:2]            
moved_T = net2_outputs[:,:,:,2:3]
seg = net2_outputs[:,:,:,3:4]
moved_seg = net2_outputs[:,:,:,4:5]
R = net2_outputs[:,:,:,5:6]

