

import numpy as np
import tensorflow
import tensorflow as tf 

import voxelmorph as vxm

from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, ReLU, LeakyReLU, PReLU
from tensorflow.keras.layers import concatenate, Input, BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam





def down_block(x, filters, kernel_size=(3,3), padding="same", strides=1):
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    c = PReLU()(c) 
    c=BatchNormalization()(c)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides)(c)
    c = PReLU()(c)  
    c=BatchNormalization()(c)
    p = MaxPool2D((2,2))(c) 
    return c,p


def up_block(x, skip, filters, kernel_size=(3,3), padding="same", strides=1):
    us = UpSampling2D((2,2))(x)
    concat = concatenate([us, skip])
    
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides)(concat)
    c = PReLU()(c) 
    c=BatchNormalization()(c)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides)(c)
    c = PReLU()(c) 
    c=BatchNormalization()(c)

    return c

def bottleneck(x, filters, kernel_size=(3,3), padding="same", strides=1):

    c = Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    c = PReLU()(c) 
    c=BatchNormalization()(c)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides)(c)
    c = PReLU()(c) 
    c=BatchNormalization()(c)

    return c


def vxm_UNet1(img_size,input_filters):
    #f = [16,32,64,128,256] #define the feature maps
    #f = [32,64,128,256,512]
    f = [128,128,128,128,128]

    z = Input((img_size,img_size,input_filters)) # noise
    T = Input((img_size,img_size,1)) #I.e. T
    R = Input((img_size,img_size,1)) # I.e. R
    disp_tensor = Input((img_size,img_size,2))

    
    inputs=[z, T, R, disp_tensor]
    
    p0 = z
    c1, p1 = down_block(p0, f[0])
    c2, p2 = down_block(p1, f[1])
    c3, p3 = down_block(p2, f[2])
    c4, p4 = down_block(p3, f[3])
        
    #bottleneck layer
    bn = bottleneck(p4, f[4])
    
    #upsampling path
    u1 = up_block(bn,c4,f[3]) 
    u2 = up_block(u1,c3,f[2]) 
    u3 = up_block(u2,c2,f[1]) 
    u4 = up_block(u3,c1,f[0])
    
    seg_tensor = Conv2D(1, (1,1), padding="same")(u4)

    seg_tensor= Activation('sigmoid')(seg_tensor)
    
    spatial_transformer = vxm.layers.SpatialTransformer(name='image_warping')
    moved_image_tensor = spatial_transformer([inputs[1], disp_tensor])
    moved_seg_tensor = spatial_transformer([seg_tensor, disp_tensor])
   

    outputs = concatenate([disp_tensor, moved_image_tensor, seg_tensor, moved_seg_tensor, R, T])

    
    model = Model(inputs=inputs, outputs = outputs)
    return model

    


def vxm_UNet2_v2(img_size,input_filters):
    #f = [16,32,64,128,256] #define the feature maps
    #f = [32,64,128,256,512]
    f = [128,128,128,128,128]

    z = Input((img_size,img_size,input_filters)) # noise
    T = Input((img_size,img_size,1)) #I.e. T
    R = Input((img_size,img_size,1)) # I.e. R
    seg_tensor = Input((img_size,img_size,1))
    
    inputs=[z, T, R, seg_tensor]
    
    p0 = concatenate([T,R])
    c1, p1 = down_block(p0, f[0]) 
    c2, p2 = down_block(p1, f[1])
    c3, p3 = down_block(p2, f[2]) 
    c4, p4 = down_block(p3, f[3]) 
        
    #bottleneck layer
    bn = bottleneck(p4, f[4])
    
    #upsampling path
    u1 = up_block(bn,c4,f[3]) 
    u2 = up_block(u1,c3,f[2]) 
    u3 = up_block(u2,c2,f[1])
    u4 = up_block(u3,c1,f[0])
    

    
    disp_tensor = Conv2D(2, (1,1), padding="same")(u4)
    

    
    spatial_transformer = vxm.layers.SpatialTransformer(name='image_warping')
    moved_image_tensor = spatial_transformer([T, disp_tensor])
    moved_seg_tensor = spatial_transformer([seg_tensor, disp_tensor])
   
    outputs = concatenate([disp_tensor, moved_image_tensor, seg_tensor, moved_seg_tensor, R, T])
    #outputs = [moved_image_tensor, disp_tensor]

    model = Model(inputs=inputs, outputs = outputs)
    return model
 