from __future__ import division
import math
import re
import os,time,scipy.io
import skimage.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import cv2
import os, sys
import dill
NUM_TRAINING_IMAGES = 7480
#NUM_TRAINING_IMAGES = 30
dimdic = dill.load(open("dim_150.pkl", "rb"))
scaledic = dill.load(open("scaledic.pkl", "rb"))
def add_void_channel(self,semantic):
    return np.concatenate((semantic,np.expand_dims(1-np.sum(semantic,axis=3),axis=3)),axis=3)

def get_saver():
    train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.contrib.framework.get_name_scope())
    ssd_var_list=[v for v in train_vars if v.name.split("/")[0].startswith("g_")]
    saver = tf.train.Saver(var_list=ssd_var_list)
    return saver

class ImageSynthesizer(object):
    def lrelu(self,x):
        return tf.maximum(0.2*x,x)

    def recursive_generator(self,label,sp, reuse=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        dim=512 if sp>=75 else 1024
        if sp==5:
            input=label
        else:
            spss2 = math.ceil(sp/2.)
            downsampled=tf.image.resize_area(label,(spss2,scaledic[spss2]),align_corners=False)
            input=tf.concat([tf.image.resize_bilinear(self.recursive_generator(downsampled,spss2, reuse=reuse),(sp,scaledic[sp]),align_corners=True),label],3)
        net=slim.conv2d(input,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=self.lrelu,scope='g_'+str(sp)+'_conv1')
        net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=self.lrelu,scope='g_'+str(sp)+'_conv2')
        if sp==150:
            net=slim.conv2d(net,3,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100')
            net=(net+1.0)/2.0*255.0


        return net


    def __init__(self,featuremap=None,reuse=None):

        sp=150#spatial resolution: 256x512
        with tf.variable_scope(tf.get_variable_scope()):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            if featuremap==None:
                featuremap=tf.placeholder(tf.float32,[None,10,31,3])
            label2_small = tf.concat((featuremap,tf.expand_dims(1-tf.reduce_sum(featuremap,axis=3),axis=3)),3)
            label2 = tf.image.resize_bilinear(label2_small,(150,496),align_corners=True)
            self.generator=self.recursive_generator(label2,sp)
            self.uint8generator=tf.cast(tf.clip_by_value(self.generator,0,255),tf.uint8)



    def generate_image_from_featuremap(self,sess,semanticmx,tensor=None):
        a = []
        bs=semanticmx.shape[0]
        if tensor==None:
            semantic[semantic<0] = 0.
            output=sess.run(self.uint8generator,feed_dict={self.label2_small:add_void_channel(semantic)})
        else:
            output=sess.run(self.uint8generator,feed_dict={tensor:semanticmx})
        return output
        """
        for idx in range(bs):
            semantic=np.expand_dims(semanticmx[idx,:,:,:],0)
            if tensor==None:
                semantic[semantic<0] = 0.
                output=self.sess.run(self.generator,feed_dict={self.label2_small:add_void_channel(semantic)})
            else:
                output=self.sess.run(self.generator,feed_dict={tensor:semantic})
            output=np.minimum(np.maximum(output,0.0),255.0)
            a.append(np.uint8(output[0,:,:,:],cmin=0,cmax=255))
        return np.array(a)
        """
