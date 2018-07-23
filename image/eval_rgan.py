import os, sys, time
import tensorlayer as tl
from glob import glob
import time
sys.path.append(os.getcwd())


import numpy as np
import tensorflow as tf
import argparse

import tflib
import tflib.plot
import tflib.save_images_hm
import tflib.ops.batchnorm
import tflib.ops.conv2d
import tflib.ops.deconv2d
import tflib.ops.linear
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from tensorflow.python.keras.layers import Activation, BatchNormalization, add, Reshape
from tensorflow.python.keras.layers import DepthwiseConv2D
slim = tf.contrib.slim
from tensorflow.python.keras import backend as K
NORMALIZE_CONST=201.
TRANSPOSE=False
act_fn_switch=tf.nn.leaky_relu

class rgan(object):
    def __init__(self, z_dim=128, nf=128, 
                 training=False,images=None, latent_vec=None,reuse=None):
        with tf.variable_scope('wgan'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            self.bn_params = {
                "decay": 0.99,
                "epsilon": 1e-5,
                "scale": True,
                "is_training": training
            }
            self.mean_vec=np.load("mean.npz")["arr_0"]
            self.z_dim = z_dim
            self.w = 31
            self.h = 10
            self.c =512
            self.nf = nf
            self.restime = 3

         

            if latent_vec==None:
                self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
            else:
                self.z=latent_vec
            self.x_p = self.generate(self.z)

            if images==None:
                self.pre_x = tf.placeholder(tf.float32, shape=[None, self.h,self.w,self.c])
            else:
                self.pre_x=images
            self.x=self.pre_x-self.mean_vec.reshape([1,1,1,self.mean_vec.shape[0]])
            #self.x += tf.random_normal(shape=tf.shape(self.x), mean=0.0, stddev=0.01)
            self.z_p = self.invert(self.x)

            self.dis_x = self.discriminate(self.x)
            self.dis_x_p = self.discriminate(self.x_p,reuse=True)
            self.rec_x = self.generate(self.z_p,reuse=True)
            self.rec_x_out = self.generate(self.z_p,reuse=True)+self.mean_vec.reshape([1,1,1,self.mean_vec.shape[0]])
            self.rec_z = self.invert(self.x_p,reuse=True)
            self.rec_x_p_out = self.generate(self.z,reuse=True)+self.mean_vec.reshape([1,1,1,self.mean_vec.shape[0]])



    def _residual_block(self, X, nf_output, resample, kernel_size=[3,3],upsample_shape=None, name='res_block'):
        with tf.variable_scope(name):
            input_shape = X.shape
            nf_input = input_shape[-1]
            """
            if input_shape[1]<kernel_size[0] or input_shape[2] < kernel_size[1]:
                X= tf.image.resize_image_with_crop_or_pad(X, kernel_size[0], kernel_size[1])
            """
            nf_input = input_shape[-1]
            if resample == 'down': 
                shortcut = slim.avg_pool2d(X, [2,2])
                shortcut = slim.conv2d(shortcut, nf_output, kernel_size=[1,1], activation_fn=None) 

                net = slim.layer_norm(X, activation_fn=act_fn_switch)
                net = slim.conv2d(net, nf_input, kernel_size=kernel_size, biases_initializer=None) 
                net = slim.layer_norm(net, activation_fn=act_fn_switch)
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size)
                net = slim.avg_pool2d(net, [2,2])

                return net + shortcut
            elif resample == 'up': 
                if upsample_shape==None:
                    upsample_shape = tuple(int(x)*2 for x in input_shape[1:3])
                #shortcut = tf.image.resize_nearest_neighbor(X, upsample_shape, alig) 
                shortcut = tf.image.resize_bilinear(X, upsample_shape) 
                shortcut = slim.conv2d(shortcut, nf_output, kernel_size=[1,1], activation_fn=None)

                net = slim.batch_norm(X, activation_fn=act_fn_switch, **self.bn_params)
                #net = tf.image.resize_nearest_neighbor(net, upsample_shape) 
                net = tf.image.resize_bilinear(net, upsample_shape) 
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size, biases_initializer=None) 
                net = slim.batch_norm(net, activation_fn=act_fn_switch, **self.bn_params)
                net = slim.conv2d(net, nf_output, kernel_size=kernel_size)

                return net + shortcut
            else:
                raise Exception('invalid resample value')

    def generate(self, z, reuse=False):
        with tf.variable_scope('generate', reuse=reuse):
            start_coef = 2**(self.restime-1)
            nf = self.nf
            if TRANSPOSE:
                startdimw = 2*2**(4-self.restime)#for h= 512,w=31,c=10
                startdimh = 32*2**(4-self.restime)#for h= 512,w=31,c=10
            else:
                startdimw = 4*2**(3-self.restime)
                startdimh = 4*2**(3-self.restime)
                upsample_shapes=[
                (4,6),
                (7,16),
                (10,31)
                ]
            net = slim.fully_connected(z, startdimh*startdimw*start_coef*nf, activation_fn=None) 
            net = tf.reshape(net, [-1, startdimh, startdimw, start_coef*nf])
            for i,p in enumerate(range(self.restime-1,-1,-1)):
                net = self._residual_block(net, (2**p)*nf,upsample_shape=(None if TRANSPOSE else upsample_shapes[i]), resample='up', name='res_block%d'%i)
            net = slim.batch_norm(net, activation_fn=act_fn_switch, **self.bn_params)
            #net = slim.conv2d(net, self.c, kernel_size=[3,3], activation_fn=tf.nn.tanh)
            net = slim.conv2d(net, self.c, kernel_size=[3,3], activation_fn=None)
            if TRANSPOSE:
                net = net[:,:,:-1,:]
            """
            else:
                net = net[:,11:21,:-1,:]
            """

            return net

    def discriminate(self, X, reuse=False):
        with tf.variable_scope('discriminate', reuse=reuse):
            nf = self.nf
            net = slim.conv2d(X, nf, [3,3], activation_fn=None) 
            for i,p in enumerate(range(1,self.restime)):
                net = self._residual_block(net, (2**p)*nf, resample='down', name='res_block%d'%i)
            net = self._residual_block(net, (2**self.restime)*nf, resample='down', name='res_block%d'%self.restime)
            net=act_fn_switch(net)
            net=tf.reduce_mean(net,axis=[1,2])
            net = slim.fully_connected(net, 1, activation_fn=None)
            return net
 
    def invert(self, x,reuse=False):
        with tf.variable_scope('invert', reuse=reuse):
            nf = self.nf
            net = slim.conv2d(x, nf, [3,3], activation_fn=None) 
            for i,p in enumerate(range(1,self.restime)):
                net = self._residual_block(net, (2**p)*nf, resample='down', name='res_block%d'%i)
            net=act_fn_switch(net)
            net=tf.reduce_mean(net,axis=[1,2])
            net = slim.fully_connected(net, nf*(2**(self.restime-1)), activation_fn=None)
            net = slim.fully_connected(net, self.z_dim, activation_fn=None)

            return net

    def generate_featuremap_from_noise(self,sess,noise):
        samples = sess.run(self.x_p, feed_dict={self.z: noise})
        return samples

    def generate_noise_from_featuremap(self,sess,featuremap,images=None):
        if images==None:
            samples = sess.run(self.z_p, feed_dict={self.pre_x: featuremap})
        else:
            real_image = featuremap
            samples = sess.run(self.z_p, feed_dict={images: real_image})
        return samples
 
    def reconstruct_featuremap(self,sess,featuremap):
        reconstructions = sess.run(self.rec_x_out, feed_dict={self.pre_x: featuremap})
        return reconstructions

def get_saver():
    train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.contrib.framework.get_name_scope())
    ssd_var_list=[v for v in train_vars if v.name.split("/")[0].startswith("wgan")]
    saver = tf.train.Saver(var_list=ssd_var_list)
    return saver
