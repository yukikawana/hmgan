import os, sys, time
import tensorlayer as tl
from glob import glob
import time
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

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
def relu6(x):
    return tf.keras.backend.relu(x, maxval=6)
from tensorflow.python.keras import backend as K
NORMALIZE_CONST=201.
TRANSPOSE=True
mean_vec=np.load("mean.npz")["arr_0"]
act_fn_switch=tf.nn.leaky_relu

class MnistWganInv(object):
    def __init__(self, x_dim=784, w=31, h=10, c=512, z_dim=64, latent_dim=64,nf=64,  batch_size=80,
                 c_gp_x=10., lamda=0.1, output_path='./',training=True):
        with tf.variable_scope('wgan'):
            self.bn_params = {
                "decay": 0.99,
                "epsilon": 1e-5,
                "scale": True,
                "is_training": training
            }
            self.x_dim = x_dim
            self.z_dim = z_dim
            self.w = w
            self.h = h
            self.c =c
            self.nf = nf
            self.restime = 3
            self.latent_dim = latent_dim
            self.batch_size = batch_size
            self.c_gp_x = c_gp_x
            self.lamda = lamda
            self.output_path = output_path


         
            self.gamma_plh = tf.placeholder(tf.float32, shape=(), name='gammaplh')

            self.z_rand = tf.placeholder(tf.float32, shape=[self.batch_size, self.z_dim])
            self.x = tf.placeholder(tf.float32, shape=[self.batch_size, self.h,self.w,self.c])

            self.z_hat = self.invert(self.x)
            self.x_hat = self.generate(self.z_hat)
            self.x_rand = self.generate(self.z_rand,reuse=True)

            self.l1_loss = self.lamda*tf.reduce_mean(tf.losses.absolute_difference(self.x_hat, self.x))

            self.d_z_hat = self.discriminate_code(self.z_hat)
            self.c_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_z_hat, labels=tf.ones_like(self.d_z_hat)))

            self.d_x_hat = self.discriminate(self.x_hat)
            self.d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_x_hat, labels=tf.ones_like(self.d_x_hat)))

            self.d_x_rand = self.discriminate(self.x_rand,reuse=True)
            self.d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_x_rand, labels=tf.ones_like(self.d_x_rand)))

            self.loss1 = self.l1_loss + self.c_loss + self.d_real_loss + self.d_fake_loss

            self.d_x = self.discriminate(self.x,reuse=True)
            self.x_loss2 = 2.0 * \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_x, labels=tf.ones_like(self.d_x))) + \
            tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_x_hat, labels=tf.zeros_like(self.d_x_hat)))
            self.z_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_x_rand, labels=tf.zeros_like(self.d_x_rand)))

            self.loss2 = self.x_loss2 + self.z_loss2
            disc_reg = self.Discriminator_Regularizer(self.d_x, self.x, self.d_x_rand, self.x_rand, self.d_x_hat, self.x_hat)
            assert self.loss2.shape == disc_reg.shape
            self.loss2 += (self.gamma_plh/2.0)*disc_reg

            self.x_loss3 = self.c_loss
            
            self.d_z_rand = self.discriminate_code(self.z_rand,reuse=True)
            self.z_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_z_rand, labels=tf.zeros_like(self.d_z_rand)))
            self.loss3 = self.x_loss3 + self.z_loss3

            train_vars=tf.trainable_variables()
            self.gen_param=[v for v in train_vars if v.name.split("/")[1] == "generate"]
            self.inv_param=[v for v in train_vars if v.name.split("/")[1] == "invert"]
            self.dis_param=[v for v in train_vars if v.name.split("/")[1] == "discriminate"]
            self.dis_code_param=[v for v in train_vars if v.name.split("/")[1] == "discriminate_code"]


            egopt = tf.train.AdamOptimizer(
                learning_rate=1e-3, beta1=0.5, beta2=0.9)
            eg_train_param =self.gen_param+self.inv_param
            self.eg_train_op= slim.learning.create_train_op(self.loss1,egopt,summarize_gradients=True,variables_to_train=eg_train_param)


            dopt = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0.5, beta2=0.9)
            self.d_train_op= slim.learning.create_train_op(self.loss2,dopt,summarize_gradients=True,variables_to_train=self.dis_param)

            dcopt = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0.5, beta2=0.9)
            self.dc_train_op= slim.learning.create_train_op(self.loss3,dcopt,summarize_gradients=True,variables_to_train=self.dis_code_param)

    def Discriminator_Regularizer(self,D1_logits, D1_arg, D2_logits, D2_arg, D3_logits, D3_arg):
        with tf.variable_scope('disc_reg'):
            D1 = tf.nn.sigmoid(D1_logits)
            D2 = tf.nn.sigmoid(D2_logits)
            D3 = tf.nn.sigmoid(D3_logits)
            grad_D1_logits = tf.gradients(D1_logits, D1_arg)[0]
            grad_D2_logits = tf.gradients(D2_logits, D2_arg)[0]
            grad_D3_logits = tf.gradients(D3_logits, D3_arg)[0]
            grad_D1_logits_norm = tf.norm( tf.reshape(grad_D1_logits, [self.batch_size,-1]) , axis=1)
            grad_D2_logits_norm = tf.norm( tf.reshape(grad_D2_logits, [self.batch_size,-1]) , axis=1)
            grad_D3_logits_norm = tf.norm( tf.reshape(grad_D3_logits, [self.batch_size,-1]) , axis=1)
            
            #set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
            grad_D1_logits_norm=tf.expand_dims(grad_D1_logits_norm,1)
            grad_D2_logits_norm=tf.expand_dims(grad_D2_logits_norm,1)
            grad_D3_logits_norm=tf.expand_dims(grad_D3_logits_norm,1)
            print('grad_D1_logits_norm.shape {} != D1.shape {}'.format(grad_D1_logits_norm.shape, D1.shape))
            print('grad_D2_logits_norm.shape {} != D2.shape {}'.format(grad_D2_logits_norm.shape, D2.shape))
            print('grad_D3_logits_norm.shape {} != D3.shape {}'.format(grad_D3_logits_norm.shape, D3.shape))
            assert grad_D1_logits_norm.shape == D1.shape
            assert grad_D2_logits_norm.shape == D2.shape
            assert grad_D3_logits_norm.shape == D3.shape
            
            reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))
            reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))
            reg_D3 = tf.multiply(tf.square(D3), tf.square(grad_D3_logits_norm))
            
            disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2 + reg_D3)
            #disc_regularizer = tf.reduce_mean(reg_D2)
            
            return disc_regularizer
  


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
            net = self._residual_block(net, (2**self.restime)*nf, resample='down', name='res_block%d'%self.restime)
            net=act_fn_switch(net)
            net=tf.reduce_mean(net,axis=[1,2])
            net = slim.fully_connected(net, self.z_dim, activation_fn=None)

            return net
 
    def discriminate_code(self, x,reuse=False):
        with tf.variable_scope('discriminate_code', reuse=reuse):
            net = slim.fully_connected(x, self.z_dim, activation_fn=act_fn_switch)
            net = slim.fully_connected(net, self.z_dim, activation_fn=act_fn_switch)
            net = slim.fully_connected(net, 1, activation_fn=None)
            return net


    def train_eg(self, sess, x, z, summary=False):
        if summary:
            _gen_cost, _, summary = sess.run([self.loss1, self.eg_train_op, self.merge],
                                    feed_dict={self.x: x, self.z_rand: z})
            return _gen_cost, summary
        else:
            _gen_cost, _ = sess.run([self.loss1, self.eg_train_op],
                                    feed_dict={self.x: x, self.z_rand: z})
            return _gen_cost 

    def train_d(self, sess, x, z, gamma, summary=False):
        if summary:
            print(gamma)
            _dis_cost, _, summary = sess.run([self.loss2, self.d_train_op, self.merge],
                  feed_dict={self.x: x, self.z_rand: z, self.gamma_plh: gamma})
            return _dis_cost, summary
        else:
            _dis_cost, _= sess.run([self.loss2, self.d_train_op],
                                    feed_dict={self.x: x, self.z_rand: z, self.gamma_plh: gamma})
            return _dis_cost

    def train_dc(self, sess, x, z, summary=False):
        if summary:
            _inv_cost, _, summary = sess.run([self.loss3, self.dc_train_op, self.merge],
                                    feed_dict={self.x: x, self.z_rand: z})
            return _inv_cost, summary
        else:
            _inv_cost, _= sess.run([self.loss3, self.dc_train_op],
                                    feed_dict={self.x: x, self.z_rand: z})
            return _inv_cost

    def generate_from_noise(self, sess, noise, frame):
        samples = sess.run(self.x_rand, feed_dict={self.z_rand: noise})
        if TRANSPOSE:
            #samplesnpz = (samples.reshape((-1, self.h,self.w,self.c)).transpose([0,3,2,1]) + 1)*NORMALIZE_CONST/2.
            samplesnpz = samples.reshape((-1, self.h,self.w,self.c)).transpose([0,3,2,1])
        else:
            #samplesnpz = (samples.reshape((-1, self.h,self.w,self.c)) + 1)*NORMALIZE_CONST/2.
            samplesnpz = samples.reshape((-1, self.h,self.w,self.c))
        samplesnpz+=mean_vec.reshape([1,1,1,mean_vec.shape[0]])

        np.savez(os.path.join(self.output_path, 'examples/samples_{}.npz'.format(frame)),samplesnpz)

        tflib.save_images_hm.save_images(
            samplesnpz,
            os.path.join(self.output_path, 'examples/samples_{}.png'.format(frame)))
        return samples

    def reconstruct_images(self, sess, images, frame):
        reconstructions = sess.run(self.x_hat, feed_dict={self.x: images})
        if TRANSPOSE:
            samplesnpz = reconstructions.reshape((-1, self.h,self.w,self.c)).transpose([0,3,2,1])
            comparison = np.zeros((images.shape[0] * 2, images.shape[3],images.shape[2],images.shape[1]),
                              dtype=np.float32)
        else:
            samplesnpz = reconstructions.reshape((-1, self.h,self.w,self.c))
            comparison = np.zeros((images.shape[0] * 2, images.shape[1],images.shape[2],images.shape[3]),
                              dtype=np.float32)
        samplesnpz+=mean_vec.reshape([1,1,1,mean_vec.shape[0]])

        np.savez(os.path.join(self.output_path, 'examples/recs_{}.npz'.format(frame)),samplesnpz)
        for i in range(images.shape[0]):
            if TRANSPOSE:
                comparison[2 * i] = images[i].transpose([2,1,0])+mean_vec.reshape([1,1,mean_vec.shape[0]])
            else:
                comparison[2 * i] = images[i]+mean_vec.reshape([1,1,mean_vec.shape[0]])
            comparison[2 * i + 1] = samplesnpz[i]


        tflib.save_images_hm.save_images(
            comparison,
            #comparison.reshape((-1, self.h,self.w,self.c))*NORMALIZE_CONST,
            os.path.join(self.output_path, 'examples/recs_{}.png'.format(frame)))
        return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epoch', type=int, default=1000, help='epoch')
    parser.add_argument('--z_dim', type=int, default=128, help='dimension of z')
    parser.add_argument('--latent_dim', type=int, default=1024,
                        help='latent dimension')
    parser.add_argument('--iterations', type=int, default=100000,
                        help='training steps')
    parser.add_argument('--dis_iter', type=int, default=2,
                        help='discriminator steps')
    parser.add_argument('--c_gp_x', type=float, default=10.,
                        help='coefficient for gradient penalty x')
    parser.add_argument('--lamda', type=float, default=4.,
                        help='coefficient for divergence of z')
    parser.add_argument('--output_path', type=str, default='./',
                        help='output path')
    parser.add_argument('--logdir', type=str, default='./logr',
                        help='log dir')
    parser.add_argument('--dataset_path', type=str, default='/workspace/imgsynth/hmpool5',
                        help='dataset path')
    parser.add_argument('--nf', type=int, default=32,
                        help='ooooooooooo')
    parser.add_argument('--input_c', type=int, default=10,
    #parser.add_argument('--input_c', type=int, default=512,
                        help='ooooooooooo')
    parser.add_argument('--input_w', type=int, default=31,
    #parser.add_argument('--input_w', type=int, default=31,
                        help='ooooooooooo')
    parser.add_argument('--input_h', type=int, default=512,
    #parser.add_argument('--input_h', type=int, default=10,
                        help='ooooooooooo')
    parser.add_argument("--gamma",type=float,default=0.1,help="noise variance for regularizer [0.1]")
    parser.add_argument("--annealing", type=bool,default=False, help="annealing gamma_0 to decay_factor*gamma_0 [False]")
    parser.add_argument("--decay_factor",type=float, default=0.01, help="exponential annealing decay rate [0.01]")
    args = parser.parse_args()

    if TRANSPOSE:
        fixed_images = np.zeros([args.batch_size,args.input_h,args.input_w,args.input_c])
    else:
        fixed_images = np.zeros([args.batch_size,args.input_c,args.input_w,args.input_h])
    for i in range(args.batch_size):
        if TRANSPOSE:
            fixed_images[i,:,:,:]=(np.load(os.path.join(args.dataset_path,"%06d.npz"%(i+7481)))['arr_0']-mean_vec.reshape([1,1,1,mean_vec.shape[0]])).transpose(0,3,2,1)[0,:,:,:]
        else:
            #fixed_images[i,:,:,:]=np.load(os.path.join(args.dataset_path,"%06d.npz"%(i+7481)))['arr_0'][0,:,:,:]/NORMALIZE_CONST
            fixed_images[i,:,:,:]=np.load(os.path.join(args.dataset_path,"%06d.npz"%(i+7481)))['arr_0'][0,:,:,:]-mean_vec.reshape([1,1,mean_vec.shape[0]])
    fixed_noise = np.random.randn(args.batch_size, args.z_dim)
    
    if TRANSPOSE:
        mnistWganInv = MnistWganInv(
        x_dim=784, z_dim=args.z_dim, w=args.input_w, h=args.input_h, c=args.input_c, latent_dim=args.latent_dim,
        nf=args.nf, batch_size=args.batch_size, c_gp_x=args.c_gp_x, lamda=args.lamda,
        output_path=args.output_path)
    else:
        mnistWganInv = MnistWganInv(
        x_dim=784, z_dim=args.z_dim, w=args.input_w, h=args.input_c, c=args.input_h, latent_dim=args.latent_dim,
        nf=256, batch_size=args.batch_size, c_gp_x=args.c_gp_x, lamda=args.lamda,
        output_path=args.output_path)

    saver = tf.train.Saver(max_to_keep=10)


    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    with tf.Session(config=config) as session:
        session.run(tf.global_variables_initializer())

        images = noise = gen_cost = dis_cost = inv_cost = None
        dis_cost_lst, inv_cost_lst = [], []
        data_files = glob(os.path.join(args.dataset_path, "*.npz"))
        data_files = sorted(data_files)
        data_files = np.array(data_files) 
        iteration=0
        npzs = {}
        if TRANSPOSE:
            batchmx = np.zeros([args.batch_size,args.input_h,args.input_w,args.input_c])
        else:
            batchmx = np.zeros([args.batch_size,args.input_c,args.input_w,args.input_h])
        print("pre epoch")
        with tf.name_scope("summary"):
            tf.summary.scalar("eg_cost",mnistWganInv.loss1)
            tf.summary.scalar("dv_cost",mnistWganInv.loss2)
            tf.summary.scalar("dc_cost",mnistWganInv.loss3)
            mnistWganInv.merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter(args.logdir,session.graph)

        summary=False
        for epoch in range(args.epoch):
            try:
                minibatch = tl.iterate.minibatches(inputs=data_files, targets=data_files, batch_size=args.batch_size, shuffle=True)
                while True:
                    pretime=time.time()
                    if iteration % 100 == 99:
                        summary=False
                    else:
                        summary=False
                    iteration+=1
                    if args.annealing:
                        gamma = args.gamma*args.decay_factor**(iteration/(args.iterations-1))
                    else:
                        gamma = args.gamma

                    for i in range(args.dis_iter):
                        noise = np.random.randn(args.batch_size, args.z_dim)
                        batch_files,_ = minibatch.__next__()
                        for idx, filepath in enumerate(batch_files):
                            if not filepath in npzs:
                                #npzs[filepath] = (np.load(filepath)['arr_0']/NORMALIZE_CONST-0.5)*2.
                                npzs[filepath] = np.load(filepath)['arr_0']-mean_vec.reshape([1,1,1,mean_vec.shape[0]])
                            if TRANSPOSE:
                                batchmx[idx,:,:,:]=npzs[filepath][0,:,:,:].transpose([2,1,0])
                            else:
                                batchmx[idx,:,:,:]=npzs[filepath][0,:,:,:]
                                
                            #batchmx[idx,:,:,:]=npzs[filepath][0,:,:,:]
                        images = batchmx

                        if summary:
                            gen_cost, mg = mnistWganInv.train_eg(session, images, noise, summary=True)
                        else:
                            gen_cost = mnistWganInv.train_eg(session, images, noise, summary=False)

                    if summary:
                        cd, md = mnistWganInv.train_d(session, images, noise, gamma, summary=True)
                    else:
                        cd = mnistWganInv.train_d(session, images, noise,gamma, summary=False)
                    dis_cost_lst += [cd]

                    if summary:
                        ci, mi = mnistWganInv.train_dc(session, images, noise, summary=True)
                    else:
                        ci = mnistWganInv.train_dc(session, images, noise, summary=False)
                    inv_cost_lst += [ci]

                    """

                        if summary:
                            cd, md = mnistWganInv.train_dis(session, images, noise, gamma, summary=True)
                        else:
                            cd = mnistWganInv.train_dis(session, images, noise, gamma, summary=False)
                        dis_cost_lst += [cd]

                        if summary:
                            ci, mi = mnistWganInv.train_inv(session, images, noise, summary=True)
                        else:
                            ci = mnistWganInv.train_inv(session, images, noise, summary=False)
                        inv_cost_lst += [ci]

                    if summary:
                        gen_cost, mg = mnistWganInv.train_gen(session, images, noise, summary=True)
                    else:
                        gen_cost = mnistWganInv.train_gen(session, images, noise, summary=False)
                    """

                    dis_cost = np.mean(dis_cost_lst)
                    inv_cost = np.mean(inv_cost_lst)
                    stime=time.time()-pretime        
                    print("epoch: %d, iteration: %d, gen_cost: %f, dis_cost: %f, inv_cost: %f, time: %.2f"%(epoch, iteration, gen_cost, dis_cost, inv_cost,stime))

                    tflib.plot.plot('train eg cost', gen_cost)
                    tflib.plot.plot('train d cost', dis_cost)
                    tflib.plot.plot('train dc cost', inv_cost)

                    if summary:
                        writer.add_summary(mg, iteration)
                        writer.add_summary(md, iteration)
                        writer.add_summary(mi, iteration)

                    #if iteration % 1000 == 999:
                    if iteration % 100 == 99:
                        mnistWganInv.generate_from_noise(session, fixed_noise, iteration)
                        mnistWganInv.reconstruct_images(session, fixed_images, iteration)

                    if iteration % 1000 == 999:
                        save_path = saver.save(session, os.path.join(
                            args.output_path, 'models/model'), global_step=iteration)

                    if iteration % 1000 == 999:
                        losses1, losses2, losses3 = [], [],[]
                        dev_images = fixed_images
                        noise = np.random.randn(args.batch_size, args.z_dim)
                        loss1, loss2, loss3= session.run(
                            [mnistWganInv.loss1, mnistWganInv.loss2, mnistWganInv.loss3],
                            feed_dict={mnistWganInv.x: dev_images,
                                       mnistWganInv.z_rand: noise,
                                       mnistWganInv.gamma_plh:gamma})
                        losses1+= [loss1]
                        losses2+= [loss2]
                        losses3+= [loss3]
                        tflib.plot.plot('dev eg cost', np.mean(losses1))
                        tflib.plot.plot('dev d cost', np.mean(losses2))
                        tflib.plot.plot('dev dcv cost', np.mean(losses3))

                    if iteration < 5 or iteration % 100 == 99:
                        tflib.plot.flush(os.path.join(args.output_path, 'models'))
            except StopIteration:
                pass
                

                tflib.plot.tick()


