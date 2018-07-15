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

            self.gen_params = self.dis_params = self.inv_params = None

            self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
            self.x_p = self.generate(self.z)

            self.x = tf.placeholder(tf.float32, shape=[None, self.h,self.w,self.c])
            #self.x += tf.random_normal(shape=tf.shape(self.x), mean=0.0, stddev=0.01)
            self.z_p = self.invert(self.x)

            self.dis_x = self.discriminate(self.x)
            self.dis_x_p = self.discriminate(self.x_p,reuse=True)
            self.rec_x = self.generate(self.z_p,reuse=True)
            self.rec_z = self.invert(self.x_p,reuse=True)

            self.gen_cost = -tf.reduce_mean(self.dis_x_p)
            #self.gen_cost = tf.reduce_mean(-self.dis_x_p)

            self.inv_cost = tf.reduce_mean(tf.square(self.x - self.rec_x))
            self.inv_cost += self.lamda * tf.reduce_mean(tf.square(self.z - self.rec_z))

            self.dis_cost = tf.reduce_mean(self.dis_x_p) - tf.reduce_mean(self.dis_x)
            #self.dis_cost = -tf.reduce_mean(self.dis_x - self.dis_x_p)

            alpha = tf.random_uniform(shape=[self.batch_size,1,1,1], minval=0., maxval=1.)
            difference = self.x_p - self.x
            print("xp",self.x_p.shape)
            print("x",self.x.shape)
            print("diff",difference.shape)
            #interpolate = self.x + alpha * difference
            interpolate = alpha*self.x + (1.-alpha)*self.x_p
            #gradient = tf.gradients(self.discriminate(interpolate,reuse=True), [interpolate])[0]
            gradient = tf.gradients(self.discriminate(interpolate,reuse=True), [interpolate])[0]
            gradient=slim.flatten(gradient)
            #slope = tf.sqrt(tf.reduce_sum(tf.square(gradient), axis=1))
            slope = tf.norm(gradient, axis=1)
            gradient_penalty = tf.reduce_mean((slope - 1.)**2)
            self.dis_cost += self.c_gp_x * gradient_penalty

            train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.contrib.framework.get_name_scope())
            self.gen_param=[v for v in train_vars if v.name.split("/")[1] == "generate"]
            self.inv_param=[v for v in train_vars if v.name.split("/")[1] == "invert"]
            self.dis_param=[v for v in train_vars if v.name.split("/")[1] == "discriminate"]

            """
            self.gen_train_op = tf.train.AdamOptimizer(
                learning_rate=2e-8, beta1=0.5, beta2=0.9).minimize(
                self.gen_cost, var_list=self.gen_params)
            self.inv_train_op = tf.train.AdamOptimizer(
                learning_rate=2e-8, beta1=0.5, beta2=0.9).minimize(
                self.inv_cost, var_list=self.inv_params)
            self.dis_train_op = tf.train.AdamOptimizer(
                learning_rate=2e-8, beta1=0.5, beta2=0.9).minimize(
                self.dis_cost, var_list=self.dis_params)
            """
            genopt = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0.5, beta2=0.9)
            self.gen_train_op= slim.learning.create_train_op(self.gen_cost,genopt,summarize_gradients=True,variables_to_train=self.gen_params)
            invopt = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0.5, beta2=0.9)
            self.inv_train_op= slim.learning.create_train_op(self.inv_cost,invopt,summarize_gradients=True,variables_to_train=self.inv_params)
            disopt = tf.train.AdamOptimizer(
                learning_rate=1e-4, beta1=0.5, beta2=0.9)
                #learning_rate=1.52*1e-5, beta1=0.5, beta2=0.9)
            self.dis_train_op= slim.learning.create_train_op(self.dis_cost,disopt,summarize_gradients=True,variables_to_train=self.dis_params)



    def _residual_block(self, X, nf_output, resample, kernel_size=[3,3], name='res_block'):
        with tf.variable_scope(name):
            input_shape = X.shape
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
                upsample_shape = tuple(int(x)*2 for x in input_shape[1:3])
                shortcut = tf.image.resize_nearest_neighbor(X, upsample_shape) 
                shortcut = slim.conv2d(shortcut, nf_output, kernel_size=[1,1], activation_fn=None)

                net = slim.batch_norm(X, activation_fn=act_fn_switch, **self.bn_params)
                net = tf.image.resize_nearest_neighbor(net, upsample_shape) 
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
            startdimw = 2*2**(4-self.restime)#for h= 512,w=31,c=10
            startdimh = 32*2**(4-self.restime)#for h= 512,w=31,c=10
            #startdimw = 2*2**(4-self.restime)
            #startdimh = 2*2**(4-self.restime)
            net = slim.fully_connected(z, startdimh*startdimw*start_coef*nf, activation_fn=None) 
            net = tf.reshape(net, [-1, startdimh, startdimw, start_coef*nf])
            for i,p in enumerate(range(self.restime-1,-1,-1)):
                net = self._residual_block(net, (2**p)*nf, resample='up', name='res_block%d'%i)
            net = slim.batch_norm(net, activation_fn=act_fn_switch, **self.bn_params)
            net = slim.conv2d(net, self.c, kernel_size=[3,3], activation_fn=tf.nn.tanh)
            net = net[:,:,:-1,:]#for h= 512,w=31,c=10
            #net = net[:,11:21,:-1,:]

            return net

    def discriminate(self, X, reuse=False):
        with tf.variable_scope('discriminate', reuse=reuse):
            nf = self.nf
            net = slim.conv2d(X, nf, [3,3], activation_fn=None) 
            for i,p in enumerate(range(1,self.restime)):
                net = self._residual_block(net, (2**p)*nf, resample='down', name='res_block%d'%i)
            net = self._residual_block(net, (2**(self.restime-1))*nf, resample='down', name='res_block%d'%self.restime)
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
 


    def train_gen(self, sess, x, z, summary=False):
        if summary:
            _gen_cost, _, summary = sess.run([self.gen_cost, self.gen_train_op, self.merge],
                                    feed_dict={self.x: x, self.z: z})
            return _gen_cost, summary
        else:
            _gen_cost, _ = sess.run([self.gen_cost, self.gen_train_op],
                                    feed_dict={self.x: x, self.z: z})
            return _gen_cost 

    def train_dis(self, sess, x, z, summary=False):
        if summary:
            _dis_cost, _, summary = sess.run([self.dis_cost, self.dis_train_op, self.merge],
                                    feed_dict={self.x: x, self.z: z})
            return _dis_cost, summary
        else:
            _dis_cost, _= sess.run([self.dis_cost, self.dis_train_op],
                                    feed_dict={self.x: x, self.z: z})
            return _dis_cost

    def train_inv(self, sess, x, z, summary=False):
        if summary:
            _inv_cost, _, summary = sess.run([self.inv_cost, self.inv_train_op, self.merge],
                                    feed_dict={self.x: x, self.z: z})
            return _inv_cost, summary
        else:
            _inv_cost, _= sess.run([self.inv_cost, self.inv_train_op],
                                    feed_dict={self.x: x, self.z: z})
            return _inv_cost

    def generate_from_noise(self, sess, noise, frame):
        samples = sess.run(self.x_p, feed_dict={self.z: noise})
        samplesnpz = samples.reshape((-1, self.h,self.w,self.c)).transpose([0,3,2,1])*NORMALIZE_CONST
        #samplesnpz = samples.reshape((-1, self.h,self.w,self.c))*NORMALIZE_CONST
        np.savez(os.path.join(self.output_path, 'examples/samples_{}.npz'.format(frame)),samplesnpz)
        tflib.save_images_hm.save_images(
            samples.reshape((-1, self.h,self.w,self.c)).transpose([0,3,2,1])*NORMALIZE_CONST,
            #samples.reshape((-1, self.h,self.w,self.c))*NORMALIZE_CONST,
            os.path.join(self.output_path, 'examples/samples_{}.png'.format(frame)))
        return samples

    def reconstruct_images(self, sess, images, frame):
        reconstructions = sess.run(self.rec_x, feed_dict={self.x: images})
        samplesnpz = reconstructions.reshape((-1, self.h,self.w,self.c)).transpose([0,3,2,1])*NORMALIZE_CONST
        #samplesnpz = reconstructions.reshape((-1, self.h,self.w,self.c))*NORMALIZE_CONST
        np.savez(os.path.join(self.output_path, 'examples/recs_{}.npz'.format(frame)),samplesnpz)
        comparison = np.zeros((images.shape[0] * 2, images.shape[1],images.shape[2],images.shape[3]),
                              dtype=np.float32)
        for i in range(images.shape[0]):
            comparison[2 * i] = images[i]
            comparison[2 * i + 1] = reconstructions[i]
        tflib.save_images_hm.save_images(
            comparison.reshape((-1, self.h,self.w,self.c)).transpose([0,3,2,1])*NORMALIZE_CONST,
            #comparison.reshape((-1, self.h,self.w,self.c))*NORMALIZE_CONST,
            os.path.join(self.output_path, 'examples/recs_{}.png'.format(frame)))
        return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epoch', type=int, default=1000, help='epoch')
    parser.add_argument('--z_dim', type=int, default=1024, help='dimension of z')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='latent dimension')
    parser.add_argument('--iterations', type=int, default=100000,
                        help='training steps')
    parser.add_argument('--dis_iter', type=int, default=5,
                        help='discriminator steps')
    parser.add_argument('--c_gp_x', type=float, default=10.,
                        help='coefficient for gradient penalty x')
    parser.add_argument('--lamda', type=float, default=.1,
                        help='coefficient for divergence of z')
    parser.add_argument('--output_path', type=str, default='./',
                        help='output path')
    parser.add_argument('--logdir', type=str, default='./log',
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
    args = parser.parse_args()


    fixed_images = np.zeros([args.batch_size,args.input_h,args.input_w,args.input_c])
    for i in range(args.batch_size):
        fixed_images[i,:,:,:]=np.load(os.path.join(args.dataset_path,"%06d.npz"%(i+7481)))['arr_0'].transpose(0,3,2,1)[0,:,:,:]/NORMALIZE_CONST
        #fixed_images[i,:,:,:]=np.load(os.path.join(args.dataset_path,"%06d.npz"%(i+7481)))['arr_0'][0,:,:,:]/NORMALIZE_CONST

    tf.set_random_seed(326)
    np.random.seed(326)
    fixed_noise = np.random.randn(args.batch_size, args.z_dim)

    mnistWganInv = MnistWganInv(
        x_dim=784, z_dim=args.z_dim, w=args.input_w, h=args.input_h, c=args.input_c, latent_dim=args.latent_dim,
        nf=args.nf, batch_size=args.batch_size, c_gp_x=args.c_gp_x, lamda=args.lamda,
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
        batchmx = np.zeros([args.batch_size,args.input_h,args.input_w,args.input_c])
        print("pre epoch")
        with tf.name_scope("summary"):
            tf.summary.scalar("gen_cost",mnistWganInv.gen_cost)
            tf.summary.scalar("inv_cost",mnistWganInv.inv_cost)
            tf.summary.scalar("dis_cost",mnistWganInv.dis_cost)
            mnistWganInv.merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter(args.logdir,session.graph)

        summary=False
        for epoch in range(args.epoch):
            try:
                minibatch = tl.iterate.minibatches(inputs=data_files, targets=data_files, batch_size=args.batch_size, shuffle=True)
                while True:
                    pretime=time.time()
                    if iteration % 100 == 99:
                        summary=True
                    else:
                        summary=False
                    iteration+=1
                    for i in range(args.dis_iter):
                        noise = np.random.randn(args.batch_size, args.z_dim)
                        batch_files,_ = minibatch.__next__()
                        for idx, filepath in enumerate(batch_files):
                            if not filepath in npzs:
                                npzs[filepath] = np.load(filepath)['arr_0']/NORMALIZE_CONST
                            batchmx[idx,:,:,:]=npzs[filepath][0,:,:,:].transpose([2,1,0])
                            #batchmx[idx,:,:,:]=npzs[filepath][0,:,:,:]
                        images = batchmx

                        """
                        if summary:
                            gen_cost, mg = mnistWganInv.train_gen(session, images, noise, summary=True)
                        else:
                            gen_cost = mnistWganInv.train_gen(session, images, noise, summary=False)
                        """

                        """
                    if summary:
                        cd, md = mnistWganInv.train_dis(session, images, noise, summary=True)
                    else:
                        cd = mnistWganInv.train_dis(session, images, noise, summary=False)
                    dis_cost_lst += [cd]

                    if summary:
                        ci, mi = mnistWganInv.train_inv(session, images, noise, summary=True)
                    else:
                        ci = mnistWganInv.train_inv(session, images, noise, summary=False)
                    inv_cost_lst += [ci]
                        """


                        if summary:
                            cd, md = mnistWganInv.train_dis(session, images, noise, summary=True)
                        else:
                            cd = mnistWganInv.train_dis(session, images, noise, summary=False)
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

                    dis_cost = np.mean(dis_cost_lst)
                    inv_cost = np.mean(inv_cost_lst)
                    stime=time.time()-pretime        
                    print("epoch: %d, iteration: %d, gen_cost: %f, dis_cost: %f, inv_cost: %f, time: %.2f"%(epoch, iteration, gen_cost, dis_cost, inv_cost,stime))

                    tflib.plot.plot('train gen cost', gen_cost)
                    tflib.plot.plot('train dis cost', dis_cost)
                    tflib.plot.plot('train inv cost', inv_cost)

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
                        dev_dis_cost_lst, dev_inv_cost_lst = [], []
                        dev_images = fixed_images
                        noise = np.random.randn(args.batch_size, args.z_dim)
                        dev_dis_cost, dev_inv_cost = session.run(
                            [mnistWganInv.dis_cost, mnistWganInv.inv_cost],
                            feed_dict={mnistWganInv.x: dev_images,
                                       mnistWganInv.z: noise})
                        dev_dis_cost_lst += [dev_dis_cost]
                        dev_inv_cost_lst += [dev_inv_cost]
                        tflib.plot.plot('dev dis cost', np.mean(dev_dis_cost_lst))
                        tflib.plot.plot('dev inv cost', np.mean(dev_inv_cost_lst))

                    if iteration < 5 or iteration % 100 == 99:
                        tflib.plot.flush(os.path.join(args.output_path, 'models'))
            except StopIteration:
                pass
                

                tflib.plot.tick()


