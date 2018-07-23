import os, sys
import pickle
import eval_rgan
import eval_imgsynth
import numpy as np
import tensorflow as tf
import cv2
import argparse
sys.path.insert(0,"scd")
import scd

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

import tflib.mnist
from mnist_wgan_inv import MnistWganInv
from search import iterative_search, recursive_search
import skimage.io


def save_adversary(adversary, filename):
    skimage.io.imsave("test/adversary.png",np.uint8(adversary['x_adv'].clip(0,255)))
    #skimage.io.imsave("test/adversary.png",adversary['x_adv'])
    """
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))

    ax[0].imshow(np.reshape(adversary['x'], (28, 28)),
                 interpolation='none', cmap=plt.get_cmap('gray'))
    ax[0].text(1, 5, str(adversary['y']), color='white', fontsize=50)
    ax[0].axis('off')

    ax[1].imshow(np.reshape(adversary['x_adv'], (28, 28)),
                 interpolation='none', cmap=plt.get_cmap('gray'))
    ax[1].text(1, 5, str(adversary['y_adv']), color='white', fontsize=50)
    ax[1].axis('off')

    fig.savefig(filename)
    plt.close()
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gan_path', type=str, default='./models/model-47999',
                        help='mnist GAN path')
    parser.add_argument('--rf_path', type=str, default='./models/mnist_rf_9045.sav',
                        help='RF classifier path')
    parser.add_argument('--lenet_path', type=str, default='./models/mnist_lenet_9871.h5',
                        help='LeNet classifier path')
    parser.add_argument('--classifier', type=str, default='rf',
                        help='classifier: rf OR lenet')
    parser.add_argument('--iterative', action='store_true',
                        help='iterative search OR recursive')
    parser.add_argument('--nsamples', type=int, default=5,
                        help='number of samples in each search iteration')
    parser.add_argument('--step', type=float, default=0.01,
                        help='Delta r for search step size')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--output_path', type=str, default='./examples/',
                        help='output path')
    args = parser.parse_args()
    #np.random.seed(10)#worked
    #np.random.seed(3)#worked
    #np.random.seed(123)#worked
    #np.random.seed(50)#forked for adv4
    np.random.seed(100)#forked for adv5,6
    np.random.seed(777)

    l=0
    cls=1
    y_t=0
    crd = 2373
    path="/workspace2/kitti/testing/image_2/007481.png"
    th=0.6
    upl=10

    l=1
    cls=1
    y_t=0
    crd = 950
    path="test/test0_2.jpg"
    th=0.6
    upl=10

    l=0
    cls=1
    y_t=0
    crd = 2617
    path="test/noise3.jpg"
    th=0.90
    upl=30
    #upl=20 #for adv1,2,3
    ra=2

    l=0
    cls=1
    y_t=0
    crd = 2609
    path="/workspace2/kitti/testing/image_2/007489.png"
    th=0.90
    upl=20
    ra=2


    args.iterative=False
    args.verbose=False
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess=tf.Session(config=config)

    zb=args.nsamples
    nf=256
    zdim=1024
    z_tensor = tf.placeholder(tf.float32,[None,zdim])
    images = tf.placeholder(tf.float32,[None,150,496,3])
    scdobj= scd.SCD(input=images)

    rgan= eval_rgan.rgan(latent_vec=z_tensor,images=scdobj.end_points["pool5"],nf=nf,z_dim=zdim,training=True)
    imgsynth = eval_imgsynth.ImageSynthesizer(rgan.rec_x_p_out)


    scd_saver = scd.get_saver()
    scd_saver.restore(sess,"/workspace/imgsynth/ssd_300_kitti/ssd_model.ckpt")
    imgsynth_saver = eval_imgsynth.get_saver()
    imgsynth_saver.restore(sess,"/workspace/imgsynth/result_kitti256p_2/model.ckpt-89")
    rgan_saver = eval_rgan.get_saver()
    rgan_saver.restore(sess,"models/model-99999")


    def cla_fn(x):
        pre =sess.run(scdobj.tensors[0][l],feed_dict={images:x})
        ps=pre.shape
        b,h,w,a = np.unravel_index(crd,ps[:-1])
        hmin=min(max(h-ra,0),ps[1]-1)
        wmin=min(max(w-ra,0),ps[2]-1)
        hmax=min(max(h+ra,0),ps[1]-1)
        wmax=min(max(w+ra,0),ps[2]-1)
        preans=pre[:,hmin:hmax,wmin:wmax,a,cls]
        pre=np.max(preans,axis=(1,2))

        """
        pre=pre.reshape(pre.shape[0],-1,8)[:,crd,cls]
        """
        ans=np.zeros_like(pre)
        #ans[pre>th] = 1
        ans[pre>th] = 1
        print("pre", pre,ans)
        return ans




    def gen_fn(z):
        x_p = imgsynth.generate_image_from_featuremap(sess,z,z_tensor)
        return x_p


    def inv_fn(x):
        z_p = rgan.generate_noise_from_featuremap(sess,x,images)
        return z_p


    if args.iterative:
        search = iterative_search
    else:
        search = recursive_search

    _, _, test_data = tflib.mnist.load_data()

    i=0
    co=[]
    image = cv2.resize(scd.imread_as_jpg(path),(496,150))
    co.append(image)
    image = np.array(co)
    x=image
    y=1
    adversary = search(gen_fn, inv_fn, cla_fn, x, y,y_t=y_t,
                h=upl, nsamples=args.nsamples, step=args.step, verbose=args.verbose)
    if args.iterative:
        filename = 'mnist_{}_iterative_{}.png'.format(str(i).zfill(4), args.classifier)
    else:
        filename = 'mnist_{}_recursive_{}.png'.format(str(i).zfill(4), args.classifier)

    save_adversary(adversary, os.path.join(args.output_path, filename))

