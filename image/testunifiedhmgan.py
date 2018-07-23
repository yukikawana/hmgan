import os, sys
import eval_rgan
import eval_imgsynth
sys.path.insert(0,"scd")
import scd
import tensorflow as tf
import numpy as np
import cv2
import skimage.io


def main():
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess=tf.Session(config=config)

    #define tensors and  models
    zb=6
    nf=256
    zdim=1024
    images = tf.placeholder(tf.float32,[2,150,496,3])
    z = tf.placeholder(tf.float32,[zb,zdim])


    scdobj = scd.SCD(input=images)
    imgsynth = eval_imgsynth.ImageSynthesizer(scdobj.end_points["pool5"])

    rgan = eval_rgan.rgan(images=scdobj.end_points["pool5"],z_dim=zdim,nf=nf, training=True)
    imgsynth_from_feature = eval_imgsynth.ImageSynthesizer(rgan.rec_x_out,reuse=True)

    rgan_from_z = eval_rgan.rgan(latent_vec=z,nf=nf,z_dim=zdim,reuse=True, training=True)
    imgsynth_from_z = eval_imgsynth.ImageSynthesizer(rgan_from_z.rec_x_p_out,reuse=True)


    #load weights
    scd_saver = scd.get_saver()
    scd_saver.restore(sess,"/workspace/imgsynth/ssd_300_kitti/ssd_model.ckpt")
    imgsynth_saver = eval_imgsynth.get_saver()
    imgsynth_saver.restore(sess,"/workspace/imgsynth/result_kitti256p_2/model.ckpt-89")
    rgan_saver = eval_rgan.get_saver()
    rgan_saver.restore(sess,"models/model-99999")
    
    co=[]
    image = cv2.resize(scd.imread_as_jpg("/workspace2/kitti/testing/image_2/007482.png"),(496,150))
    co.append(image)
    image = cv2.resize(scd.imread_as_jpg("/workspace2/kitti/testing/image_2/007481.png"),(496,150))
    co.append(image)
    #reses = imgsynth.generate_image_from_featuremap(np.expand_dims(image,0),images)
    image = np.array(co)

    #simply get detection result
    reses = scdobj.get_image(sess,image)
    for i, a in enumerate(reses):
        print(i)
        skimage.io.imsave("test/detected%d.jpg"%i,a)
    #model:image=>feature, image decoder:feature=>image_hat
    reses = imgsynth.generate_image_from_featuremap(sess,image,images)
    for i, a in enumerate(reses):
        print(i)
        skimage.io.imsave("test/test%d.jpg"%i,a)


    #model:image=>feature, gan:feature=>latent vector=>feature_hat, image decoder:feature_hat=>image_hat
    reses = imgsynth_from_feature.generate_image_from_featuremap(sess,image,images)
    for i, a in enumerate(reses):
        print(i)
        skimage.io.imsave("test/test%d_2.jpg"%i,a)

    #get detection result thru gan and image decoder
    reses = scdobj.get_image(sess,reses)
    for i, a in enumerate(reses):
        print(i)
        skimage.io.imsave("test/gan_detected%d.jpg"%i,a)

    return 0
    #latent vector=>feature_hat, image decoder:feature_hat=>image_hat
    np.random.seed(0)
    zs = []
    zsp = np.random.randn(zb,zdim)+1
    zs = np.array(zsp)
    reses = imgsynth_from_z.generate_image_from_featuremap(sess,zs,z)
    for i, a in enumerate(reses):
        print(i)
        skimage.io.imsave("test/noise%d.jpg"%i,a)

    #noise interpolation
    zsb = []
    for i in range(0,zb):
        #zs.append(zsp[0]*float(i)/(float(zb)-1.)+zsp[1]*(1.-float(i)/(float(zb)-1.)))
        zsb.append(zsp[i]*(1.-i/(zb-1)))
    zsb = np.array(zsb)
    reses = imgsynth_from_z.generate_image_from_featuremap(sess,zsb,z)
    for i, a in enumerate(reses):
        print(i)
        skimage.io.imsave("test/noise_interpolate%d.jpg"%i,a)

    #model:images=>features, gan:features=>latent vectors=>features_hat, interpolate:features_hat=>features_interplated, image decoder:feature_interplated=>image_hat
    zs = []
    zsp = rgan.generate_noise_from_featuremap(sess,image,images)
    for i in range(0,zb):
        zs.append(zsp[0]*float(i)/zb+zsp[1]*(1-float(i)/zb))
    zs = np.array(zs)
    reses = imgsynth_from_z.generate_image_from_featuremap(sess,zs,z)
    for i, a in enumerate(reses):
        print(i)
        skimage.io.imsave("test/interpolate%d.jpg"%i,a)
    
    zs = []
    zsp = rgan.generate_noise_from_featuremap(sess,image,images)
    for i in range(0,zb):
        zs.append(zsp[0]*float(i)/zb+zsp[1]*(1-float(i)/zb)+3)
    zs = np.array(zs)
    reses = imgsynth_from_z.generate_image_from_featuremap(sess,zs,z)
    for i, a in enumerate(reses):
        print(i)
        skimage.io.imsave("test/interpolate2%d.jpg"%i,a)

if __name__ == "__main__":
    main()


