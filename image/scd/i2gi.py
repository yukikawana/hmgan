import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import sys, os
import cv2
import skimage.io
sys.path.insert(0,".")
import eval_rgan
import eval_imgsynth

sys.path.insert(0,"scd")
import scd
def main():
    nf=256
    zdim=1024
    filepath="/workspace2/kitti/testing/image_2/007481.png"
    filepath="test/test0_2.jpg"
    filepath="test/test1_2.jpg"
    filepath="test/noise3.jpg"
    filepath="test/adversary.png"
    filepath="/workspace2/kitti/testing/image_2/007489.png"
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    sess=tf.Session(config=config)
    images = tf.placeholder(tf.float32,[1,150,496,3])
    scdobj = scd.SCD(input=images)
    rgan = eval_rgan.rgan(images=scdobj.end_points["pool5"],z_dim=zdim,nf=nf, training=True)
    imgsynth_from_feature = eval_imgsynth.ImageSynthesizer(rgan.rec_x_out)

    scd_saver = scd.get_saver()
    scd_saver.restore(sess,"/workspace/imgsynth/ssd_300_kitti/ssd_model.ckpt")
    imgsynth_saver = eval_imgsynth.get_saver()
    imgsynth_saver.restore(sess,"/workspace/imgsynth/result_kitti256p_2/model.ckpt-89")
    rgan_saver = eval_rgan.get_saver()
    rgan_saver.restore(sess,"models/model-99999")

    image = cv2.resize(scd.imread_as_jpg(filepath),(496,150))
    image=np.expand_dims(image,0)
    reses = imgsynth_from_feature.generate_image_from_featuremap(sess,image,images)
    for i, a in enumerate(reses):
        print(i)
        skimage.io.imsave("test/single_generated.jpg",a)
if __name__ == "__main__":
    main()
