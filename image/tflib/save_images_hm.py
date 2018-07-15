"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
import scipy.misc
from scipy.misc import imsave
import cv2
import dill
div=4
def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')


    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = int(rows),int(n_samples/rows)

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        #X = X.transpose(0,3,2,1)
        h, w = X[0].shape[:2]
        nX = np.zeros([X.shape[0],h*div//2,w*div//2,1],dtype=np.uint8)
        nX2 = np.zeros([X.shape[0],h*div//2,w*div//2,3],dtype=np.uint8)
        for n in range(div):
            if n==0:
                i=0
                j=0
            elif n==1:
                i=0
                j=1
            elif n==2:
                i=1
                j=0
            elif n==3:
                i=1
                j=1
            i=int(i)
            j=int(j)
            #nX[:,i*(h):(i+1)*(h),i*(w):(i+1)*(w),0]=np.uint8((X[:,:,:,i]/(X[:,:,:,i].max()+1e-12)*255).clip(0,255))
            nX[:,j*h:j*h+h,i*w:i*w+w,0]=np.uint8((X[:,:,:,n*2]/(X[:,:,:,n*2].max()+1e-12)*255).clip(0,255))
            """
            if i == 3:
                nX[:,i*(h):(i+1)*(h),i*(w):(i+1)*(w),0]=0
            """
        for i in range(X.shape[0]):
            nX2[i,:,:,:]=cv2.applyColorMap(nX[i,:,:,0],cv2.COLORMAP_JET)
        X=nX2
        img = np.zeros((int(h*nh*div/2), int(w*nw*div/2), 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        nh = int(nh)
        nw = int(nw)
        img = np.zeros((h*nh, w*nw))

    h*=int(div/2)
    w*=int(div/2)
    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        i=int(i)
        j=int(j)
        img[j*h+1:j*h+h-1, i*w+1:i*w+w-1,:] = x[1:-1,1:-1,:]
    

    imsave(save_path, img)

def main():
    batch_size=16
    mx = np.zeros([batch_size,10,31,512])
    #mx=mx.transpose(0,3,2,1)
    for i in range(batch_size):
        loaded= np.load(open("../../imgsynth/hmpool5/%06d.npz"%i,"rb"))['arr_0'][0,:,:,:]
        mx[i,:,:,:] =loaded 
    img=cv2.applyColorMap(np.uint8((mx[0,:,:,0]/(mx[:,:,:,0].max()+1e-12)*255).clip(0,255)),cv2.COLORMAP_JET)
    imsave("test2.png",img)
    comparison = np.zeros((mx.shape[0] * 2, mx.shape[1],mx.shape[2],mx.shape[3]),
                          dtype=np.float32)
    for i in range(mx.shape[0]):
        comparison[2 * i] = mx[i]
        comparison[2 * i + 1] = mx[i]-10
    save_images(mx, "test.png")
    save_images(comparison, "comp.png")
    


if  __name__ == '__main__':
    main()
