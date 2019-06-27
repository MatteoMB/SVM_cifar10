import numpy as np
import matplotlib.pyplot as plt
from cv2 import Sobel, CV_32F, cartToPolar, COLOR_RGB2GRAY, cvtColor

SZ=32
bin_n=16
patch_sz = (8,8)

def extract_patches(image, patch_size):
    # non-overlapping patches
    patches = []
    for i in range(image.shape[0]//patch_size[0]):
        for j in range(image.shape[1]//patch_size[1]):
            patches.append(image[i*patch_size[0]:(i+1)*patch_size[0],j*patch_size[1]:(j+1)*patch_size[1]])
    return patches
   
def hog(img):
    gx = Sobel(img, CV_32F, 1, 0)
    gy = Sobel(img, CV_32F, 0, 1)

    mag, ang = cartToPolar(gx, gy)
    
    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))
    bins[np.argwhere(bins >= bin_n)] = 15

    # Divide to 4 sub-squares
    bin_cells = extract_patches(bins, patch_sz)
    mag_cells = extract_patches(mag, patch_sz)
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist


def preprocess(img,y = 0,i=-1,samples = 20):
    R = img[0:1024].reshape(SZ,SZ)
    G = img[1024:2048].reshape(SZ,SZ)
    B = img[2048:].reshape(SZ,SZ)
    out = cvtColor(np.dstack((R,G,B)),COLOR_RGB2GRAY)
    if i > -1 and i < samples:
        plt.subplot(2*samples/5,5,i+1)
        plt.title(y)
        plt.imshow(out,cmap="gist_gray")
        plt.axis('off')
    return hog(out).astype(np.double)

def preprocess_all(x_train,y_train = np.empty(0), debug = False):
    if debug:
        plt.figure(figsize=(15,30))
        if len(y_train) == 0:
            print("Wrong input setting")
            return y_train
    imgs = []
    for i,img in enumerate(x_train):
        if debug:
            imgs.append(preprocess(img,y_train[i],i))
        else:
            imgs.append(preprocess(img))
    if debug:
        plt.savefig('images.png')
    return np.vstack(imgs)