# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 15:26:09 2018

@author: FluroSat
"""

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage import generate_binary_structure

file_list = glob(r'D:\histogram-binned-sample-points\erode\*.npy')
for f in file_list:
    plt.figure()
    img = np.load(f)
    
    plt.subplot(131)
    plt.imshow(img)
    
    selem = np.array([
                    [0,0,1,0,0],
                    [0,1,1,1,0],
                    [1,1,1,1,1],
                    [0,1,1,1,0],
                    [0,0,1,0,0],
                     ])
#    selem=None
    img_bin1 = img.astype(np.bool)
    img_bin2 = binary_erosion(img_bin1, selem=selem)
    img_bin3 = binary_dilation(img_bin2, selem=selem)
    img_new = np.where(img_bin3,img,0)
    
    plt.subplot(132)
    plt.imshow(img_new)
    plt.subplot(133)
    check = img_bin3 & ~img_bin1
    plt.imshow(check)
    print(len(check[check==True]))
    