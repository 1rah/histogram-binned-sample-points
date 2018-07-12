# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 15:33:48 2018

@author: FluroSat
"""
#REF: https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/


# Python program to demonstrate erosion and 
# dilation of images.
import cv2
import numpy as np
 
# Reading the input image
img = cv2.imread('img_out_2.png', 0)
 
# Taking a matrix of size 5 as the kernel
kernel = np.ones((5,5), np.uint8)
 
# The first parameter is the original image,
# kernel is the matrix with which image is 
# convolved and third parameter is the number 
# of iterations, which will determine how much 
# you want to erode/dilate a given image. 
img_erosion = cv2.erode(img, kernel, iterations=1)

#img_dilation = cv2.dilate(img, kernel, iterations=1)

plt.figure()
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(img_erosion)
#plt.subplot(133)
#plt.imshow(img_dilation)
plt.show()
 

img