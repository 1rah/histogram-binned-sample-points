# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:44:20 2018

@author: FluroSat
"""

from scipy import ndimage as ndi
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pi
import cv2


img = pi.open('img_out_2.tif').convert("L")
beach = pi.open('beach.jpeg')


#img_erosion = ndi.binary_erosion(img2, structure=np.ones((5,5))).astype(img2.dtype)
#mask = (img_erosion > 30).astype(np.float)



b = (np.array(img) > 30)

res = beach.resize(img.size)



plt.imshow(res)