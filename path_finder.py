# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:29:30 2018

@author: FluroSat
"""

import PIL.Image as im

img_list = list()
for i in range(1,4):
    img_list.append(im.open('img_out_{}.tif'.format(i)))