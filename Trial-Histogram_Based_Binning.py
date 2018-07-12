# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:49:10 2018

@author: FluroSat

"""

import numpy as np
import matplotlib.pyplot as plt

indice_img = np.loadtxt('indice_data.csv', delimiter=',')
plt.imshow(indice_img)

class_bounds = np.loadtxt('class_bounds(3).csv', delimiter=',')