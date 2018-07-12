# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 16:46:36 2018

@author: FluroSat
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
import numpy as np
from numpy.random import rand

fig, ax = plt.subplots()
plt.ion()
ax.imshow(bw)
plt.draw()

def onclick(event):
    pString = '{}, {}'.format(event.xdata, event.ydata)
    print(pString)
    if event.dblclick:
        print('DOUBLE')
        ax.imshow(out)
        plt.draw()
        


cid = fig.canvas.mpl_connect('button_press_event', onclick)