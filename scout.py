

'''
    Rough script on scouting code


'''

import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np 
import classification as classi
import sys



def main():

    indice_path = 'AvonmoreFogdon_index_ccci.tif'
    ds = gdal.Open(indice_path)
    indice_data = ds.GetRasterBand(1).ReadAsArray() #nans for all -1000s
    indice_data[indice_data == -10000] = np.nan


    # define paths for using Tom's code
    classesNpy = indice_path.split('.')[0] + '_.npy'
    smoothed_results_path = indice_path.split('.')[0] + '_smoothed.tif'
    jsonName = indice_path.split('.')[0] + '_points.json'

    # define method, smoothing, classes
    m = 'equalInterval'
    s = 10
    c = 5

    # smooth
    classi.filter(indice_path, smoothed_results_path, smoothing=[s], res=None)
    # classify
    classi.classify(smoothed_results_path, classesNpy, m, [c], class_breaks=None) 

    collated_list = np.load(classesNpy, allow_pickle=True)
    class_data = indice_data
    class_avg = collated_list[1]
    class_bounds = collated_list[2]

    for bound in class_bounds:
        # get all data between upper and lower bounds
        indexs = np.where(np.logical_and(class_data >= bound[0], class_data <=  bound[1]))
        values = class_data[indexs]

        # get hist vals
        #hist, bins = np.histogram(values, bins=10)

        # get data between biggest bin range

        # print (hist)
        # print (bins)

        plt.title(bound)
        plt.ylabel('Freq')
        plt.xlabel('Pix val')
        plt.hist(values, bins=200)
        plt.show()

    sys.exit()




if __name__ == '__main__':
    main()

'''

def show_output(name):
    cv2.imshow("Output", name)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




file = 'AvonmoreFogdon_index_ccci.tif'
ds = gdal.Open(file)
filtered_data = ds.GetRasterBand(1).ReadAsArray()

plt.imshow(filtered_data, cmap='Greys')
plt.show()
sys.exit()



filtered_data[2004, 1621] = 0
img_resize = cv2.resize(filtered_data, (500,500))

show_output(img_resize)

# separate data by bounds
for bound in class_bounds:
    indexs = np.where(np.logical_and(class_data >= bound[0], class_data <=  bound[1]))
    values = class_data[indexs]

    # plot hist
    plt.title(bound)
    plt.ylabel('Freq')
    plt.xlabel('Pix val')
    plt.hist(values, bins=10)

    # get hist vals
    hist, bins = np.histogram(values, bins=10)
    plt.show()

    # get biggest bin
    biggest_bin = bins[0:2]

    # get data in biggest bin
    max_indexs = np.where(np.logical_and(class_data >= biggest_bin[0], class_data <=  biggest_bin[1]))
    max_values = class_data[max_indexs]

    # 
'''
'''

# Below is code to view pixel rows and cols on QGIS.
# https://gis.stackexchange.com/questions/261504/getting-row-col-on-click-of-a-pixel-on-a-qgis-map

from qgis.gui import QgsMapTool
from PyQt5.QtCore import Qt, QPoint
from math import floor

# references to QGIS objects 
canvas = iface.mapCanvas() 
layer = iface.activeLayer() 
data_provider = layer.dataProvider()

# properties to map mouse position to row/col index of the raster in memory 
extent = data_provider.extent() 
width = data_provider.xSize() if data_provider.capabilities() & data_provider.Size else 1000 
height = data_provider.ySize() if data_provider.capabilities() & data_provider.Size else 1000 
xres = extent.width() / width 
yres = extent.height() / height

class ClickTool(QgsMapTool): 
    def __init__(self, canvas):
        QgsMapTool.__init__(self, canvas)
        self.canvas = canvas 
    def canvasPressEvent(self, event):
        if event.button() == Qt.LeftButton: 
            x = event.pos().x()
            y = event.pos().y()
            point = self.canvas.getCoordinateTransform().toMapCoordinates(x, y)
            if extent.xMinimum() <= point.x() <= extent.xMaximum() and \
                extent.yMinimum() <= point.y() <= extent.yMaximum():
                col = int(floor((point.x() - extent.xMinimum()) / xres))
                row = int(floor((extent.yMaximum() - point.y()) / yres))
                print (row, col)

tool = ClickTool(iface.mapCanvas())
iface.mapCanvas().setMapTool(tool)

'''