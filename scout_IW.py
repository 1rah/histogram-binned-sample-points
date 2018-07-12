

'''
    Rough script on scouting code


'''

import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np 
import classification as cn


    

if __name__ == '__main__':
    
    indice_path = '20170929T184129_T10SFG_S2B_ndvi_gray.tif'
#    indice_path = 'AvonmoreFogdon_index_ccci.tif'
    ds = gdal.Open(indice_path)
    indice_data = ds.GetRasterBand(1).ReadAsArray() #nans for all -1000s       
    indice_data = indice_data.astype(np.float32)
    indice_data[indice_data <= -10000] = np.nan

    # define paths for using Tom's code
    classesNpy = indice_path.split('.')[0] + '_.npy'
    smoothed_results_path = indice_path.split('.')[0] + '_smoothed.tif'
    jsonName = indice_path.split('.')[0] + '_points.json'

    # define method, smoothing, classes
    m = 'equalInterval'
    s = 100
    c = 3


    # smooth
    cn.filter(indice_path, smoothed_results_path, smoothing=[s], res=None)
    # classify
    cn.classify(smoothed_results_path, classesNpy, m, [c], class_breaks=None) 

    collated_list = np.load(classesNpy, allow_pickle=True)
    class_avg = collated_list[1]
    class_bounds = collated_list[2]

    #subplot(nrows, ncols, index)
    
    indice_data = np.nan_to_num(indice_data)
    plt.figure()
    nrows = 3
    ncols = len(class_bounds)
    
    nbins = 4
    
    
    def add_to_list(img, bgv=0):
        out = list()
        rows, cols = img.shape
        for r in range(rows):
            for c in range(cols):
                pxv = img[r,c]
                if pxv != bgv:
                    out.append({'row':np.array(r,dtype=int), 'col':np.array(c,dtype=int), 'pxv':pxv})
        return(out)
    
    
    out=list()
    for index, bound in enumerate(class_bounds, 1):
        lwr, upr = bound[0], bound[1]
        img = np.where((indice_data < lwr) | (indice_data >  upr), 0, indice_data)
        plt.subplot(nrows, ncols, index)
        plt.imshow(img)
        
        
        data = img.flatten()
        data = data[data>0]
        plt.subplot(nrows, ncols, index + ncols)
        binwidth = (max(data) - min(data)) / nbins
        n, bins, patches = plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth))
        
        i_maxbin = np.argmax(n)
        bin_upr = i_maxbin
        if i_maxbin == len(n):
            img2 = np.where((img >= bins[-1]), 0, img)
        else:
            img2 = np.where((img >= bins[i_maxbin])&(indice_data < bins[i_maxbin+1]), 0, img)
        
        
        out += add_to_list(img2, bgv=0)
        
        plt.subplot(nrows, ncols, index + (ncols*2))
        plt.imshow(img2)
        plt.imsave(fname='img_out_{}.tif'.format(index),
                   arr=img,
                   format='tif')
    
    plt.show()
    

    """
    Find Path
    
    Maybe apply dilate then erode filter?
    
    Define starting point x,y
    convert pixels to list of x,y coords, with class label and px value
    
    find closest point from any class
    get class of point - remove class points from explore list
    repeat until no classes left
    """


    def get_class(pxv, class_bounds):
        if pxv == 0:
            return 0
        for c, bound in enumerate(class_bounds, 1):
            lwr, upr = bound[0], bound[1]
            if (pxv >= lwr)&(pxv < upr):
                return c
        # greater than max value of max class
        return c+1
        
    
    def make_xy_list(img, bgv=0):
        out = list()
        rows, cols = img.shape
        for r in range(rows):
            for c in range(cols):
                pxv = img[r,c]
                if pxv != bgv:
                    out.append({'row':np.array(r,dtype=int), 'col':np.array(c,dtype=int), 'pxv':pxv})
        return(out)
    
    
#    df = make_xy_list(indice_data)
#    df = pd.DataFrame(df)
#    df['class'] = df.pxv.apply(lambda x: get_class(x,class_bounds))
#    
##    imgc = np.zeros(indice_data.shape+(3,))
#    imgc = np.zeros(indice_data.shape)
#    
#    
#    for i, r in df.iterrows():
#        if r['class'] == 2:
#            imgc[r['row'].astype(np.int), r['col'].astype(np.int)] = 254
#    
#    plt.figure()
#    plt.imshow(imgc)
#    
    def tabulate(x):
        XX,YY = np.meshgrid(np.arange(x.shape[1]),np.arange(x.shape[0]), sparse=True)
        table = np.vstack((x.ravel(),XX.ravel(),YY.ravel())).T
        return(table)
    
    

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