
'''
Apply histogram binning to a classified image, to select the maximum frequency 
points within each bin.

Starting from a user specified point, find the short path to visit a point in
each of the maximum binned classes.
'''

import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np 
import classification as cn
from cv2 import erode
from itertools import cycle
import heapq
from PIL import Image, ImageDraw, ImageFont
import re
import argparse

def make_list(img, cls):
    """
    convert image to a list of pixels which have non-zero values
    
    Parameters
    ----------
    img : numpy array - graysacle image from input tif, classified to single class image only
    cls : what class label to assign each pixel
    
    Returns
    -------
    list of tuples [(px row, px column, px class), ...]
    
    """
    out = np.where(img>0)
    return list(zip(out[0], out[1], cycle((cls,))))


def dist(p0, p1):
    """
    Return straight line distance between two pixels
    
    Parameters
    ----------
    p0 : tuple in (pixel row, pixel columm)
    p1 : tuple in (pixel row, pixel columm)
    
    Returns
    -------
    out : straight line distance as float
    """
    dr = p0[0] - p1[0]
    dc = p0[1] - p1[1]
    return (dr**2 + dc**2)**(1/2)


def get_distance(p0, px_list):
    """
    convert list of pixels distance between two pixels
    
    Parameters
    ----------
    p0 : tuple in (px row, px columm)
    px_list : list of tuples [(px row, px column, px class), ...]
    
    
    Returns
    -------
    out : list of tuples [(dist to p0, px row, px column, px class), ...]
    """
    def process(px):
        return(dist(px, p0), px[0], px[1], px[2])
    return [process(px) for px in px_list]


def px_list_from_binned_classes(class_img_list, nbins = 4, erosion_kernel = 3, do_plots=True):
    """
    convert a list of classified images to a list of pixels
    that occur in the max-bin of each image
    utilise image erosion to exclude small clusters
    
    Parameters
    ----------
    class_img_list : list of classified image arrays with the class name: [(class_name, image_array), ...]
    nbins : int - number of bins to use for max-binning partitions
    erosion_kernel : int - size of kernel to use on erosion filter
    do_plots : Boolean - print intermediate steps for data checking
    
    (outputs from classifier)
    class_avg : numpy array, average value for each class
    class_avg : numpy array, [min, max] value for each class
    
    
    Returns
    -------
    out : list of tuples [(dist to p0, px row, px column, px class), ...]
    """
    if do_plots:
        nrows = 4
        ncols = len(class_img_list)
        plt.figure()
    
    px_list=list()
    for index, (class_name, img) in enumerate(class_img_list, 1):    
        if do_plots:
            plt.subplot(nrows, ncols, index)
            plt.imshow(img)
        
        data = img.flatten()
        data = data[data>0]
        binwidth = (max(data) - min(data)) / nbins
        
        if do_plots:
            plt.subplot(nrows, ncols, index + ncols)
            n, bins, patches = plt.hist(data, bins=np.arange(min(data), max(data) + binwidth, binwidth))
            
        else:
            # bin without plotting histogram
            n, bins = np.histogram(data, bins=np.arange(min(data), max(data) + binwidth, binwidth))
        
        i_maxbin = np.argmax(n)
        if i_maxbin == len(n):
            img = np.where((img >= bins[-1]), 0, img)
        else:
            img = np.where((img >= bins[i_maxbin])&(img < bins[i_maxbin+1]), 0, img)
        
        if do_plots:
            plt.subplot(nrows, ncols, index + (ncols*2))
            plt.imshow(img)
        
        #use erosion to filter to eliminate small clusters
        kernel = np.ones((erosion_kernel,erosion_kernel), np.uint8)
        img = erode(img, kernel, iterations=1)
        
        px_list += make_list(img, class_name)
        
        if do_plots:
            plt.subplot(nrows, ncols, index + (ncols*3))
            plt.imshow(img)
    
    if do_plots:
        plt.show()
        
    return px_list


def get_path(p0, px_list, min_px_dist=0):
    """
    calculate the path, starting at p0, then selecting the closest point from
    the closest unvisited class.
    
    Parameters
    ----------
    p0 : tuple in (pixel row, pixel columm)
    px_list : list of tuples [(px row, px column, px class), ...]
    min_px_dist : > 0 if there is a minimum separation requirment between points
    
    Returns
    -------
    path : list of px points in order of visit [(p0 row, p0 column), (px row, px column, px class), ...]
    class_list : list of the class names in px_list
    """
    class_list = []
    
    #(dist, r, c, class no.)
    h = get_distance(p0, px_list) 
    
    #heapiy the list by distance
    heapq.heapify(h)
    
    #get closest px from list, add to out list
    p1 = heapq.heappop(h)
    path=[p0, p1]
    
    #remove all px from the visited class
    px_list = [(x[1],x[2],x[3]) for x in h if x[3]!=p1[3]]
    class_list.append(p1[3])

    while len(px_list) > 0:
        #update start pos'n
        p0 = (p1[1], p1[2])
        #get px distances relative to p0
        h = get_distance(p0, px_list)
        
        #ensure min distance requirement
        if min_px_dist > 0:
            h = [x for x in h if x[0]>= min_px_dist]
        heapq.heapify(h)
        p1 = heapq.heappop(h)
        path+=[p1]
        px_list = [(x[1],x[2],x[3]) for x in h if x[3]!=p1[3]]
        class_list.append(p1[3])
    class_list.sort() #ensure same order with every path genereted
    return path, class_list


def draw_path_from_start(bw, px_list, class_name_list, path=None):    
    """
    draw an image showing the determined path, overlayed on the input image
    
    Parameters
    ----------
    bw : imput image
    path : list of px points in order of visit [(p0 row, p0 column), (px row, px column, px class), ...]
    
    Returns
    -------
    out : output image
    """
    colour_list = Colours()
    
    base = bw.copy()
    base_shape = (bw.size[1],bw.size[0])
    
    for i, cls in enumerate(class_name_list):
        mask = img_from_list([x for x in px_list if x[-1]==cls], base_shape)
        col = Image.new('RGB',bw.size, colour_list[i])
        mask = Image.fromarray(mask).convert("1")
        base = Image.composite(base,col,mask).convert('RGB')
    
    # no path defined render base image only
    if path is None:
        return base
    
    #extract points from path
    line = [path[0]] #the fisrt point, p0, is differnt format tuple (row, col)
    for p in path[1:]: # get (row, col) from rest of path points
        line.append((p[1],p[2]))
    line= [(p1,p0) for p0,p1 in line]

    #draw path plus numeric labels
    base = base.convert('RGBA')
    # make a blank image for the text, initialized to transparent text color
    txt = Image.new('RGBA', base.size, (255,255,255,0))
    fnt = ImageFont.truetype('C:\ProgramData\Anaconda3\Library\lib\DejaVuSerif.ttf', 40)
    d = ImageDraw.Draw(txt)
    
    d.line(line, fill=(255,0,0,64), width=5)
    
    for p in line:
        r = [p[0]-4, p[1]-4, p[0]+4, p[1]+4]
        d.rectangle(r, fill=(255,0,0,64))
        d.text(p, "{}".format(line.index(p)), font=fnt, fill=(255,0,0,128), align='left')    
    
    out = Image.alpha_composite(base, txt)
    return(out)


def img_from_list(px_list, shape):
    """
    create a binary image array using a list of non-zero pixel values.
    the ouput image shape will be the same as the unclassified input image
    
    Parameters
    ----------
    px_list : list of tuples [(px row, px column, px class), ...]
    shape : shape of the original unclassified image
    
    Returns
    -------
    new : ouptut image
    """
    new = np.full(shape, 255)
    for r,c,_cls in px_list:
        new[r,c] = 0
    return new


class Colours:
    """
    Class used to cylically iterate over a list of colours
    and return an RGB value based on an input int
    """
    def __init__(self):
    
        #https://www.rapidtables.com/web/color/RGB_Color.html
        c = """
         	Red	#FF0000	(255,0,0)
         	Lime	#00FF00	(0,255,0)
         	Blue	#0000FF	(0,0,255)
         	Yellow	#FFFF00	(255,255,0)
         	Cyan / Aqua	#00FFFF	(0,255,255)
         	Magenta / Fuchsia	#FF00FF	(255,0,255)
         	Maroon	#800000	(128,0,0)
         	Olive	#808000	(128,128,0)
         	Green	#008000	(0,128,0)
         	Purple	#800080	(128,0,128)
         	Teal	#008080	(0,128,128)
         	Navy	#000080	(0,0,128)
        """
        
        c = re.findall(r'(\([\d\,]*\))', c)
        self.cList = [eval(x) for x in c]
        self.len = len(self.cList)
        
    def __getitem__(self, i):
        return self.cList[i%self.len]  


def run_toms_classifier(indice_path):
    """
    Driver function to run Tom's code to generate classes
    """
    # define paths for using Tom's code
    classesNpy = indice_path.split('.')[0] + '_.npy'
    smoothed_results_path = indice_path.split('.')[0] + '_smoothed.tif'

    # define method, smoothing, classes
    m = 'equalInterval'
    s = 100
    c = 4

    # smooth
    cn.filter(indice_path, smoothed_results_path, smoothing=[s], res=None)
    # classify
    cn.classify(smoothed_results_path, classesNpy, m, [c], class_breaks=None) 
    collated_list = np.load(classesNpy, allow_pickle=True)
    class_avg = collated_list[1]
    class_bounds = collated_list[2]
    
    return class_avg, class_bounds



def read_arguments():
   '''
      Function reads the pickle from the zoning code: [0] = raster indice, [1...n].
      

   '''

   # get the pickle files location
   parser = argparse.ArgumentParser()
   parser.add_argument('pickle', type=str, help='src pickle path')
   parser.add_argument('-i',action='store_true', help='run interactive mode')
   args = parser.parse_args()

   return args

"""
-------------------------------------------------------------------------------
Main Driver Function
-------------------------------------------------------------------------------
"""
if __name__ == '__main__':
#    args = read_arguments()
    
#    print(args)
    
       # read pickle
#    p = np.load(args.pickle, allow_pickle=True)
    p = np.load('clarks_30ccci2.0stddev5bw_classes.npy', allow_pickle=True)
    indice_data = p[0]
    indice_data = (indice_data*255).astype(np.uint8)
    
    img_list = p[1:]
    
    class_img_list = []
    class_name_list = []
    for i, img in enumerate(img_list):
        img2 = np.where(img>0,indice_data,0)
        if len(np.where(img2>0)[0]) > 0:
            class_img_list.append((i, img2))
            class_name_list.append(i)
    
    
#    print(class_name_list)
#    plt.imshow(class_img_list[0])
#    plt.show(block=True)
    
#    indice_path = '20170929T184129_T10SFG_S2B_ndvi_gray.tif'
#    
#    #open tif file, and read image layer as array
#    ds = gdal.Open(indice_path)
#    indice_data = ds.GetRasterBand(1).ReadAsArray() #nans for all -1000s       
#    indice_data = indice_data.astype(np.float32)
#    indice_data[indice_data <= -10000] = np.nan
#    
#    # run toms classifier code to retrieve the class_bounds
#    class_avg, class_bounds = run_toms_classifier(indice_path)
#
#    # generate list of classified images 1 image per class as [(class_name, image), ...]
#    class_img_list = list()
#    class_name_list = 'low med-low med-high high'.split() #user defined class names
#    for ci, bound in enumerate(class_bounds):
#        lwr, upr = bound[0], bound[1]
#        # set px outside class range to 0
#        img = np.where((indice_data < lwr) | (indice_data >=  upr), 0, indice_data) 
#        class_name = class_name_list[ci]
#        class_img_list.append((class_name, img))
#
    # generate a list of pixels that occur in the max bin of each class
    # and that meet a minimum cluster size, based on erosion
    px_list = px_list_from_binned_classes(class_img_list, nbins=2, do_plots=False)
    
    # convert input array to grayscale image
    bw = Image.fromarray(indice_data).convert("L")    

    # plot, user selects the start point, then generate path
    
    
    base = draw_path_from_start(bw, px_list, class_name_list)
    
    fig = plt.figure()
    plt.imshow(np.array(base))
    plt.ion()
    
    def onclick(event):
        p0 = (event.ydata, event.xdata)
        pString = '{}'.format(p0)
        print(pString)
        if event.dblclick:
            p0 = (int(event.ydata), int(event.xdata)) # retrieve start point, p0, from double click on image
            print('p0:',p0)
            path, class_list = get_path(p0, px_list) # get list of points, and classes
            print(path, class_list)
            out_img = draw_path_from_start(bw, px_list, class_name_list, path=path)
            plt.imshow(out_img)
            plt.draw()
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)
#    
    
    



    
    

    

