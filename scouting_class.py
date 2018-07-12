
'''
    Project:
        Tissue sample scouting

    Author:
        Irah W
        
    File:
        Tissue scouting class


    Notes:
        This file contains all required for creating the tissue sampling scouting path


'''

import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np 
import cv2
from itertools import cycle
import heapq
from PIL import Image, ImageDraw, ImageFont
import re
import argparse
import math
import sys
from skimage.morphology import binary_erosion, binary_dilation

class Scout(object):
    '''
         Scout class
    '''

    def __init__(self, histograms_data, greyscale_class_imgs, indice_data):
        '''
            Store data from previous zoning class
        '''

        self.histograms_data = histograms_data
        self.greyscale_class_imgs = greyscale_class_imgs
        self.px_list = []
        self.indice_data = indice_data
        self.class_name_list = histograms_data.keys()
        self.path = None


    def make_list(self, img, cls):
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


    def img_from_list(self, px_list, shape):
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



    def px_list_from_binned_classes(self, show_hist=False, filter_small_clusters=True):
        '''
            convert a list of classified images to a list of pixels
            that occur in the max-bin of each image
            utilise image erosion to exclude small clusters
            
            Parameters
            ----------
            class_img_list : list of classified image arrays with the class name: [(class_name, image_array), ...]
            nbins : int - number of bins to use for max-binning partitions
            show_hist : Boolean - print intermediate steps for data checking
            
            (outputs from classifier)
            class_avg : numpy array, average value for each class
            class_avg : numpy array, [min, max] value for each class
            
            
            Returns
            -------
            out : list of tuples [(dist to p0, px row, px column, px class), ...]
        '''
        
        def remove_small_pixel_clusters(
                img,
                selem = np.array([
                    [0,0,1,0,0],
                    [0,1,1,1,0],
                    [1,1,1,1,1],
                    [0,1,1,1,0],
                    [0,0,1,0,0],
                     ]),
                                        ):
            img_bin = img.astype(np.bool)
            img_bin = binary_erosion(img_bin, selem=selem)
            img_bin = binary_dilation(img_bin, selem=selem)
            return np.where(img_bin,img,0)
        
        # iterate through histogram data
        if show_hist:
            index=1
            ncols = len(self.histograms_data)
            nrows = 3
            plt.figure()
            plt.tight_layout()

        for class_name in self.histograms_data:

            n, bins = self.histograms_data[class_name]

            i_maxbin = np.argmax(n)

            class_img = self.greyscale_class_imgs[class_name]
            img = np.where(class_img == 255, self.indice_data, 0)
            # plt.figure()
            # plt.imshow(img)

            if show_hist:

                ax = plt.subplot(nrows, ncols, index)
                #ax.set_xlabel('min{},max{}'.format(np.min(img[img>0]), np.max(img)))
                plt.imshow(img)

                ax = plt.subplot(nrows, ncols, index + (ncols*1))
                flat = img.flatten()
                flat = flat[flat!=0]
                plt.hist(flat, bins)
                # ax.set_xlim(
                #     np.min(img[img>0])-1,
                #     np.max(img)+2)
                #ax.set_xlabel('MaxBin[{}, {}]'.format(bins[i_maxbin], bins[i_maxbin+1]))

            if i_maxbin == len(n):
                img = np.where((img >= bins[-1]), img,0)
            else:
                img = np.where((img >= bins[i_maxbin])&(img < bins[i_maxbin+1]), img, 0)

            if filter_small_clusters:
                img = remove_small_pixel_clusters(img)
            
            if show_hist:
                ax = plt.subplot(nrows, ncols, index + (ncols*2))
                #ax.set_xlabel('min{},max{}'.format(np.min(img[img>0]), np.max(img)))
                plt.imshow(img)
                index+=1

            self.px_list += self.make_list(img, class_name)

            
        
        if show_hist:
            plt.show()



    def path_to_line(self, path):
        line = [path[0]] #the fisrt point, p0, is differnt format tuple (row, col)
        for p in path[1:]: # get (row, col) from rest of path points
            line.append((p[1],p[2]))
        line= [(p1,p0) for p0,p1 in line]
        return line




    def get_distance(self, p0, px_list):
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
        def dist(p0, p1):
            dr = p0[0] - p1[0]
            dc = p0[1] - p1[1]
            return math.sqrt(dr**2 + dc**2)
            # return (dr,dc,dr**2,dc**2,(dr**2 + dc**2)**(0.5))

        def process(px):
            return(dist(px, p0), px[0], px[1], px[2])

        return [process(px) for px in px_list]

    def get_path(self, p0, min_px_dist=0):
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
        def filter_by_class(plist, vis_list):
            out=list()
            for  x in plist:
                if not(x[-1] in vis_list):
                    out.append(x)
                # else:
                #     print(x0,x1,x2,x3 )
                #     sys.exit()
            return out


        vis_list = []
        plist = self.px_list
        
        #(dist, r, c, class no.)
        h = self.get_distance(p0, plist) #->[(dist to p0, px row, px column, px class), ...]

        #heapiy the list by distance
        heapq.heapify(h)

        #get closest px from list, add to out list
        p1 = heapq.heappop(h)
        path=[p0, p1]
        vis_list.append(p1[3])

        while True:
            # print('check',len(h), path, vis_list, p1[3], not(p1[3] in vis_list))
            #update start pos'n
            p0 = (p1[1], p1[2])

            #get px distances relative to p0
            plist = filter_by_class(plist, vis_list)
            if len(plist)==0:
                break
            h = self.get_distance(p0, plist)
            heapq.heapify(h)
            
            #remove closest point, append to path
            p1 = heapq.heappop(h)
            path+=[p1]
            vis_list.append(p1[3])
        vis_list.sort() #ensure same order with every path genereted
        return path, vis_list


    def draw_path_from_start(self, bw, path=None):    
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
        
        for i, cls in enumerate(self.class_name_list):
            mask = self.img_from_list([x for x in self.px_list if x[-1]==cls], base_shape)
            col = Image.new('RGB',bw.size, colour_list[i])
            mask = Image.fromarray(mask).convert("1")
            base = Image.composite(base,col,mask).convert('RGB')
        
        # no path defined render base image only
        if path is None:
            return base
        
        #extract points from path
        line = self.path_to_line(path)

        #draw path plus numeric labels
        base = base.convert('RGBA')
        # make a blank image for the text, initialized to transparent text color
        txt = Image.new('RGBA', base.size, (255,255,255,0))
        #fnt = ImageFont.truetype('C:\ProgramData\Anaconda3\Library\lib\DejaVuSerif.ttf', 100)
        d = ImageDraw.Draw(txt)
        
        d.line(line, fill=(255,0,0,90), width=25)
        
        for p in line:
            r = [p[0]-4, p[1]-4, p[0]+4, p[1]+4]
            d.rectangle(r, fill=(255,0,0,64))
            #d.text(p, "{}".format(line.index(p)), font=fnt, fill=(255,0,0,128), align='left')    
        
        out = Image.alpha_composite(base, txt)
        return(out) 




    def find_path(self, p0=None, user_input=False):
        '''
            Compute the best path for tissue scouting

            Parameters
            ----------
            p0 : is the starting point in (row, colum) format
            user_input : set to True for interactive plotting
        '''

        if user_input:
            path=None
            p0=None
            print('User input mode. Double click a pixel to define the starting location')
            # convert input array to grayscale image
            bw = Image.fromarray(self.indice_data).convert("L")    
        
            # plot, user selects the start point, then generate path
            base = self.draw_path_from_start(bw)
            
            fig = plt.figure()
            plt.imshow(np.array(base))
            plt.ion()
            
            def onclick(event):
                p0 = (event.ydata, event.xdata)
                pString = '{}'.format(p0)
                print(pString)
                if event.dblclick:
                    p0 = (int(event.ydata), int(event.xdata)) # retrieve start point, p0, from double click on image
                    print('start point (p0):',p0)
                    path, class_list = self.get_path(p0) # get list of points, and classes
                    
                    print('Path points:', self.path_to_line(path))
                    out_img = self.draw_path_from_start(bw, path=path)
                    plt.imshow(out_img)
                    plt.draw()
            
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show(block=True)
        
        else:
            #start from the centre of the image
            if p0 is None:
                #if no start point defined start from centre of image
                p0 = (int(self.indice_data.shape[1]//2), int(self.indice_data.shape[0]//2))
            # print('start point (p0):',p0)
            path, vis_list = self.get_path(p0)
            self.path = self.path_to_line(path)
           # print('Path points:', self.path)
        


class Colours:
    """
    Class used to cylically iterate over a list of colours
    and return an RGB value based on an input int
    """
    def __init__(self):
    
        #https://www.rapidtables.com/web/color/RGB_Color.html
        c = """
            Red #FF0000 (255,0,0)
            Lime    #00FF00 (0,255,0)
            Blue    #0000FF (0,0,255)
            Yellow  #FFFF00 (255,255,0)
            Cyan / Aqua #00FFFF (0,255,255)
            Magenta / Fuchsia   #FF00FF (255,0,255)
            Maroon  #800000 (128,0,0)
            Olive   #808000 (128,128,0)
            Green   #008000 (0,128,0)
            Purple  #800080 (128,0,128)
            Teal    #008080 (0,128,128)
            Navy    #000080 (0,0,128)
        """
        
        c = re.findall(r'(\([\d\,]*\))', c)
        self.cList = [eval(x) for x in c]
        self.len = len(self.cList)
        
    def __getitem__(self, i):
        return self.cList[i%self.len]  
