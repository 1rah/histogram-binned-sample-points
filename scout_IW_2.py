

'''
    Rough script on scouting code


'''

import matplotlib.pyplot as plt
from osgeo import gdal
import numpy as np 
import classification as cn
import cv2
from itertools import cycle
import heapq
from PIL import Image, ImageDraw, ImageFont
import re
    

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
    c = 4

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
    nrows = 4
    ncols = len(class_bounds)
    
    nbins = 4
    
    
#    def add_to_list(img, cls, bgv=0):
#        out = list()
#        rows, cols = img.shape
#        for r in range(rows):
#            for c in range(cols):
#                pxv = img[r,c]
#                if pxv > bgv:
#                    out.append({'row':np.array(r,dtype=int), 'col':np.array(c,dtype=int), 'pxv':pxv})
#        return(out)
    
    def make_list(img, cls):
        out = np.where(img>0)
        return list(zip(out[0], out[1], cycle((cls,))))
    
    
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
            img = np.where((img >= bins[-1]), 0, img)
        else:
            img = np.where((img >= bins[i_maxbin])&(indice_data < bins[i_maxbin+1]), 0, img)
        
        
        plt.subplot(nrows, ncols, index + (ncols*2))
        plt.imshow(img)
        
        
        kernel = np.ones((4,4), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        
        out += make_list(img, index)
        
        plt.subplot(nrows, ncols, index + (ncols*3))
        plt.imshow(img)
    
    plt.show()
    
    

        
    #join lists, add DIST field, add to priorty queue by DIST, pluck lowest val (check dist at list X)
    # x = column, y = row
    #c_start = 262, r_start = 304
    
    
    def dist(p0, p1):
        dr = p0[0] - p1[0]
        dc = p0[1] - p1[1]
        return (dr**2 + dc**2)**(1/2)
    
    def get_distance(p0, px_list):
        def process(px):
            return(dist(px, p0), px[0], px[1], px[2])
        return [process(px) for px in px_list]
    
    def get_path(p0, px_list, min_px_dist=0):
        #(dist, r, c, class no.)
        h = get_distance(p0, px_list) 
        
        #heapiy by distance
        heapq.heapify(h)
        
        #get closest px from list, add to out list
        p1 = heapq.heappop(h)
        path=[p0, p1]
        
        #remove all px from the visited class
        px_list = [(x[1],x[2],x[3]) for x in h if x[3]!=p1[3]]
    
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
        return path
            
    p0=(234,610) #ydata, xdata
    px_list=out
    path = get_path(p0, px_list)
    
    
    
    #https://www.rapidtables.com/web/color/RGB_Color.html
    colours = """
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
#    colours = re.findall(r'(\([\d\,]*\))',re.sub(r'[^\d\,\(\)]','',colours))
    colours = re.findall(r'(\([\d\,]*\))',colours)
    colours = [eval(x) for x in colours]
    
    def img_from_list(px_list, img=indice_data):
        new = np.full_like(img,255)
        for r,c,cls in px_list:
            new[r,c] = 0
        return new
    
    bw = Image.fromarray(indice_data).convert("L")
    
    for cls in [1,2,3]:
        mask = img_from_list([x for x in px_list if x[-1]==cls])
        col = Image.new('RGB',bw.size,colours[cls%len(colours)])
        mask = Image.fromarray(mask).convert("1")
        bw = Image.composite(bw,col,mask).convert('RGB')
    

