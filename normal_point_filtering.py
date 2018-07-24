# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 16:40:12 2018

@author: FluroSat
"""
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import cascaded_union
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation
import geopandas as gpd

def write_bin_array(array, src_tif, zone_file_name):
    #extract profile from src_tif
    with rasterio.open(src_tif) as src:
        profile = src.profile
    
    # write array to disk
    with rasterio.open(zone_file_name, 'w', **profile) as dst:
        dst.write(array, 1)
    
def get_max_bin_mask():
    #take one classified image, return a mask of the max binned pixels
    #default to 3 bins
    #return binary array
    pass

def convert_gsd():
    #up sample to fixed gsd - 5m (maybe 2.5)
    pass

def clip_boundary():
    #erode the input image, to shink it down (15m border ~3px)
    #this will shrink each class file individually
    
    #alternatively could use the polygon of the raster
    pass

def raster_to_poly():
    #convert raster to polygon
    pass

def drop_small_regions():
    #try a few things here
    #i.e shrink polygons, then drop small areas, then grow remaning regions
    
    #could try filling the region with 'circles'
    
    #maybe drop the resolution
    pass

def remove_small_pixel_clusters(img):
    n=10    
    selem = [0,]*n
    selem[len(selem)//2] = 1
    selem = [selem,]*n
    selem[len(selem)//2] = [1,]*n
    selem = np.array(selem)
    
    img_bin = img.astype(np.bool)
    img_bin = binary_erosion(img_bin, selem=selem)
    img_bin = binary_dilation(img_bin, selem=selem)
    return np.where(img_bin,img,0)


#NOTES
#input "profile" wll be standard to mask images, as from same image source
if __name__ is '__main__':
     
    zone_masks = [
     'D:\\test-inputs\\oleksi-issues-normal-points\\srcGSD0.0MjenksC4A0.0S0-z00.tif',
     'D:\\test-inputs\\oleksi-issues-normal-points\\srcGSD0.0MjenksC4A0.0S0-z01.tif',
     'D:\\test-inputs\\oleksi-issues-normal-points\\srcGSD0.0MjenksC4A0.0S0-z02.tif',
     'D:\\test-inputs\\oleksi-issues-normal-points\\srcGSD0.0MjenksC4A0.0S0-z03.tif',
     ]
    
    zone_mask =zone_masks[0]
    with rasterio.open(zone_mask) as src:
        img = src.read(1)
        profile = src.profile.copy()
    
#    plt.figure()
#    plt.imshow(img)
#    
#    img = remove_small_pixel_clusters(img)
#    
#    plt.figure()
#    plt.imshow(img)
    
    shp = shapes(img, mask=img.astype(bool), connectivity=4)
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(shp)
        )
    geoms = list(results)
    
    g = gpd.GeoDataFrame.from_features(geoms)
    g = g[g.area>0]
    
    g.plot()
    
    if len(g)>0:
        b=3
        g1 = g.geometry.buffer(2*b)
        g1 = g1.geometry.buffer(-6*b)
        g1 = g1.geometry.buffer(2*b)
        g2 = g1[g1.area > 0]
        g2.plot()
        
    plt.imshow(img)
        

    
    