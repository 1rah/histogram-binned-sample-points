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
from skimage.transform import resize
import geopandas as gpd
import affine
import utm
from rasterio import crs
from rasterio.warp import calculate_default_transform, reproject, Resampling

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


def rescale_to_gsd(img_array, img_affine, new_gsd):
    img_gsd = abs(np.array((img_affine.a, img_affine.e)))
    new_shape = (
            img_array.shape / 
            (new_gsd / img_gsd)
            )
    new_shape = new_shape.astype(np.int)
    print(new_shape)
    new_img_array = resize(
            img_array,
            new_shape,
            )
    x_rs, y_rs = np.array(img_array.shape) / new_img_array.shape
    new_affine = list(img_affine[0:6])
    new_affine[0] = new_affine[0] * x_rs
    new_affine[4] = new_affine[4] * y_rs
    new_affine = affine.Affine(*new_affine)
    return new_img_array, new_affine


def bound_to_utm(bnds):
    lng1, lat1, lng2, lat2 = bnds
    
    u1 = utm.from_latlon(lat1, lng1)
    u2 = utm.from_latlon(lat2, lng2)
    if u1[-2:] != u2[-2:]:
        raise Exception('geometry split across multiple UTM zones')
    
    letter = u1[-1]
    zone = u1[-2]
    # get reference zone
    if letter in 'CDEFGHJKLM':
        new_crs = 32700 + zone
    else:
        assert letter in 'NPQRSTUVWXX'
        new_crs = 32600 + zone
    return new_crs


def convert_img_to_utm(src_array, src, dst_crs, i=1):
        
    # Calculate the ideal dimensions and transformation in the new crs
    dst_affine, dst_width, dst_height = calculate_default_transform(
            src['crs'],
            dst_crs,
            src['width'],
            src['height'],
            *src['bounds']
            )
    # Reproject and write band i
    dst_array = np.empty((dst_height, dst_width), dtype='uint8')
    reproject(
        # Source parameters
        source=src_array,
        src_crs=src['crs'],
        src_transform=src['affine'],
        # Destination paramaters
        destination=dst_array,
        dst_transform=dst_affine,
        dst_crs=dst_crs,
        # Configuration
        resampling=Resampling.nearest,
        )
    
    profile = src
    # update the relevant parts of the profile
    profile.update({
        'crs': dst_crs,
        'transform': dst_affine,
        'affine': dst_affine,
        'width': dst_width,
        'height': dst_height
    })
    
    return dst_array, profile


#NOTES
#input "profile" wll be standard to mask images, as from same image source
if __name__ is '__main__':
     
    zone_masks = [
     'D:\\test-inputs\\oleksi-issues-normal-points\\srcGSD0.0MjenksC4A0.0S0-z00.tif',
     'D:\\test-inputs\\oleksi-issues-normal-points\\srcGSD0.0MjenksC4A0.0S0-z01.tif',
     'D:\\test-inputs\\oleksi-issues-normal-points\\srcGSD0.0MjenksC4A0.0S0-z02.tif',
     'D:\\test-inputs\\oleksi-issues-normal-points\\srcGSD0.0MjenksC4A0.0S0-z03.tif',
     ]
    
    zone_mask =zone_masks[3]
    with rasterio.open(zone_mask) as src:
        img = src.read(1)
        src_profile = src.profile.copy()
        src_profile.update({'bounds':src.bounds})
        dst_crs = crs.CRS.from_epsg(bound_to_utm(src.bounds))
        img_utm, new_profile = convert_img_to_utm(img, src_profile, dst_crs)
    
    
        #TODO, convert to UTM, change to 1m GSD
    
#    plt.figure()
#    plt.imshow(img)
#    
#    img = remove_small_pixel_clusters(img)
#    
#    plt.figure()
#    plt.imshow(img)
    
    img_utm = rescale_to_gsd(img_utm, new_profile['affine'], 20)[0]
    img_utm = np.where(img_utm>0,255,0)
    
    #todo change masks to utm
    shp = shapes(img_utm, mask=img_utm.astype(bool), connectivity=4)
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
        g1 = g.geometry.buffer(2)
        g1 = g1.geometry.buffer(-6)
        g1 = g1.geometry.buffer(3)
        g2 = g1[g1.area > 0]
        g2.plot()
        
    plt.imshow(img_utm)
        
    #TODO nest polygons into multipolygon

    