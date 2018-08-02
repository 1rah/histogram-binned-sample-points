# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30

@author: Irah Wajchman
"""
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, Polygon
from shapely.ops import cascaded_union
from shapely.geometry.multipolygon import MultiPolygon
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import geopandas as gpd
import affine
import utm
from rasterio import crs
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio import features
import argparse
import os

# ----------------------------------------------------------------------------
# Smooth Mask to Polygon
# ----------------------------------------------------------------------------

def rescale_to_gsd(img_array, img_affine, new_gsd=5):
    img_gsd = abs(np.array((img_affine.a, img_affine.e)))
    #calc new shape
    new_shape = (
            img_array.shape / 
            (new_gsd / img_gsd)
            )
    new_shape = new_shape.astype(np.int)
    #resize to new shape
    new_img_array = resize(
            img_array,
            new_shape,
            preserve_range=True,
            clip=True,
            )
    #adjust affine to sui new gsd
    x_rs, y_rs = np.array(img_array.shape, dtype=np.float) / new_img_array.shape
    new_affine = list(img_affine[0:6])
    new_affine[0] = new_affine[0] * x_rs
    new_affine[4] = new_affine[4] * y_rs
    new_affine = affine.Affine(*new_affine)
#    print(img_array.shape,  new_img_array.shape, x_rs, y_rs)
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
    
    invert = lambda e,n: utm.to_latlon(e,n,zone,letter)
    
    return new_crs, invert

def bound_from_affine(array, afn):
    left, top = (0,0)*afn
    right, bottom = (array.shape[1], array.shape[0])*afn
    bb = namedtuple('BoundingBox','left right top bottom'.split())
    return bb(left=left, bottom=bottom, right=right, top=top)

def convert_img_to_utm(src_array, src, dst_crs, src_bounds, i=1):
    src=src.copy()
    # Calculate the ideal dimensions and transformation in the new crs
    dst_affine, dst_width, dst_height = calculate_default_transform(
            src['crs'],
            dst_crs,
            src['width'],
            src['height'],
            *src_bounds)
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
    try:
        profile.update({
            'crs': dst_crs,
            'transform': dst_affine,
            'affine': dst_affine,
            'width': dst_width,
            'height': dst_height
        })
    except TypeError:
        profile.update({
            'crs': dst_crs,
            'transform': dst_affine,
            'width': dst_width,
            'height': dst_height
        })
    
    return dst_array, profile

def polgonise(img_mask):
    shp = shapes(img_mask, mask=img_mask.astype(bool), connectivity=4)
    results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(shp)
        )
    geoms = list(results)
    g = gpd.GeoDataFrame.from_features(geoms)
    g = g[g.area>0] #filter out empty shapes
    return g

def clean_poly_list(geoms):
    
    def convert_3d_to_2d(poly):
        if poly.has_z:
            coords = [(x,y) for (x,y,z) in list(poly.boundary.coords)]
            return Polygon(coords)
        else:
            return poly

    def process_poly(poly):
        if poly.geom_type == 'MultiPolygon':
            out = list()
            for p in poly:
                out.extend(process_poly(p))
            return out
        elif poly.geom_type == 'Polygon':
            poly = convert_3d_to_2d(poly)
            return [poly,]
        else:
            raise ValueError('kml file must have Polygon / MultiPolygon features only')

    
    
    out=[]
    for poly in geoms:
        out.extend(process_poly(poly))
    return out

def plot_polys(m):
    for p in m:
        x,y = p.exterior.xy
        plt.plot(x,y)

def check_crs(crs):
    img_crs = crs['init'].split(':')[-1]
    raise_string = '{} - invalid projection for input accepts(epsg:4326, 32600 to 32661, 32700 to 32761)'.format(img_crs)
    try:
        img_crs = int(img_crs)
    except:
        raise ValueError(raise_string)
    if (img_crs >= 32600 and img_crs <= 32661) or (img_crs >= 32700 and img_crs <= 32761):
        #acceptable epsg code range for UTM
        return 'UTM'
    elif img_crs in {4326, 4030} :
        #acceptable epsg code for wgs84
        return 'WGS84'
    else:
        raise ValueError(raise_string)


def smooth_mask(mask_img, mask_img_profile, mask_img_bounds):
    
    #convert input image to UTM (if in WGS84)
    if check_crs(mask_img_profile['crs']) == 'WGS84':
        epsg_n, invert = bound_to_utm(mask_img_bounds)
        utm_crs = crs.CRS.from_epsg(epsg_n)
        mask_img_utm, new_profile = convert_img_to_utm(mask_img, mask_img_profile, utm_crs, mask_img_bounds)
    else:
        epsg_n = mask_img_profile['crs']['init'].split(':')[-1]
        invert = (lambda (x,y): (x,y) )
        mask_img_utm, new_profile = mask_img, mask_img_profile

    #rescale mask to 5m reolution image
    mask_img_utm_res, affine_res = rescale_to_gsd(mask_img_utm, new_profile['affine'], new_gsd=5)
    mask_img_utm_res = mask_img_utm_res.astype(np.uint8)
           
    #turn mask into polygons, then smooth
    gdf = polgonise(mask_img_utm_res).simplify(1)
    gdf = cascaded_union(gdf.geometry)
    gdf_smooth = gdf.buffer(-6).buffer(18).buffer(-15)

    # if polygons remain, create raster image        
    if not gdf_smooth.is_empty:
        
        #if multiploygon feature was created
        if type(gdf_smooth) != Polygon:
            m = clean_poly_list(gdf_smooth)
            poly_list = list((p,255) for p in m)
            coords = [p.exterior.coords for p in gdf_smooth]
        #if single ploygon feature was created
        else:
            poly_list = [(gdf_smooth,255),]
            coords = [gdf_smooth.exterior.coords,]
        
        burned = features.rasterize(shapes=poly_list, out_shape=mask_img_utm_res.shape)
        
        #reproject if src and dst in different crs
        if mask_img_profile['crs']['init'] != crs.CRS.from_epsg(epsg_n)['init']:
            
            # Reproject back to WGS84
            dst_array = np.empty((mask_img_profile['height'], mask_img_profile['width']), dtype='uint8')
            reproject(
                # Source parameters
                source=burned,
                src_crs=crs.CRS.from_epsg(epsg_n),
                src_transform=affine_res,
                # Destination paramaters
                destination=dst_array,
                dst_transform=mask_img_profile['affine'],
                dst_crs=mask_img_profile['crs'], #convert back to input CRS
                )
        
        #convert polygon coordinates, convert to dataframe
        
        poly_out=[]
        for c in coords:
            pts = [(px,py)*affine_res for px,py in c]
            pts = [invert(e,n) for e,n in pts]
            pts = [(py,px) for px,py in pts]
            poly_out.append(Polygon(pts))
        poly_out = gpd.GeoDataFrame({'geometry':poly_out})  
        
    # if all poly zones are too small, returned array and json will be empty
    else:
        dst_array = np.zeros((mask_img_profile['height'], mask_img_profile['width']), dtype='uint8')
        poly_out = gpd.GeoDataFrame({'geometry':[]})
    
    return dst_array, affine_res, poly_out


# ----------------------------------------------------------------------------
# Max Bin Filtering (pre-filter)
# ----------------------------------------------------------------------------

def get_min_max(array_img, array_mask, bins=4):
    #Get histogram thresholds
    masked = np.where(array_mask.astype(bool), array_img, np.nan)
    masked_flat = masked.flatten()
    masked_flat = masked_flat[~np.isnan(masked_flat)]
    cnt, val = np.histogram(masked_flat, bins=bins)
    imax = cnt.argmax()
    min_max = val[imax: imax+2] #upper and lower threshold values
    return min_max

def filter_to_min_max(tif_img, img_mask, min_max):
    #if lower and upper bound given
    if len(min_max)==2:
        l,u = min_max
        f = np.where(
                (tif_img>=l) & (tif_img<u) & (img_mask.astype(bool)),
                tif_img,
                np.nan,
                )
    else:
    #else just upper bound
        assert len(min_max)==1
        l = min_max[0]
        f = np.where(
                (tif_img>=l) & (img_mask.astype(bool)),
                tif_img,
                np.nan,
                )
    return f

def max_bin_filter(tif_img, mask_img, bins=3):
    #do max bin filtering - Remove pixels which do no occur in the max bin
    min_max = get_min_max(tif_img, mask_img, bins=bins)
    f = filter_to_min_max(tif_img, mask_img, min_max)
    return np.where(np.isnan(f),0,255)


# ----------------------------------------------------------------------------
# Main Driver
# ----------------------------------------------------------------------------
def main(
        zone_mask_files = [
             'D:\\test-inputs\\oleksi-issues-normal-points-2\\srcGSD0.0MjenksC4A0.0S0-z00.tif',
             'D:\\test-inputs\\oleksi-issues-normal-points-2\\srcGSD0.0MjenksC4A0.0S0-z01.tif',
             'D:\\test-inputs\\oleksi-issues-normal-points-2\\srcGSD0.0MjenksC4A0.0S0-z02.tif',
             'D:\\test-inputs\\oleksi-issues-normal-points-2\\srcGSD0.0MjenksC4A0.0S0-z03.tif',
             ],
        index_file = 'D:\\test-inputs\\oleksi-issues-normal-points-2\\src.tif',
        max_binning_prefilter = True,
        set_bin_count = 3,
        show_plots = True,
        out_name_ext = r'_smooth',
        ):    
    #open the index image file and extract data
    with rasterio.open(index_file) as src:
        src_profile = src.profile.copy()
        tif_img = src.read(1)
    
    for zone_mask in zone_mask_files:
        
            #open the mask image file and extract data
            with rasterio.open(zone_mask) as src:
                mask_img = src.read(1)
                mask_img_profile = src.profile.copy()
                mask_img_bounds = src.bounds

            #prefilter mask - max binning
            if max_binning_prefilter:
                mask_img_filt = max_bin_filter(tif_img, mask_img, bins=set_bin_count)
            else:
                mask_img_filt = mask_img
        
            # smooth mask image and create json and array
            dst_array, affine_res, poly_out = smooth_mask(mask_img_filt, mask_img_profile, mask_img_bounds)
            
            if show_plots:
                plt.figure()
                plt.subplot(131)
                plt.imshow(mask_img)
                plt.subplot(132)
                plt.imshow(np.where(mask_img_filt.astype(bool),tif_img,np.nan))
                plt.subplot(133)
                plt.imshow(dst_array)
            
            # Write outputs image to disk
            fp, fn = os.path.split(zone_mask)
            fnb, fne = os.path.splitext(fn)
            outFile = os.path.join(fp, fn+out_name_ext+fne)
            src_profile.update({'transform':src_profile['affine']})
            with rasterio.open(outFile, "w", **src_profile) as dest:
                dest.write(dst_array,1)      
            # Write outputs json to disk
            outFile = os.path.join(fp, fn+out_name_ext+'.json')            
            with open(outFile, 'w') as dst:
                dst.write(poly_out.to_json())


# ----------------------------------------------------------------------------
# Run from Command Line
# ----------------------------------------------------------------------------
if __name__ is '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('index_file', help="path to index image, single channel 8bit tif")
    parser.add_argument('zone_mask_files', nargs="+",
                         help="list of tif files as 8bit binary arrays, one per zone")
    parser.add_argument('-max_binning_prefilter', action='store_true',
                        help="perform max binning prefiltering on masks before smoothing")
    parser.add_argument('-set_bin_count', type=int,
                       help="set the number of bins in 'Max Binning Filter'")
    parser.add_argument('-show_plots', action='store_true',
                        help="show output plots")
    #for cmd line
#    args = parser.parse_args()
    
    #for debug
    args = parser.parse_args(r"""
     D:\test-inputs\oleksi-issues-normal-points-2\src.tif
     D:\test-inputs\oleksi-issues-normal-points-2\srcGSD0.0MjenksC4A0.0S0-z00.tif
     D:\test-inputs\oleksi-issues-normal-points-2\srcGSD0.0MjenksC4A0.0S0-z01.tif
     D:\test-inputs\oleksi-issues-normal-points-2\srcGSD0.0MjenksC4A0.0S0-z02.tif
     D:\test-inputs\oleksi-issues-normal-points-2\srcGSD0.0MjenksC4A0.0S0-z03.tif
     -max_binning_prefilter -set_bin_count 3 -show_plots
     """.split())
    
    kwargs = vars(args)
    main(**kwargs)
    







    