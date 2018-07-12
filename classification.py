"""
    Project:
        Classification & Prescription map generation

    Author:
        Tom Lawson (tom@flurosat.com)

    File:
        Classification

    Dependencies:
        numpy
        scipy
        glob
        gdal (http://www.gdal.org/)
        jenkspy (http://github.com/mthh/jenkspy/)
        matplotlib (https://matplotlib.org/)
        skimage (http://scikit-image.org/)
        shapely (http://pypi.python.org/pypi/Shapely)
        pyproj (http://pypi.python.org/pypi/pyproj)

    Notes:
        This file contains the functions necessary for smoothing, classification and export of resulting data.
"""

# External imports
from scipy.misc import imsave
from scipy.misc import imresize
import numpy as np
import numpy.ma as ma
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
from scipy.stats import mode
import jenkspy
import math
from scipy.ndimage import gaussian_filter
from skimage import measure
import matplotlib.cm as colormap
import shapely.wkb
import zipfile
import pyproj
import shapely
import shapely.ops as ops
from functools import partial
import json
import sys
#import cv2


'''
    Classification Pipeline Main Functions
    
    The pipeline works as follows:
    Filter -> Classify -> Export
    
    So, if a new smoothing setting is chosen, the filter function must be called, followed by the classify function
    and the export function.
    
    If a new number of classes was chosen, then the only the classify and export functions need to be called.
'''


# =============== Processing functions =============== #

def filter(in_path, output_path, smoothing, res=None):
    """
        Function that smooths input data and saves it to the hard disk in the form of a geotiff.

        Multiple files can be processed at once by specifying in_path, output_path and smoothing as lists.
        In this case, all lists must have the same length.

        :param in_path: Path to raw grayscale input tiff
        :param output_path: Path to output file
        :param smoothing: Amount of smoothing to use
        :param res: Optional parameter that specifies a resolution to downsample the data to
        :return None
    """


    # Loop through each smoothing parameter given
    for s in range(0, len(smoothing)):

        # If an alternate downsample resolution was chosen, use that; else load in the raw data directly
        if res != None:
            input_ds = downsample(in_path, '', res)
        else:
            input_ds = gdal.Open(in_path)

        # Get raw data as a float array
        raw_data = input_ds.GetRasterBand(1).ReadAsArray().astype(np.float64)

        # Determine background value
        bkg_value = get_bkg_val(raw_data)

        # Pad the raw data with a 1 pixel background boundary to ensure that the background is one contiguous region
        raw_data = np.pad(raw_data,1,mode='constant',constant_values=bkg_value)

        # Get a mask of the background
        bkg_mask = get_bkg_mask(raw_data)

        # Set the isolated background areas located on the field to 0
        raw_data[raw_data == bkg_value] = np.nan

        # Suppress warnings due to dividing by 0
        np.seterr(divide='ignore', invalid='ignore')

        # Apply gaussian smoothing
        if smoothing[s] > 0:

            U = raw_data

            V = U.copy()
            V[U != U] = 0
            VV = gaussian_filter(V, smoothing[s])

            W = 0 * U.copy() + 1
            W[U != U] = 0
            WW = gaussian_filter(W, smoothing[s])

            out_data = VV / WW

        else:
            out_data = raw_data

        # Apply the NaNs to the background only
        out_data[np.argwhere(np.isnan(bkg_mask))[:,0],np.argwhere(np.isnan(bkg_mask))[:,1]] = np.nan

        # Remove the pixel padding that was added previously
        out_data = out_data[1:-1, 1:-1]

        # Get geotransform of input data
        geo_trans = input_ds.GetGeoTransform()

        # Get projection of input data
        prj = input_ds.GetProjection()

        # Save smoothed data as geotiff
        driver_path = output_path

        print (driver_path)

        driver_name = 'GTiff'

        driver = gdal.GetDriverByName(driver_name)
        out_ds = driver.Create(driver_path, out_data.shape[1], out_data.shape[0], 1, gdal.GDT_Float64)

        # Apply the geotransform and projection to the new dataset
        out_ds.SetGeoTransform(geo_trans)
        out_ds.SetProjection(prj)

        # Write output data to the array
        out_ds.GetRasterBand(1).WriteArray(out_data)
        out_ds.FlushCache()  # Write to disk

def classify(in_path, output_path, method, n_classes, class_breaks=None):
    """
        Function that classifies data outputted by the smoothing function, into n classes.

        The result will be saved to disk as Python pickle files which can then be passed to the relevant by the
        export functions (see below).

        Class breaks can also be specified manually by choosing method='manual' and an array of class breaks in
        class_breaks.

        :param in_path: Path to the output file generated by the filter() function
        :param output_path:
        :param method: Classification method name; possible choices are:
                       - "uniqueValue"
                       - "equalInterval"
                       - "bestFitJenks"
                       - "equalValue"
                       - "equalRecords"
                       - "stddev"
                       - "manual"
        :param n_classes: Number of classes to classify the input image into, for segmentation method manual
        :param class_breaks: Class breaks specified in array format if the
        :return class_data, class_avg, class_bounds ; if a blank output path is specified
    """

    # Load the filtered dataset
    input_ds = gdal.Open(in_path)

    # Get data as array
    filtered_data = input_ds.GetRasterBand(1).ReadAsArray()

    # Loop through each number of classes given
    for c in range(0, len(n_classes)):

        # If class breaks were specified manually
        if class_breaks is not None:
            class_data, class_avg, class_bounds = segment_manual(filtered_data, class_breaks)

        # If a standard classification method was chosen
        else:

            # Perform classification depending on the method chosen
            methods_dict = {"uniqueValue": segment_uniqueValue,
                            "equalInterval": segment_equalInterval,
                            "bestFitJenks": segment_bestFitJenks,
                            "equalValue": segment_equalValue,
                            "equalRecords": segment_equalRecords,
                            "stddev": segment_stddev}

            if method in methods_dict:
                class_data, class_avg, class_bounds = methods_dict[method](filtered_data, n_classes[c])

            else:
                print("Error: Segmentation method " + method + " is in invalid.")
                return

        # If an output path was specified save the results of the classification as a Python pickle file
        if output_path is not '':
            collated_list = [class_data, class_avg, class_bounds]
            np.save(output_path, collated_list, allow_pickle=True)

        # Else return the results of the classification directly
        else:
            return class_data, class_avg, class_bounds


# =============== Query functions =============== #

def get_class_areas(filtered_path, class_path, percentage=True):
    """
        Function that returns the area of each class, either in percentage or in Ha.
        Areas are returned in the following format:
        [ [class ID, area], [class ID, area], ...]

        :param filtered_path: Path of the geotiff that was outputted by the filter() function
        :param class_path: Path to the python pickle containing the class data
        :param percentage: If true, return field areas in %, else return field areas in Ha.
        :return: A vector of areas returned in the following format: [ [class ID, area], [class ID, area], ...]
    """

    # Load the classification data from file
    collated_list = np.load(class_path, allow_pickle=True)

    class_data = collated_list[0]
    class_avg = collated_list[1]

    # Determine number of classes
    n_classes = len(class_avg)

    # Create a driver to create a gdal dataset (in memory)
    gdal_driver = gdal.GetDriverByName('MEM')

    # Create a gdal dataset in memory with one raster layer to store the input data (input data type is Uint8)
    raster_ds = gdal_driver.Create('', class_data.shape[1], class_data.shape[0], 1, gdal.GDT_Byte)

    # Load source dataset
    input_ds = gdal.Open(filtered_path)

    # Get projection from the initial dataset
    prj = input_ds.GetProjection()

    # Input SRS
    inSpatialRef = osr.SpatialReference(wkt=prj)

    # Copy geotransform info to the raster
    raster_ds.SetGeoTransform(input_ds.GetGeoTransform())

    # Apply input SRS to the raster
    raster_ds.SetProjection(inSpatialRef.ExportToWkt())

    # Projection to which to reproject geometry. Needs to be WGS84 (required)
    EPSG = 4326

    # Output SRS, will be different if an alternate output projection was specified
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(EPSG)
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)  # Create the CoordinateTransformation

    # Get the first raster band
    raster_band = raster_ds.GetRasterBand(1)

    # Set the no-data-value to the correct bkg label value so that the background can be masked out for polygonisation
    raster_band.SetNoDataValue(-1)

    # Write the data to the raster
    raster_band.WriteArray(class_data)

    # Create a Vector Layer in memory (OGR requires us to create a layer even though shapefiles have only one layer)
    ogr_driver = ogr.GetDriverByName('Memory')

    # Create the OGR data source that the shapefile data will be stored in while it is in memory
    vector_src = ogr_driver.CreateDataSource('')

    # Create a new layer to store the polygons we are about to create from the class edge contours
    polygon_layer = vector_src.CreateLayer('polygon', inSpatialRef, ogr.wkbMultiPolygon)

    # Create a field to store the class numbers
    class_fieldName = 'class'
    field_classes = ogr.FieldDefn(class_fieldName, ogr.OFTInteger)

    # Add the field to the polygon layer
    polygon_layer.CreateField(field_classes)

    # Create vector polygons from the contours of all connected regions of common class pixels in the raster
    # Note that GetMaskBand() returns the raster as an array with the background masked out
    gdal.Polygonize(raster_band, raster_band.GetMaskBand(), polygon_layer, 0)

    # Vector of class areas
    class_areas = []

    # Loop through each class
    for i in range(1, n_classes + 1):

        # Select all features in polygon layer with class i
        polygon_layer.SetAttributeFilter('%s = %s' % (class_fieldName, str(i)))

        # Area of current class in Ha
        cumul_area = 0.0

        for feature in polygon_layer:

            # Get geometry of current feature
            feat_geom = feature.geometry()

            # If there is geometry associated with the current feature, reproject it in WGS84
            if feat_geom is not None:

                # Reproject the geometry
                feat_geom.Transform(coordTrans)

                # Convert to shapely polygon
                geom = shapely.wkb.loads(feat_geom.ExportToWkb())

                # Check if the geometry self-intersects and if so clean it up to remove the self-intersection
                if not geom.is_valid:
                    geom = geom.buffer(0)  # Pass 0 distance to buffer to clean self-intersecting polygons

                # Compute area of current polygon by reprojecting to equal area projection
                geom_area = ops.transform( partial( pyproj.transform,
                                                    pyproj.Proj(init='EPSG:4326'),
                                                    pyproj.Proj( proj='aea', lat1=geom.bounds[1], lat2=geom.bounds[3] )),
                                                    geom )

                # Add to cumulative area total, in Ha
                cumul_area += geom_area.area*1e-4

        # Add to list of class areas
        class_areas.append( cumul_area )

    # Prepare class areas in the format [class ID, area]
    out_fmt = np.ones((n_classes, 2))

    # Prepare class areas in the format [class ID, area]
    for i in range(0,n_classes):
        out_fmt[i, 0] = i+1 # class ID

        # If the percentage parameter is True, return 0
        if percentage:
            out_fmt[i, 1] = round(class_areas[i]/sum(class_areas) * 100, 2)

        else:
            out_fmt[i, 1] = round(class_areas[i], 2)

    return out_fmt

def get_class_bounds(class_path, output_path):
    """
        Function that returns the class bounds and averages in the following format:
        [min, mean, max]

        :param class_path: Path to the python pickle containing the class data
        :return: Returns information in the following format: [[min1, mean1, max1], [min2, mean2, max2], ...]
    """

    # Load the classification data from file
    collated_list = np.load(class_path, allow_pickle=True)

    class_avg = collated_list[1]
    class_bounds = collated_list[2]

    print (collated_list[0])
    print ("\n")
    print (class_bounds)


    print ("here")
    sys.exit()

    # Determine number of classes
    n_classes = len(class_avg)

    # Vector to store output information
    out_fmt = np.ones((n_classes,3))

    out_fmt[:,0] = class_bounds[:,0] # min
    out_fmt[:,1] = class_avg # mean
    out_fmt[:,2] = class_bounds[:,1] # max

    # --

    cmap = colormap.get_cmap('viridis')
    min_avg = class_avg.min()
    range_avg = class_avg.ptp()
    class_avg_norm = (class_avg-min_avg)/range_avg

    colours = np.round( cmap(class_avg_norm)*255 ).astype(np.uint8)
    colours = np.uint8(colours)

    result = []
    for n in range(len(colours)):
        result.append( (colours[n].tolist(), out_fmt[n].tolist()) )

    with open(output_path, 'w') as out_file:
        json.dump(result, out_file, indent=2)

    return out_fmt

# =============== Export functions =============== #

def export_as_shapefile(filtered_path, class_path, output_path, EPSG=None):
    """
        Function that exports the segmented data as a shapefile.
        If an EPSG is specified as an additional parameter, the ouput shapefile will be reprojected.

        filtered_path specifies the path of the filtered data generated by the filter function
        class_path specifies the path of the pickle file generated by the classify function.

        :param filtered_path: Path to the image generated by the filter() function
        :param class_path: Path to the pickle (.npy) file generated by the classify() function
        :param output_path: Path to the output file
        :param EPSG: EPSG code of the projection for which the output file will be reprojected to
        :return None
    """

    # Load the classification data from file
    collated_list = np.load(class_path, allow_pickle=True)

    class_data = collated_list[0]
    class_avg = collated_list[1]
    class_bounds = collated_list[2]

    # Determine number of classes
    n_classes = len(class_avg)

    # Load source dataset
    input_ds = gdal.Open(filtered_path)

    # Create a driver to create a gdal dataset (in memory)
    gdal_driver = gdal.GetDriverByName('MEM')

    # Create a gdal dataset in memory with one raster layer to store the input data (input data type is Uint8)
    raster_ds = gdal_driver.Create('', class_data.shape[1], class_data.shape[0], 1, gdal.GDT_Byte)

    # Get projection from the initial dataset
    prj = input_ds.GetProjection()

    # Input SRS
    inSpatialRef = osr.SpatialReference(wkt=prj)

    # Copy geotransform info to the raster
    raster_ds.SetGeoTransform(input_ds.GetGeoTransform())

    # Apply input SRS to the raster
    raster_ds.SetProjection(inSpatialRef.ExportToWkt())

    # Output SRS, will be different if an alternate output projection was specified
    if EPSG is not None:
        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromEPSG(EPSG)
        coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)  # Create the CoordinateTransformation
    else:
        outSpatialRef = inSpatialRef

    # Create instance of driver for the shapefile format
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # Get the first raster band
    raster_band = raster_ds.GetRasterBand(1)

    # Set the no-data-value to the correct bkg label value so that the background can be masked out when generating the polygons
    raster_band.SetNoDataValue(-1)

    # Write the data to the raster
    raster_band.WriteArray(class_data)

    # Create a Vector Layer in memory (OGR requires us to create a layer even though shapefiles have only one layer)
    ogr_driver = ogr.GetDriverByName('Memory')

    # Create the OGR data source that the shapefile data will be stored in while it is in memory
    vector_src = ogr_driver.CreateDataSource(output_path)

    # Create a new layer to store the polygons we are about to create from the class edge contours
    polygon_layer = vector_src.CreateLayer('polygon', inSpatialRef, ogr.wkbMultiPolygon)

    # Create a field to store the class numbers
    class_fieldName = 'class'
    field_classes = ogr.FieldDefn(class_fieldName, ogr.OFTInteger)

    # Add the field to the polygon layer
    polygon_layer.CreateField(field_classes)

    # Create vector polygons from the contours of all connected regions of common class pixels in the raster
    # Note that GetMaskBand() returns the raster as an array with the background masked out
    gdal.Polygonize(raster_band, raster_band.GetMaskBand(), polygon_layer, 0)

    # Create the OGR data source that the shapefile data will be stored in while it is in memory
    output_src = driver.CreateDataSource(output_path)

    # Create the layer in the output data source
    output_layer = output_src.CreateLayer('classes', outSpatialRef, geom_type=ogr.wkbMultiPolygon)

    # Create a field for the classes in the new output layer, and set the field values to be strings
    output_layer.CreateField(ogr.FieldDefn(class_fieldName, ogr.OFTString))

    # Loop through each class
    for i in range(1, n_classes + 1):

        # Select the features in the polygon layer with class == i
        polygon_layer.SetAttributeFilter('%s = %s' % (class_fieldName, str(i)))

        # Generate a feature (contains both attributes and geometry)
        multi_feature = ogr.Feature(output_layer.GetLayerDefn())

        # Generate a polygon based on the geometry defined in the layer
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)

        for feature in polygon_layer:
            # Clean the polygon geometry to remove self-intersections
            geom = shapely.wkb.loads(feature.geometry().ExportToWkb())  # Convert to shapely polygon
            clean = geom.buffer(0)  # Pass 0 distance to buffer to clean self-intersecting polygons
            if clean.is_valid == False:
                print('WARNING: Geometry is invalid')

            geom = ogr.CreateGeometryFromWkb(clean.wkb)  # Convert back to OGR geometry

            #geom = feature.geometry()

            # Aggregate all the input geometry sharing the class value i
            multipolygon.AddGeometry(geom)

        # Add the merged geometry to the current feature
        multi_feature.SetGeometry(multipolygon)

        # Reproject, if an alternate EPSG was specified
        if EPSG is not None:
            # Get the input geometry
            geom = multi_feature.GetGeometryRef()

            # If there is geometry associated with the current feature, reproject it in the new projection
            if geom is not None:
                # Reproject the geometry
                geom.Transform(coordTrans)

            # Set the geometry and attribute
            multi_feature.SetGeometry(geom)

        # Calculate the current class bounds
        class_lower = class_bounds[i - 1, 0]
        class_upper = class_bounds[i - 1, 1]

        # Set the field of the current feature
        if class_lower == class_upper:  # if min = max, then don't display a range in the legend
            multi_feature.SetField(class_fieldName, str(round(class_lower, 3)))

        else:  # else display the full range of values in the class
            multi_feature.SetField(class_fieldName, str(round(class_lower, 3)) + ' - ' + str(round(class_upper, 3)))

        # Add the current feature to the layer
        output_layer.CreateFeature(multi_feature)

        # Destroy the current feature
        multi_feature.Destroy()

def export_as_image(filtered_path, class_path, output_path, x_size=None, y_size=None, cmap_name='viridis'):
    """
        Function that exports the segmentated data as a non-georeferenced image using an arbitrary colour scheme.
        The output format is determined by the file extension specified in the output path

        filtered_path specifies the path of the filtered data generated by the filter function
        class_path specifies the path of the pickle file generated by the classify function.

        :param filtered_path: Path to the image generated by the filter() function
        :param class_path: Path to the pickle (.npy) file generated by the classify() function
        :param output_path: Path to the output file
        :param EPSG: EPSG code of the projection for which the output file will be reprojected to
        :param x_size: Output width in pixels of the output image
        :param y_size: Output height in pixels of the output image
        :param cmap_name: Name of the colormap
        :return None
    """

    # ---------- Load & Normalise class data + generate colours ---------- #

    # Load the classification data from file
    collated_list = np.load(class_path, allow_pickle=True) # Open python pickle file
    class_data = collated_list[0] # Extract class data from collated array
    class_avg = collated_list[1] # Extract class averages from collated array

    # If an alternative output size was specified, then upsample it to the correct size
    # We don't need to worry about what happens to the georeferencing when we scale the pixels since the output
    # won't be georeferenced.
    if x_size is not None and y_size is not None:
        class_data = upsample(class_data, x_size, y_size)

    # Create a color map to use using the Red-Yellow-Green color scale
    cmap = colormap.get_cmap(cmap_name)

    # Normalise the class averages to between 0 and 1 so that the colormap can be applied to it
    min_avg = class_avg.min()
    range_avg = class_avg.ptp()
    class_avg_norm = (class_avg-min_avg)/range_avg

    # Generate array of colors using the color scale and transparent background
    color_bkg = np.array([[255,255,255,0]],dtype=np.uint8) # Add transparent background as the first colour
    color_scale = np.round( cmap(class_avg_norm)*255 ).astype(np.uint8) # Create an array of class colours in RGBA with 8-bit color
    colours = np.vstack( (color_bkg, color_scale) ) # Concatenate colors

    # ---------- Map indices to colours and export as image ---------- #

    # Convert to uint8 to display the image as RGB
    colours = np.uint8(colours)

    # Change the background to 0
    img_indices = class_data.flatten()
    img_indices[img_indices == -1] = 0

    # Append the actual colours to each corresponding label
    img = colours[img_indices]

    # Reshape the array back into RGBA image form (4 for R,G,B,A values)
    img = img.reshape(class_data.shape[0], class_data.shape[1], 4)

    # Save the image data to the output path
    imsave(output_path, img)

def export_as_kml(filtered_path, class_path, output_path, EPSG=None):
    """
        Function that exports the segmented data as a kml.
        If an EPSG is specified as an additional parameter, the output shapefile will be reprojected.

        filtered_path specifies the path of the filtered data generated by the filter function
        class_path specifies the path of the pickle file generated by the classify function.

        :param filtered_path: Path to the image generated by the filter() function
        :param class_path: Path to the pickle (.npy) file generated by the classify() function
        :param output_path: Path to the output file
        :param EPSG: EPSG code of the projection for which the output file will be reprojected to
        :return None
    """

    # ---------- Load & Normalise class data + generate colours ---------- #

    # Load the classification data from file
    collated_list = np.load(class_path, allow_pickle=True)

    class_data = collated_list[0]
    class_avg = collated_list[1]
    class_bounds = collated_list[2]

    # Determine number of classes
    n_classes = len(class_avg)

    # Now that we have the class data, match each value to its colour using a colourmap
    cmap = colormap.get_cmap('viridis')

    # Normalise the class averages to between 0 and 1 so that the colormap can be applied to it
    min_avg = class_avg.min()
    range_avg = class_avg.ptp()

    class_avg_norm = (class_avg-min_avg)/range_avg

    # Create an array of class colours
    class_colours = cmap(class_avg_norm)

    # ---------- Generate Geometry and export KML ---------- #

    # Load source dataset
    input_ds = gdal.Open(filtered_path)

    # Create a driver to create a gdal dataset (in memory)
    gdal_driver = gdal.GetDriverByName('MEM')

    # Create a gdal dataset in memory with one raster layer to store the input data (input data type is Uint8)
    raster_ds = gdal_driver.Create('', class_data.shape[1], class_data.shape[0], 1, gdal.GDT_Byte)

    # Get projection from the initial dataset
    prj = input_ds.GetProjection()

    # Input SRS
    inSpatialRef = osr.SpatialReference(wkt=prj)

    # Copy geotransform info to the raster
    raster_ds.SetGeoTransform(input_ds.GetGeoTransform())

    # Apply input SRS to the raster
    raster_ds.SetProjection(inSpatialRef.ExportToWkt())

    # Output SRS, will be different if an alternate output projection was specified
    if EPSG is not None:
        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromEPSG(EPSG)
        coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)  # Create the CoordinateTransformation
    else:
        outSpatialRef = inSpatialRef

    # Create instance of driver for the shapefile format
    driver = ogr.GetDriverByName('KML')

    # Get the first raster band
    raster_band = raster_ds.GetRasterBand(1)

    # Set the no-data-value to the correct bkg label value so that the background can be masked out when generating the polygons
    raster_band.SetNoDataValue(-1)

    # Write the data to the raster
    raster_band.WriteArray(class_data)

    # Create a Vector Layer in memory (OGR requires us to create a layer even though shapefiles have only one layer)
    ogr_driver = ogr.GetDriverByName('Memory')

    # Create the OGR data source that the shapefile data will be stored in while it is in memory
    vector_src = ogr_driver.CreateDataSource(output_path)

    # Create a new layer to store the polygons we are about to create from the class edge contours
    polygon_layer = vector_src.CreateLayer('polygon', inSpatialRef, ogr.wkbMultiPolygon)

    # Create a field to store the class numbers
    class_fieldName = 'class'
    field_classes = ogr.FieldDefn(class_fieldName, ogr.OFTInteger)

    # Add the field to the polygon layer
    polygon_layer.CreateField(field_classes)

    # Create vector polygons from the contours of all connected regions of common class pixels in the raster
    # Note that GetMaskBand() returns the raster as an array with the background masked out
    gdal.Polygonize(raster_band, raster_band.GetMaskBand(), polygon_layer, 0, options=['8CONNECTED=1'])

    # Create the OGR data source that the shapefile data will be stored in while it is in memory
    output_src = driver.CreateDataSource(output_path)

    # Create the layer in the output data source
    output_layer = output_src.CreateLayer('classes', outSpatialRef, geom_type=ogr.wkbMultiPolygon)

    # Create a field for the classes in the new output layer, and set the field values to be strings
    output_layer.CreateField(ogr.FieldDefn(class_fieldName, ogr.OFTString))

    # Loop through each class
    for i in range(1, n_classes + 1):

        # Select the features in the polygon layer with class == i
        polygon_layer.SetAttributeFilter('%s = %s' % (class_fieldName, str(i)))

        # Generate a feature (contains both attributes and geometry)
        multi_feature = ogr.Feature(output_layer.GetLayerDefn())

        # Generate a polygon based on the geometry defined in the layer
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)

        for feature in polygon_layer:
            # Aggregate all the input geometry sharing the class value i
            multipolygon.AddGeometry(feature.geometry())

        # Add the merged geometry to the current feature
        multi_feature.SetGeometry(multipolygon)

        # Reproject, if an alternate EPSG was specified
        if EPSG is not None:
            # Get the input geometry
            geom = multi_feature.GetGeometryRef()

            # If there is geometry associated with the current feature, reproject it in the new projection
            if geom is not None:
                # Reproject the geometry
                geom.Transform(coordTrans)

            # Set the geometry and attribute
            multi_feature.SetGeometry(geom)

        # Calculate the current class bounds
        class_lower = class_bounds[i - 1, 0]
        class_upper = class_bounds[i - 1, 1]

        # Set the field of the current feature
        if class_lower == class_upper:  # if min = max, then don't display a range in the legend
            multi_feature.SetField(class_fieldName, str(round(class_lower, 3)))

        else:  # else display the full range of values in the class
            multi_feature.SetField(class_fieldName, str(round(class_lower, 3)) + ' - ' + str(round(class_upper, 3)))

        # Add the current feature to the layer
        output_layer.CreateFeature(multi_feature)

        # Destroy the current feature
        multi_feature.Destroy()

    # Remove file locks so we can access the KML
    del output_src

    return

def export_as_kmz(filtered_path, class_path, output_path, EPSG=None):
    """
        THIS FUNCTION IS EXPERIMENTAL.

        Function that exports the segmented data as a kmz.
        If an EPSG is specified as an additional parameter, the output shapefile will be reprojected.

        filtered_path specifies the path of the filtered data generated by the filter function
        class_path specifies the path of the pickle file generated by the classify function.

        :param filtered_path: Path to the image generated by the filter() function
        :param class_path: Path to the pickle (.npy) file generated by the classify() function
        :param output_path: Path to the output file
        :param EPSG: EPSG code of the projection for which the output file will be reprojected to
        :return None
    """

    # Print warning about experimental function
    print('WARNING: export_as_kmz is an experimental function. It is not suitable for implementation on a non-dev platform.')

    # ---------- Load & Normalise class data + generate colours ---------- #

    # Load the classification data from file
    collated_list = np.load(class_path, allow_pickle=True)

    class_data = collated_list[0]
    class_avg = collated_list[1]
    class_bounds = collated_list[2]

    # Determine number of classes
    n_classes = len(class_avg)

    # Now that we have the class data, match each value to its colour using a colourmap
    cmap = colormap.get_cmap('viridis')

    # Normalise the class averages to between 0 and 1 so that the colormap can be applied to it
    min_avg = class_avg.min()
    range_avg = class_avg.ptp()

    class_avg_norm = (class_avg-min_avg)/range_avg

    # Create an array of class colours
    class_colours = cmap(class_avg_norm)

    # ---------- Generate Geometry and export KML ---------- #

    # Load source dataset
    input_ds = gdal.Open(filtered_path)

    # Create a driver to create a gdal dataset (in memory)
    gdal_driver = gdal.GetDriverByName('MEM')

    # Create a gdal dataset in memory with one raster layer to store the input data (input data type is Uint8)
    raster_ds = gdal_driver.Create('', class_data.shape[1], class_data.shape[0], 1, gdal.GDT_Byte)

    # Get projection from the initial dataset
    prj = input_ds.GetProjection()

    # Input SRS
    inSpatialRef = osr.SpatialReference(wkt=prj)

    # Copy geotransform info to the raster
    raster_ds.SetGeoTransform(input_ds.GetGeoTransform())

    # Apply input SRS to the raster
    raster_ds.SetProjection(inSpatialRef.ExportToWkt())

    # Output SRS, will be different if an alternate output projection was specified
    if EPSG is not None:
        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromEPSG(EPSG)
        coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)  # Create the CoordinateTransformation
    else:
        outSpatialRef = inSpatialRef

    # Create instance of driver for the shapefile format
    driver = ogr.GetDriverByName('LIBKML')

    # Get the first raster band
    raster_band = raster_ds.GetRasterBand(1)

    # Set the no-data-value to the correct bkg label value so that the background can be masked out when generating the polygons
    raster_band.SetNoDataValue(-1)

    # Write the data to the raster
    raster_band.WriteArray(class_data)

    # Create a Vector Layer in memory (OGR requires us to create a layer even though shapefiles have only one layer)
    ogr_driver = ogr.GetDriverByName('Memory')

    # Create the OGR data source that the shapefile data will be stored in while it is in memory
    vector_src = ogr_driver.CreateDataSource(output_path)

    # Create a new layer to store the polygons we are about to create from the class edge contours
    polygon_layer = vector_src.CreateLayer('polygon', inSpatialRef, ogr.wkbMultiPolygon)

    # Create a field to store the class numbers
    class_fieldName = 'class'
    field_classes = ogr.FieldDefn(class_fieldName, ogr.OFTInteger)

    # Add the field to the polygon layer
    polygon_layer.CreateField(field_classes)

    # Create vector polygons from the contours of all connected regions of common class pixels in the raster
    # Note that GetMaskBand() returns the raster as an array with the background masked out
    gdal.Polygonize(raster_band, raster_band.GetMaskBand(), polygon_layer, 0, options=['8CONNECTED=1'])

    # Create the OGR data source that the shapefile data will be stored in while it is in memory
    output_src = driver.CreateDataSource(output_path)

    # Create the layer in the output data source
    output_layer = output_src.CreateLayer('classes', outSpatialRef, geom_type=ogr.wkbMultiPolygon)

    # Create a field for the classes in the new output layer, and set the field values to be strings
    output_layer.CreateField(ogr.FieldDefn(class_fieldName, ogr.OFTString))

    # Loop through each class
    for i in range(1, n_classes + 1):

        # Select the features in the polygon layer with class == i
        polygon_layer.SetAttributeFilter('%s = %s' % (class_fieldName, str(i)))

        # Generate a feature (contains both attributes and geometry)
        multi_feature = ogr.Feature(output_layer.GetLayerDefn())

        # Generate a polygon based on the geometry defined in the layer
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)

        for feature in polygon_layer:
            # Aggregate all the input geometry sharing the class value i
            multipolygon.AddGeometry(feature.geometry())

        # Add the merged geometry to the current feature
        multi_feature.SetGeometry(multipolygon)

        # Reproject, if an alternate EPSG was specified
        if EPSG is not None:
            # Get the input geometry
            geom = multi_feature.GetGeometryRef()

            # If there is geometry associated with the current feature, reproject it in the new projection
            if geom is not None:
                # Reproject the geometry
                geom.Transform(coordTrans)

            # Set the geometry and attribute
            multi_feature.SetGeometry(geom)

        # Calculate the current class bounds
        class_lower = class_bounds[i - 1, 0]
        class_upper = class_bounds[i - 1, 1]

        # Set the field of the current feature
        if class_lower == class_upper:  # if min = max, then don't display a range in the legend
            multi_feature.SetField(class_fieldName, str(round(class_lower, 3)))

        else:  # else display the full range of values in the class
            multi_feature.SetField(class_fieldName, str(round(class_lower, 3)) + ' - ' + str(round(class_upper, 3)))

        # Add the current feature to the layer
        output_layer.CreateFeature(multi_feature)

        # Destroy the current feature
        multi_feature.Destroy()

    # Remove file locks so we can access the KML
    del output_src



    # Unzip the KML from the KMZ










    return

def export_as_gtiff(filtered_path, class_path, output_path, EPSG=None, cmap_name='viridis'):
    """
        Function that saves the prescription maps (obtained by filtering and classifying the data) as a GeoTiff.

        :param filtered_path: Path of the geotiff that was outputted by the filter() function
        :param class_path: Path to the python pickle containing the class data
        :param output_path: Output path of the geojpeg
        :param EPSG: Optional parameter; you can include an alternate projection from EPSG
        :param cmap_name: Name of colormap to use for pseudocolor
        :return:
    """

    # ---------- Load & Normalise class data + generate colours ---------- #

    # Load the classification data from file
    collated_list = np.load(class_path, allow_pickle=True) # Open python pickle file
    class_data = collated_list[0] # Extract class data from collated array
    class_avg = collated_list[1] # Extract class averages from collated array

    # Create a color map to use using the Red-Yellow-Green color scale
    cmap = colormap.get_cmap(cmap_name)

    # Normalise the class averages to between 0 and 1 so that the colormap can be applied to it
    min_avg = class_avg.min()
    range_avg = class_avg.ptp()
    class_avg_norm = (class_avg-min_avg)/range_avg

    # Generate array of colors using the color scale and transparent background
    color_bkg = np.array([[255,255,255,0]],dtype=np.uint8) # Add transparent background as the first colour
    color_scale = np.round( cmap(class_avg_norm)*255 ).astype(np.uint8) # Create an array of class colours in RGBA with 8-bit color
    class_colors = np.vstack( (color_bkg, color_scale) ) # Concatenate colors

    # ---------- Generate the colored pixel array ---------- #

    # Change the background to 0
    img_indices = class_data.flatten()
    img_indices[img_indices == -1] = 0

    # Append the actual colours to each corresponding label
    img = class_colors[img_indices]

    # Reshape the array back into RGB image form (4 for R,G,B,A values)
    img = img.reshape(class_data.shape[0], class_data.shape[1], 4)

    # Create a GDAL geo jpeg driver
    driver = gdal.GetDriverByName('MEM')

    # Create the destination data source
    dest_ds = driver.Create( '', class_data.shape[1], class_data.shape[0], gdal.GDT_Byte )

    # --------- Apply projection ---------- #

    # Load-in the filtered dataset to obtain projection info
    input_ds = gdal.Open(filtered_path)

    # Get projection from the initial dataset
    prj = input_ds.GetProjection()

    # Input SRS
    inSpatialRef = osr.SpatialReference(wkt=prj)

    # Output SRS, will be different if an alternate output projection was specified
    if EPSG is not None:
        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromEPSG(EPSG)
    else:
        outSpatialRef = inSpatialRef

    dest_ds.SetGeoTransform(input_ds.GetGeoTransform())  # Copy geotransform info to the dataset
    dest_ds.SetProjection(inSpatialRef.ExportToWkt())  # Apply input SRS to the dataset

    # Go through each raster band and write data + apply projection
    for b in range(0,img.shape[2]):
        dest_ds.GetRasterBand(b+1).WriteArray(img[:,:,b]) # Write data

        if b < img.shape[2]-1:
            dest_ds.AddBand()

    gdal.Warp(output_path, dest_ds, format='GTiff', dstSRS = outSpatialRef, creationOptions = ["PHOTOMETRIC=RGB","ALPHA=YES"], outputType = gdal.GDT_Byte)

    return

def export_as_gjpeg(filtered_path, class_path, output_path, EPSG=None, cmap_name='viridis'):
    """
        Function that saves the prescription maps (obtained by filtering and classifying the data) as a rectified JPEG.
        The pixel data is stored in a .jpg file, and the georeference info is stored in both a world file (.wld) and
        a .aux.xml file (only one is required to specify the projection).

        :param filtered_path: Path of the geotiff that was outputted by the filter() function
        :param class_path: Path to the python pickle containing the class data
        :param output_path: Output path of the geojpeg
        :param EPSG: Optional parameter; you can include an alternate projection from EPSG
        :param cmap_name: Name of colormap to use for pseudocolor
        :return:
    """

    # ---------- Load & Normalise class data + generate colours ---------- #

    # Load the classification data from file
    collated_list = np.load(class_path, allow_pickle=True) # Open python pickle file
    class_data = collated_list[0] # Extract class data from collated array
    class_avg = collated_list[1] # Extract class averages from collated array

    # Create a color map to use using the Red-Yellow-Green color scale
    cmap = colormap.get_cmap(cmap_name)

    # Normalise the class averages to between 0 and 1 so that the colormap can be applied to it
    min_avg = class_avg.min()
    range_avg = class_avg.ptp()
    class_avg_norm = (class_avg-min_avg)/range_avg

    # Generate array of colors using the color scale and transparent background
    color_bkg = np.array([[0,0,0,0]],dtype=np.uint8) # Add transparent background as the first colour
    color_scale = np.round( cmap(class_avg_norm)*255 ).astype(np.uint8) # Create an array of class colours in RGBA with 8-bit color
    class_colors = np.vstack( (color_bkg, color_scale) ) # Concatenate colors

    # ---------- Generate the colored pixel array ---------- #

    # Change the background to 0
    img_indices = class_data.flatten()
    img_indices[img_indices == -1] = 0

    # Append the actual colours to each corresponding label
    img = class_colors[img_indices]

    # Reshape the array back into RGB image form (4 for R,G,B,A values)
    img = img.reshape(class_data.shape[0], class_data.shape[1], 4)

    # Create a GDAL geo jpeg driver
    driver = gdal.GetDriverByName('MEM')

    # Create the destination data source
    dest_ds = driver.Create( '', class_data.shape[1], class_data.shape[0], gdal.GDT_Byte )

    # --------- Apply projection ---------- #

    # Load-in the filtered dataset to obtain projection info
    input_ds = gdal.Open(filtered_path)

    # Get projection from the initial dataset
    prj = input_ds.GetProjection()

    # Input SRS
    inSpatialRef = osr.SpatialReference(wkt=prj)

    # Output SRS, will be different if an alternate output projection was specified
    if EPSG is not None:
        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromEPSG(EPSG)
    else:
        outSpatialRef = inSpatialRef

    dest_ds.SetGeoTransform(input_ds.GetGeoTransform())  # Copy geotransform info to the dataset
    dest_ds.SetProjection(inSpatialRef.ExportToWkt())  # Apply input SRS to the dataset

    # Go through each raster band and write data + apply projection
    for b in range(0,img.shape[2]):
        dest_ds.GetRasterBand(b+1).WriteArray(img[:,:,b]) # Write data

        if b < img.shape[2]-1:
            dest_ds.AddBand()


    # Reproject the dataset if necessary
    if EPSG is not None:
        reprj_ds = gdal.AutoCreateWarpedVRT(dest_ds, prj, outSpatialRef.ExportToWkt())
    else:
        reprj_ds = dest_ds

    # Translate the in-memory dataset to JPEG format
    gdal.Translate(output_path, reprj_ds, format='JPEG', creationOptions=["WORLDFILE=YES"])

    return


# =============== Segmentation methods =============== #

def segment_uniqueValue(in_data, n_classes):
    """
        Method of segmentation where each unique value has its own class in the legend.
        Suitable for a soil type or crop type map, but not for raw data.
        Returns the class data (where each pixel has the value of its class), class averages and class boundaries.

        :param in_data: 2d array of input data values
        :param n_classes: Number of classes to segment data into
        :return in order: class_data, class_avg, class_bounds, n_classes
    """

    # Get unique values in the array, which will be the class 'averages'
    class_avg = np.unique(in_data)

    # Remove the background from the set of unique values
    bkg_value = -1
    class_avg = class_avg[class_avg != bkg_value]

    # Pre-allocate class data
    class_data = np.zeros(in_data.shape, dtype=np.int64)

    # Pre-allocate class bounds
    class_bounds = np.zeros((class_avg.shape[0], 2), dtype=np.float32)

    # Loop through each class
    k = 0  # current class
    for c in class_avg:
        # Get indices of members of the current class in the original array and populate the new class data array
        indices = np.where(in_data == c)
        class_data[indices] = k

        # Create the class bounds
        class_bounds[k, :] = class_avg[k]

        # Increment the current class
        k = k + 1

    return class_data, class_avg, class_bounds

def segment_equalInterval(in_data, n_classes):
    """
        Method of segmentation where each class has the same interval of values.
        Some classes may be empty.
        Returns the class data (where each pixel has the value of its class), class averages and class boundaries.

        :param in_data: 2d array of input data values
        :param n_classes: Number of classes to segment data into
        :return in order: class_data, class_avg, class_bounds, n_classes
    """

    # Create mask of filtered_data where the background is considered the invalid value
    mask = ma.masked_invalid(in_data)

    # Save indices of valid values in the masked array
    indices_valid = np.where(mask == in_data)

    # Get a flat array of the valid values
    flat_data = mask.compressed()

    # Get sorted data
    sorted_data = np.sort(flat_data)

    # Get the indices that would sort the data from smallest to largest
    data_sort_indices = flat_data.argsort().argsort()

    # Get the range of the data
    data_range = np.ptp(sorted_data)

    # Get min of data
    data_min = np.min(sorted_data)

    # Get class width
    class_width = (data_range / n_classes)

    # Preallocate the class data flattened array
    class_data_flat = np.ones(flat_data.shape, dtype=np.int64)

    # pre-allocate class averages
    class_avg = np.zeros(n_classes, dtype=np.float32)

    # pre-allocate class bounds
    class_bounds = np.zeros((class_avg.shape[0], 2), dtype=np.float64)

    # loop through each class except for the last one
    for i in range(0, n_classes):
        # Calculate class bounds
        class_bounds[i, :] = [i * class_width + data_min, (i + 1) * class_width + data_min]

        # Find indices of values that belong to current class
        indices = np.where(np.logical_and(sorted_data >= class_bounds[i, 0],
                                          sorted_data <= class_bounds[i, 1]))

        # Convert values to class labels
        class_data_flat[indices] = i + 1

        # calculate class averages
        class_avg[i] = (class_bounds[i, 0] + class_bounds[i, 1]) / 2

    # Sort class data into correct order
    class_data_flat = class_data_flat[data_sort_indices]

    # Create data array that is all background
    class_data = np.ones(in_data.shape, dtype=np.int64) * (-1)

    # Populate it with our values
    class_data[indices_valid] = class_data_flat

    return class_data, class_avg, class_bounds

def segment_bestFitJenks(in_data, n_classes):
    """
        THIS FUNCTION IS EXPERIMENTAL.

        Method of segmentation where the class breaks are computed using the Jenks best fit method.
        Returns the class data (where each pixel has the value of its class), class averages and class boundaries.

        :param in_data: 2d array of input data values
        :param n_classes: Number of classes to segment data into
        :return in order: class_data, class_avg, class_bounds, n_classes
    """

    # Create mask of filtered_data where the background is considered the invalid value
    mask = ma.masked_invalid(in_data)

    # Save indices of valid values in the masked array
    indices_valid = np.where(mask == in_data)

    # Get a flat array of the valid values
    flat_data = mask.compressed()

    # Compute class breaks using jenks best fit
    breaks = jenkspy.jenks_breaks(flat_data, n_classes)

    # Preallocate the class data flattened array
    class_data_flat = np.zeros(flat_data.shape, dtype=np.int64)

    # pre-allocate class averages
    class_avg = np.zeros(n_classes, dtype=np.float32)

    # pre-allocate class bounds
    class_bounds = np.zeros((class_avg.shape[0], 2), dtype=np.float64)

    # Loop through each class except for the last
    for i in range(0, n_classes):
        # Compute class bounds
        class_bounds[i, :] = [breaks[i], breaks[i + 1]]

        # Compute class average
        class_avg[i] = (breaks[i] + breaks[i + 1]) / 2

        # Find indices of values that belong to current class
        indices = np.where(np.logical_and(flat_data >= class_bounds[i, 0],
                                          flat_data <= class_bounds[i, 1]))

        # Convert values to class labels
        class_data_flat[indices] = i + 1

    # Create data array that is all background
    class_data = np.ones(in_data.shape, dtype=np.int64) * (-1)

    # Populate it with our values
    class_data[indices_valid] = class_data_flat

    return class_data, class_avg, class_bounds

def segment_equalValue(in_data, n_classes):
    """
         Places an equal number of values in each class independent of the frequency of the value.
         Returns the class data (where each pixel has the value of its class), class averages and class boundaries.

        :param in_data: 2d array of input data values
        :param n_classes: Number of classes to segment data into
        :return in order: class_data, class_avg, class_bounds, n_classes
    """

    # Create mask of filtered_data where the background is considered the invalid value
    mask = ma.masked_invalid(in_data)

    # Save indices of valid values in the masked array
    indices_valid = np.where(mask == in_data)

    # Get a flat array of the valid values
    flat_data = mask.compressed()

    # Get sorted data
    sorted_data = np.sort(flat_data)

    # Get the indices that would sort the data from smallest to largest
    data_sort_indices = flat_data.argsort().argsort()

    # Get number of values per class
    N_per_class = math.floor(len(flat_data) / n_classes) - 1

    # Preallocate the class data flattened array
    class_data_flat = np.zeros(flat_data.shape, dtype=np.int64)

    # pre-allocate class averages
    class_avg = np.zeros(n_classes, dtype=np.float32)

    # pre-allocate class bounds
    class_bounds = np.zeros((class_avg.shape[0], 2), dtype=np.float64)

    # Loop through each class except for the last
    for i in range(0, n_classes - 1):
        # Compute flattened class data
        class_data_flat[i * N_per_class: (i + 1) * N_per_class - 1] = i + 1

        # Compute class bounds
        class_bounds[i, :] = [sorted_data[i * N_per_class], sorted_data[(i + 1) * N_per_class - 1]]

        # Compute class average
        class_avg[i] = (sorted_data[i * N_per_class] + sorted_data[(i + 1) * N_per_class - 1]) / 2

    # Do the last class- we need to do this separately to account for the possibilty of an uneven number of values
    i = n_classes - 1
    class_data_flat[i * N_per_class: len(class_data_flat)] = i + 1  # Flattened class data
    class_bounds[i, :] = [sorted_data[i * N_per_class], sorted_data[len(class_data_flat) - 1]]  # Class bounds
    class_avg[i] = (sorted_data[i * N_per_class] + sorted_data[len(class_data_flat) - 1]) / 2  # Class averages

    # Sort class data into correct order
    class_data_flat = class_data_flat[data_sort_indices]

    # Create data array that is all background
    class_data = np.ones(in_data.shape, dtype=np.int64) * (-1)

    # Populate it with our values
    class_data[indices_valid] = class_data_flat

    return class_data, class_avg, class_bounds

def segment_equalRecords(in_data, n_classes):
    """
        Places an equal number of values in each class without splitting two of the same value into different classes.
        Returns the class data (where each pixel has the value of its class), class averages and class boundaries.

        :param in_data: 2d array of input data values
        :param n_classes: Number of classes to segment data into
        :return in order: class_data, class_avg, class_bounds, n_classes
    """

    # Create mask of filtered_data where the background is considered the invalid value
    mask = ma.masked_invalid(in_data)

    # Save indices of valid values in the masked array
    indices_valid = np.where(mask == in_data)

    # Get a flat array of the valid values
    flat_data = mask.compressed()

    # Get sorted data
    sorted_data = np.sort(flat_data)

    # Get the indices that would sort the data from smallest to largest
    data_sort_indices = flat_data.argsort().argsort()

    # Get number of values per class
    N_per_class = math.ceil(len(flat_data) / n_classes) - 1

    # Preallocate the class data flattened array
    class_data_flat = np.ones(flat_data.shape, dtype=np.int64)

    # pre-allocate class averages
    class_avg = np.zeros(n_classes, dtype=np.float32)

    # pre-allocate class bounds
    class_bounds = np.zeros((class_avg.shape[0], 2), dtype=np.float64)

    # Get array of the unique values in the data, indices of the first occurrences of the unique values and the counts
    unique_vals, unique_indices = np.unique(sorted_data, return_index=True)

    # Loop through all the unique values and adjust the class breaks accordingly (except for the last one)
    k = 0  # Current class
    last_break = -1  # Break of previous class
    for i in range(0, len(unique_vals) - 1):
        # If the last occurence of the current unique value is in another category, push the class break forward
        if unique_indices[i + 1] - 1 > (k + 1) * N_per_class:
            class_bounds[k, :] = [sorted_data[last_break + 1],
                                  sorted_data[unique_indices[i + 1] - 1]]  # Compute bounds
            last_break = unique_indices[i + 1] - 1  # Save last class break
            class_avg[k] = (class_bounds[k, 0] + class_bounds[k, 1]) / 2  # Compute class average
            k = k + 1

    # Need to to do the last one separately if we have an uneven number of values
    if k != n_classes:
        class_bounds[k, :] = [sorted_data[last_break + 1], sorted_data[len(sorted_data) - 1]]  # Compute class bounds
        class_avg[k] = (class_bounds[k, 0] + class_bounds[k, 1]) / 2  # Compute class average

    # For each class, convert the actual data values to the class labels
    for i in range(0, n_classes):
        # Compute indices that belong to the current class
        indices = np.where(np.logical_and(sorted_data >= class_bounds[i, 0], sorted_data <= class_bounds[i, 1]))

        # Convert values to class labels
        class_data_flat[indices] = i + 1

    # Sort class data into correct order
    class_data_flat = class_data_flat[data_sort_indices]

    # Create data array that is all background
    class_data = np.ones(in_data.shape, dtype=np.int64) * (-1)

    # Populate it with our values
    class_data[indices_valid] = class_data_flat

    return class_data, class_avg, class_bounds

def segment_stddev(in_data, n_classes):
    """
        Function that classifies the input data using standard deviation breaks.
        First, the mean and standard deviation of the data are calculated, then the the class breaks are calculated
        as proportions from the standard deviation (i.e, 1, 0.5, 0.25, 0.1... etc).

        :param in_data: 2d array of input data values
        :param n_classes: Number of classes to segment data into
        :return in order: class_data, class_avg, class_bounds, n_classes
    """

    # Create mask of filtered_data where the background is considered the invalid value
    mask = ma.masked_invalid(in_data)

    # Save indices of valid values in the masked array
    indices_valid = np.where(mask == in_data)

    # Get a flat array of the valid values
    flat_data = mask.compressed()

    # Calculate stats
    stddev = np.std(flat_data)
    mean = np.mean(flat_data)

    # Compute max and min of data
    min_val = flat_data.min()
    max_val = flat_data.max()

    # Get range of each class
    class_range = np.ptp(flat_data)/n_classes

    # Calculate the proportion of the standard deviation at which class breaks will be defined
    p = class_range/stddev

    # Preallocate the class data flattened array
    class_data_flat = np.ones(flat_data.shape, dtype=np.int64)

    # Pre-allocate class averages
    class_avg = np.zeros(n_classes, dtype=np.float32)

    # Pre-allocate class bounds
    class_bounds = np.zeros((class_avg.shape[0], 2), dtype=np.float64)



    # ---------- Compute class bounds and determine which interval mean is in ---------- #

    # Value by which to shift the class bounds so that the mean lines up with one of the class breaks
    mean_shift = 0

    # Loop through all classes
    for i in range(0, n_classes):
        class_bounds[i, :] = [min_val + stddev*p*i, min(min_val + stddev*p*(i + 1), max_val)]

        # If the mean is in the current interval then calculate the amount to shift the bounds by so that mean lines up
        # with one of the class breaks
        if np.logical_and(mean >= class_bounds[i,0], mean < class_bounds[i,1]):
            mean_shift = abs(mean-class_bounds[i,0])

    # Shift all the class bounds up so that mean lines up with one of the breaks
    class_bounds = class_bounds + mean_shift

    # Adjust the top and bottom class limits to be the max and min, since these will be shifted as well
    class_bounds[n_classes-1,1] = max_val
    class_bounds[0,0] = min_val

    # # If the upper class is now outside of the range, delete it
    # if class_bounds[n_classes-1,0] >= max_val:
    #     class_bounds = np.delete(class_bounds, n_classes-1, axis=0)
    #
    # # If we need to add a new class at the start, add it
    # if class_bounds[0,0] > min_val:
    #     new_bounds = [min_val, class_bounds[0,0]]
    #     class_bounds = np.insert(class_bounds, 0, new_bounds, axis=0)

    # ---------- Compute class averages and data ---------- #

    # Loop through all classes
    for i in range(0, n_classes):
        # Compute indices that belong to the current class
        indices = np.where(np.logical_and(flat_data >= class_bounds[i, 0], flat_data <= class_bounds[i, 1]))

        # Compute class average
        class_avg[i] = (class_bounds[i, 0] + class_bounds[i, 1]) / 2

        # Convert values to class labels
        class_data_flat[indices] = i + 1

    # Create data array that is all background
    class_data = np.ones(in_data.shape, dtype=np.int64) * (-1)

    # Populate it with our values
    class_data[indices_valid] = class_data_flat

    return class_data, class_avg, class_bounds

def segment_manual(in_data, class_breaks):
    """
        Function that classifies the input data using manually inputted class breaks.

        :param in_data: 2d array of input data values
        :param n_classes: Number of classes to segment data into
        :return in order: class_data, class_avg, class_bounds, n_classes
    """

    # Check that the class breaks are valid
    if len(class_breaks) >= 1:
        # Determine the number of classes
        n_classes = len(class_breaks) + 1
    else:
        return None, None, None

    # Create mask of filtered_data where the background is considered the invalid value
    mask = ma.masked_invalid(in_data)

    # Save indices of valid values in the masked array
    indices_valid = np.where(mask == in_data)

    # Get a flat array of the valid values
    flat_data = mask.compressed()

    # Preallocate the class data flattened array
    class_data_flat = np.ones(flat_data.shape, dtype=np.int64)

    # Pre-allocate class averages
    class_avg = np.zeros(n_classes, dtype=np.float32)

    # Pre-allocate class bounds
    class_bounds = np.zeros((class_avg.shape[0], 2), dtype=np.float64)

    # Get min and max of the data
    data_min = flat_data.min()
    data_max = flat_data.max()












    # Add the min and max to the class breaks
    class_breaks.append(data_max)
    class_breaks = np.insert(class_breaks, 0, data_min, axis=0)

    # Loop through all classes
    for i in range(0, n_classes):
        class_bounds[i, :] = [ class_breaks[i], class_breaks[i+1] ] # Compute class bounds

        class_avg[i] = (class_bounds[i, 0] + class_bounds[i, 1]) / 2 # Compute class average

        # Compute indices that belong to the current class
        indices = np.where(np.logical_and(flat_data >= class_bounds[i, 0], flat_data <= class_bounds[i, 1]))

        # Convert values to class labels
        class_data_flat[indices] = i + 1

    # Create data array that is all background
    class_data = np.ones(in_data.shape, dtype=np.int64) * (-1)

    # Populate it with our values
    class_data[indices_valid] = class_data_flat

    return class_data, class_avg, class_bounds


# =============== Helper functions =============== #

def downsample(input_path, output_path, res_out):
    # Load dataset
    src = gdal.Open(input_path)

    # Get the input projection
    prj = src.GetProjection()

    # Find the unit of measure used by the projection to catch cases where the unit is not in metres
    srs = osr.SpatialReference(wkt=prj)

    prj_units = srs.GetAttrValue('PROJCS|UNIT') # Get units of projection

    if prj_units != "metre" and prj_units != "meter": # Throw an error if the source is not in units of metres
        print('Downsample error: Unit of source projection is not metres')
        return None

    # If a non-empty output file path was specified, save the output to file
    if output_path != '':
        # Warp geotiff to required resolution and output
        gdal.Warp(output_path + '_ndvi.tif', src,
                        options=gdal.WarpOptions(format='GTiff', xRes=res_out, yRes=res_out))

    # Else if non-empty output just return the dataset
    else:
        return gdal.BuildVRT('', src, xRes=res_out, yRes=res_out)

def upsample(data_in, x_size, y_size):
    """
        Upsamples the current filtered data in the class to the chosen size, using the Kronecker product
    """
    min_val = np.min(data_in)
    max_val = np.max(data_in)
    data_out = imresize(data_in, [x_size, y_size], mode='L', interp='nearest')

    return ( data_out / 255 * (max_val - min_val) + min_val ).astype(int)

def get_bkg_mask(data_in):
    """
        Detects the background in some input data and then returns the original data with the background as NaN values.
        Background MUST be contiguous for the output to be correct.
    """

    # Get array dims
    rows = data_in.shape[0]
    cols = data_in.shape[1]

    # Sample corner points
    corner_vals = []
    corner_vals.append(data_in[0, 0]) # top-left
    corner_vals.append(data_in[0, cols-1]) # top-right
    corner_vals.append(data_in[rows-1, 0]) # bottom-left
    corner_vals.append(data_in[rows-1, cols-1]) # bottom-right

    # Determine background value from mode of corner values
    bkg_value = mode(corner_vals)[0][0]

    # If the background value is NaN, then handle it differently (can't test equality for nans since np.nan==np.nan = False)
    if np.isnan(bkg_value):
        # Create copy of original data where background is 0 and field is 1
        bin_data = np.isnan(data_in)
    else:
        bin_data = (data_in == bkg_value)

    # Label the regions in the data using neighbour-only connectivity (sides only, no diagonals)
    labeled = measure.label(bin_data, background=None, connectivity=1)

    # Determine the value of the background label from the top-left pixel
    bkg_label = labeled[0,0]

    # Get the background region properties
    rp = measure.regionprops(labeled)
    props = rp[bkg_label-1]

    # Create the output data
    out_data = data_in.copy()

    # Give list of coordinates to the background data
    out_data[props.coords[:,0], props.coords[:,1]] = np.nan

    return out_data

def read_zip_file(path):
    """
        Function that reads a zip file and returns a specific file

    """

    zfile = zipfile.ZipFile(path)

    zfile.extract('Fantasy Jungle.pdf', 'C:\\Stories\\Fantasy')

    for finfo in zfile.infolist():
        ifile = zfile.open(finfo)
        line_list = ifile.readlines()
        print(line_list)

def get_bkg_val(array):
    """
        Determines the background value of an array by sampling the corner points and taking the mode.

    """
    # Get array dims
    rows = array.shape[0]
    cols = array.shape[1]

    # Sample corner points
    corner_vals = []
    corner_vals.append(array[0, 0]) # top-left
    corner_vals.append(array[0, cols-1]) # top-right
    corner_vals.append(array[rows-1, 0]) # bottom-left
    corner_vals.append(array[rows-1, cols-1]) # bottom-right

    # Determine background value from mode of corner values
    bkg_val = mode(corner_vals)[0][0]

    return bkg_val