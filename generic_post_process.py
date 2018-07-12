

'''
    Project:
        Post processing

    Author:
        Alex Cornelio (ascornelio@flurosat.com)

    File:
       Generic post processing

    Dependencies and Version:
        python 3.6.3
        gdal 2.2.3
        scipy

    Notes:
        This script is a quick and dirty solution for applying post processing for generated indices.
        The post processing includes:
        - All classification methods with 5 classes for each on all indices

    Assumptions:
        Indices are already generated

    TODO:
        Add downsample param

'''

from glob import glob
import os
import operator
import threading
import subprocess
import sys
from datetime import datetime
from osgeo import gdal
from osgeo import gdal_array
import numpy as np
import time
import os
import math
from scipy.misc import imsave
import psutil
import logging
import math
import shutil
import argparse

import classification as classi     # toms code


# METHODS:
CLASSIFICATION_METHODS = ['equalInterval', 'equalRecords', 'stddev']
CLASSES = [3,4,5,6,7,8,10]
SMOOTHINGS = [0, 10, 15, 20, 25, 30, 35,40]
DEFAULT_METHODS = CLASSIFICATION_METHODS
DEFAULT_SMOOTHINGS = [10]
DEFAULT_CLASSES = [5]
DEFAULT_DOWNSAMPLE = True


class GenericPostProcessing(object):
    '''
        Class to hold all informatino for post processing procedures
    '''
    def __init__(self, methods, classes, smoothings):
        self.methods = methods
        self.classes = classes
        self.smoothings = smoothings
        self.indice_to_results = {}   # list of dictionaries where each dict has a key of a index path and a value of its results path
        self.indice_paths = []
        self.filtered_indice_paths = []
        self.starting_path = os.getcwd()



    def get_indices(self):
        '''
        This method uses depth first search to find all the directories that contain a .p4d project (ie. directories with results from pix4d processing)
        :return: list of directories of indice results
        '''
        fringe_dirs = glob(self.starting_path + '\\*\\')

        while fringe_dirs:

            # expand deepest node from fringe and remove it
            depth_levels = [i.count('\\') for i in fringe_dirs]
            max_index, max_value = max(enumerate(depth_levels), key=operator.itemgetter(1))
            node_to_expand = fringe_dirs.pop(max_index)

            # get children
            children = glob(node_to_expand + '\\*\\')

            # check for index tiff files
            for file in os.listdir(node_to_expand):
                file_path = os.path.join(node_to_expand, file)
                if 'index' in file and file[-3:].lower() == 'tif' and os.path.getsize(file_path) > 0:
                    self.indice_paths.append(file_path)                    
            else:
                fringe_dirs += children

        logger.info("The following indice paths have been found: {}".format(self.indice_paths))




    def filter_indices(self):
        '''
            Function filters the paths to the indice tifs
            Does this by ensuring certain paths do not have folders in directories such as  'tiles' or 'contours'
        '''
        for path in self.indice_paths:
            if 'tiles' in path or 'contour' in path:
                self.indice_paths.remove(path)


        logger.info("Paths to filtered indices include: {}".format(self.indice_paths))




    def indice_classification(self):
        '''
            Function is a wrapper for Tom Lawson's Classification code.
            For each indice, preform all methods specified for all number of classes and smoothing parameters. 

            For more details on the classification, please see Tom's Classification pipeline document found
            on the FluroSat-py github repo
        '''

        # iterate through indice path to results dictrionary
        for indice_path in self.indice_to_results:

            # get corresponding results path
            results_path = self.indice_to_results[indice_path]

            # select all methods
            class_names = []

            for m in self.methods:

                for c in self.classes:

                    for s in self.smoothings:

                        # Configure output directory paths for all exports
                        indice_results_basename = os.path.basename(indice_path) + '_' + m + '_c' + str(c) + '_s' + str(s)
                        smoothed_results_path = os.path.join(results_path, (indice_results_basename + '_filtered' + '.tif'))
                        classesNpy = os.path.join(results_path, (indice_results_basename + '_classes.npy'))
                        jsonName = os.path.join(results_path, (indice_results_basename + '_points.json'))
                        gtiffName = os.path.join(results_path, (indice_results_basename + '_gtiff.tif'))
                        gjpegName = os.path.join(results_path, (indice_results_basename + '_gjpeg.jpg'))
                        wldName = os.path.join(results_path, (indice_results_basename + '_gjpeg.wld'))                       
                        auxName = os.path.join(results_path, (indice_results_basename + '_gjpeg.jpg.aux.xml'))
                        previewName = os.path.join(results_path, (indice_results_basename + '_preview.png'))
                        shapefilePath = os.path.join(results_path, (indice_results_basename + '_shapefile'))
                        kmlPath = os.path.join(results_path, (indice_results_basename + '.kml'))

                        logger.info("Running classifcation for {}".format(indice_results_basename))


                        # Filter (smooth) data
                        if s != 0:
                            classi.filter(indice_path, smoothed_results_path, smoothing=[s], res=None) # Smoothing = variable, grid resolution = original
                        else:
                            smoothed_results_path = indice_path

                        # Classify data
                        classi.classify(smoothed_results_path, classesNpy, m, [c], class_breaks=None)  # Classify the filtered image

                        # Get class bounds and averages
                        class_bounds = classi.get_class_bounds(classesNpy, jsonName)

                        # Get class areas in units of Ha
                        areas = classi.get_class_areas( smoothed_results_path,
                                                            classesNpy,
                                                            percentage=False )

                        # Export to various output formats
                        classi.export_as_gtiff( smoothed_results_path,
                                                    classesNpy,
                                                    gtiffName, EPSG=4326 )

                        classi.export_as_gjpeg( smoothed_results_path,
                                                classesNpy,
                                                gjpegName, EPSG=4326 )

                        # classi.export_as_kmz( 'output/'+os.path.splitext(paths[i])[0]+'filtered_'+str(c)+'classes_smoothing'+str(s)+'.tif',
                        #                       'output/' + os.path.splitext(paths[i])[0] + 'class_data_' + str(c) + 'classes.npy',
                        #                       'output/' + os.path.splitext(paths[i])[0] + str(c) + 'classes.kmz', EPSG=4326 )

                        classi.export_as_kml( smoothed_results_path,
                                                classesNpy,
                                                kmlPath, EPSG=4326 )

                        classi.export_as_image( smoothed_results_path,
                                                classesNpy,
                                                previewName )

                        classi.export_as_shapefile( smoothed_results_path,
                                                    classesNpy,
                                                    shapefilePath,
                                                    EPSG=4326 )

                        logger.info("Successfully completed all exports for {}".format(indice_results_basename))


            #     # move all results to a folder
            #     # create dir
            #     os.makedirs(baseName)
            #     class_names.append(baseName)
            #     # move to new dir
            #     shutil.move(classesNpy, baseName)
            #     shutil.move(filteredName, baseName)
            #     shutil.move(gtiffName, baseName)
            #     shutil.move(gjpegName, baseName)
            #     shutil.move(previewName, baseName)
            #     shutil.move(shapefilePath, baseName)
            #     shutil.move(kmlPath, baseName)
            #     shutil.move(jsonName, baseName)
            #     shutil.move(auxName, baseName)
            #     shutil.move(wldName, baseName)

            # # move subfolders to a higher dir
            # final_path  = path[:-4]
            # os.makedirs(final_path)
            # shutil.move(path, final_path)
            # for clas in class_names:
            #     shutil.move(clas, final_path)





    def create_indice_classification_dir(self):
        '''
            Function creates a convienent directory path to store all classified results.
            It does this by getting all relative paths to indices, appending a prefix of 'classified_results' and then
            creating all those dirs
        '''

        for path in self.indice_paths:
            # get relative path and ignore the .tif
            relative_results_path = os.path.join('Classified_Results', os.path.relpath(os.path.dirname(path)))

            # store path to tif and path to tif's results
            self.indice_to_results[path] = relative_results_path

            # try and create results path
            try:
                os.makedirs(relative_results_path)
            except Exception as e:
                logger.debug('{} already exists'.format(relative_results_path))
                logger.debug(e)

        logger.info("Paths from indices to corresponding results include: {}".format(self.indice_to_results))




def receive_args():
    '''
        Collect user's command line inputs.
        Use arg parse to do this convienently. 
        The three keys are Methods, Classes, Smoothings. All keys will have a value of a list. 

        This function also checks that the inputs are valid. If not, program will quit.
        This function writes all inputs to logger.

    '''
    description = "python generic_post_processing.py --Methods < > --Classes < > --Smoothing < > \n For example: python generic_post_processing --Methods equalInterval, equalRecords, stddev --Classes 1 5 10 --Smoothing 10 20 30"
    parser = argparse.ArgumentParser(description)
    parser.add_argument('--Methods', nargs='+', help = 'Enter the different types of classification methods you wish to run. These include equalInterval, equalRecords, stddev')
    parser.add_argument('--Classes', nargs='+', help = 'Enter the amount of classes to classify with. A number of classes can be run. For example 1,3,5,8,10')
    parser.add_argument('--Smoothings', nargs='+', help = 'Enter the amount of smoothing to classify with. Again a number of smoothings can be run. For example 10,15,20,25')
    #parse.add_argument('--Downsample', nargs='+', help = 'Enter True or False if you wish to downsample the indices')
    args = parser.parse_args()

    # Collect results
    for key, value in parser.parse_args()._get_kwargs():
        if key == 'Methods':
            methods = value
        elif key == 'Classes':
            classes = value
        elif key == 'Smoothings':
            smoothings = value
        elif key == 'Downsample':
            downsample = value
        else:
            logger.debug("Incorrect input entered: {}".format(value))

    # check methods. if empty set to default. if not, check validity
    if not methods:
        logger.info('No input for methods. Methods set to default')
        methods = DEFAULT_METHODS
    else:
        for method in methods:
            if method not in CLASSIFICATION_METHODS:
                logger.critical('Input for --Method is incorrect: {}'.format(method))
                logger.critical('Please enter the correct inputs.')
                logger.critical('Post processing will shutdown now')
                sys.exit()

    # check classes. if empty set to default. if not, check validity
    if not classes:
        logger.info('No input for classes. Classes set to default')
        classes = DEFAULT_CLASSES
    else:
        for classs in classes:
            if int(classs) not in CLASSES:
                logger.critical('Input for --Classes is incorrect: {}'.format(classs))
                logger.critical('Please enter the correct inputs.')
                logger.critical('Post processing will shutdown now')
                sys.exit()

    # check smoothing. if empty set to default. if not, check validity
    if not smoothings:
        logger.info('No input for smoothings. Smoothings set to default')
        smoothings = DEFAULT_SMOOTHINGS
    else:
        for smoothing in smoothings:
            if int(smoothing) not in SMOOTHINGS:
                logger.critical('Input for --Smoothing is incorrect: {}'.format(smoothing))
                logger.critical('Please enter the correct inputs.')
                logger.critical('Post processing will shutdown now')
                sys.exit()

    # # check downsampling
    # if not downsample:
    #     logger.info('No input for smoothings. Smoothings set to default')
    #     downsample = DEFAULT_DOWNSAMPLE
    # else:
    #     for downsample in downsamples:
    #         if downsamples not True or False:
    #             logger.critical('Input for --Smoothing is incorrect: {}'.format(smoothing))
    #             logger.critical('Please enter the correct inputs.')
    #             logger.critical('Post processing will shutdown now')
    #             sys.exit()


    # yay you did it, tell the logger
    logger.info("User input successful. Methods: {}. Classes: {}. Smoothings: {}.".format(methods, classes, smoothings))

    return methods, classes, smoothings



def create_log_file():
    '''
        Function creates a log file that will have messages appended to it throughout the entire pipeline. 
        The name of this log file is pix4d_processing.log and can be found in the directory the code was called from.
        In the log file, the INFO term is used for significant processing events. The DEBUG term is used for encountered problems. 
    '''
    # set logger to DEBUG mode and create a instance
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    # create a file handler
    handler = logging.FileHandler("post_processing.log")

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handler to the logger
    logger.addHandler(handler)

    # write initalisation message
    logger.info("\tPOST PROCESSING INITIATED")

    return logger


def main():
    '''
        Main body of code
    '''
    # create global log file
    global logger
    logger = create_log_file()

    # receive arguments for processing
    methods, classes, smoothings = receive_args()

    # create instance of class
    post_processing = GenericPostProcessing(methods, classes, smoothings)

    # find all indices
    post_processing.get_indices()

    # filter paths
    post_processing.filter_indices()

    # indice classification post processing done here
    post_processing.create_indice_classification_dir()

    # configure classification methods, number of classes and smoothing
    post_processing.indice_classification()







if __name__ == '__main__':
    main()