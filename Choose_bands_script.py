# Function for creating a lower band image by inputting the desired centre
# wavelengths and the wavelength bandwidths

# Import packages
from spectral import *

# Function for finding index or band of the wavelength of interest
def get_wl_idx(imag, wl_value):
    # Import numpy
	import numpy as np
	# Use the centers property of the image to find index closest to wl_value
    return int((np.abs(np.array(imag.bands.centers)-wl_value)).argmin())

# Function for producing a broadband image	
def broadband_image(imag, wl_value, delta_wl):
    # Function takes in the SPy image, wavelength of interest and the bandwidth
	# in order to output a new image
	
	# First find the index for the lower and upper limits of the wavelengths
	low_idx = get_wl_idx(imag,(wl_value-(delta_wl/2)))
	upp_idx= get_wl_idx(imag,(wl_value+(delta_wl/2)))
	
	# Then add the contribution from each band in the range
	bb_imag = imag.read_band(upp_idx)
	for idx in range(low_idx,upp_idx):
	    bb_imag = bb_imag + imag.read_band(idx)
	
	# Next divide by the number of indexes to find the average:
	bb_imag = bb_imag/(upp_idx-low_idx)
	return bb_imag

def choose_bands_image(imag,wl_list,wl_bw):
    # Function takes in SPy file object and a list of the centre wavelengths
	# that you wish to create. The bandwidth of the wavelengths can either be
	# all the same (i.e. single number) or a list of bandwidths
	import numpy as np
	
	new_imag = np.zeros([imag.shape[0],imag.shape[1],len(wl_list)],dtype='uint16')
	
	if type(wl_bw) == list:
	    for i in range(len(wl_list)):
		    new_imag[:,:,i] = broadband_image(imag,wl_list[i],wl_bw[i])
	else:
	    for i in range(len(wl_list)):
		    new_imag[:,:,i] = broadband_image(imag,wl_list[i],wl_bw)
			
	return new_imag
		
def save_geotiff(filename, SPy, imag):
	from osgeo import gdal
	from osgeo import osr	
	# Function takes in string "filename" for the desired output name, the 
	# original SPy File object which contains the metadata that is to be
	# retained 'SPy' and lastly the image array that is to be saved 'imag'
	save_rgb(filename, imag, format='tiff')
	dst_ds = gdal.Open(filename, gdal.GA_Update)
    dst_ds.SetGeoTransform([float(SPy.metadata['map info'][3]), float(SPy.metadata['map info'][5]), 0, float(SPy.metadata['map info'][4]), 0, -float(SPy.metadata['map info'][6])])
    srs = osr.SpatialReference()
    srs.SetUTM(int(SPy.metadata['map info'][7]), 0) # SetUTM(utm_zone, North_true={1 for north, 0 for south})
    srs.SetWellKnownGeogCS("WGS84")
    dst_ds.SetProjection(srs.ExportToWkt())
    # Once we're done, close properly the dataset
    dst_ds = None
	SPy = None
	imag = None
	filename = None	
	
# Open image using a hdr file that has been altered to include the byte order 
# line
img = envi.open('DiseaseTest_Cube90.bip.hdr')	
	
"""
    ...: Sentinel-2 Bands:			WL(nm)   Res(m)  bandwidth(nm)
    ...: Band 2 – Blue    				490    10    65
    ...: Band 3 – Green    				560    10    35
    ...: Band 4 – Red    				665    10    30
    ...: Band 5 – Vegetation Red Edge   705    20    15
    ...: Band 6 – Vegetation Red Edge   740    20    15
    ...: Band 7 – Vegetation Red Edge   783    20    20
    ...: Band 8A – Narrow NIR    		865    20    20
    ...: """	

wl_list = [490,560,665,705,740,783,865]
wl_bw = [65,35,30,15,15,20,20]

img_Sent2_bands = choose_bands_image(img,wl_list,wl_bw)

view = imshow(img_Sent2_bands)

save_geotiff('img_20bands.tiff',img,img_20_band)	