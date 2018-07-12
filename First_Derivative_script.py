# Function for determining the first derivative:

# Import packages
from spectral import *
import numpy as np

# Function for finding index or band of the wavelength of interest
def get_wl_idx(imag, wl_value):
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
	
def FD_image(img,wl):
    # Function takes in SPy object "img" and the wavelength of desire "wl"
    wl_idx = get_wl_idx(img,wl)
	delta_wl = img.bands.centers[wl_idx+1]-img.bands.centers[wl_idx-1]
	return (img.read_band(wl_idx+1) - img.read_band(wl_idx-1))/(2*delta_wl)
	
view = imshow(FD_731,stretch=0.001)	