# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.visualization import ZScaleInterval
#%matplotlib inline
from astropy.io import fits
import os
import glob
import math
from scipy.ndimage import shift


def master_bias(filelist, outputfile):
    '''
    Generates a master bias from a list of bias frames, writes the master file in .fit format
    
    Parameters: 
    filelist (list of str): A list containing all the bias files
    outputfile (str): a file path to save master bias file
    
    Returns:
    Array of master bias 
    '''
    # set n equal to the number of files in filelist
    n = len(filelist)

    # get first frame header
    first_frame_header = fits.getheader(filelist[0])

    # set first_frame_data equal to the data array of the first file in filelist
    first_frame_data = fits.getdata(filelist[0])

    # get the dimensions of the first file in filelist
    imsize_y, imsize_x = first_frame_data.shape

    # set the values in the array equal to zero
    fits_stack = np.zeros((imsize_y, imsize_x , n))

    # Insert each frame into a three dimensional stack, one by one:
    for ii in range(0, n):
        im = fits.getdata(filelist[ii])
        fits_stack[:,:,ii] = im

    # Take the median of the stack
    med_frame = np.median(fits_stack, axis = 2)

    # create output file directory if it does not exist
    if not os.path.exists(os.path.dirname(outputfile)):
        os.mkdir(os.path.dirname(outputfile))
    
    # write median combined output fits file
    fits.writeto(outputfile, med_frame, header=first_frame_header, output_verify='exception', overwrite=True, checksum=False)
    print(f"Master Bias file saved as: {os.path.basename(outputfile)}")

    return med_frame



def master_dark(filelist, master_bias, outputfile):
    '''
    Generates a master dark from a list of dark frames, writes the master file in .fit format

    Parameters:
    filelist (str): list of all dark frame fits files
    master_bias (array): the array of master bias
    outputfile (str): a file path to save master dark file 

    Returns:
    array: master dark frame
    '''
    # set n equal to the number of files in filelist
    n = len(filelist)

    # get first frame header
    first_frame_header = fits.getheader(filelist[0])

    # set first_frame_data equal to the data array of the first file in filelist
    first_frame_data = fits.getdata(filelist[0])

    # get the dimensions of the first file in filelist
    imsize_y, imsize_x = first_frame_data.shape

    # set the values in the array equal to zero
    fits_stack = np.zeros((imsize_y, imsize_x , n))

    # Insert each frame into a three dimensional stack, subtracts bias
    # and normalizes by exposure time, one by one:
    for ii in range(0, n):
        exptime = fits.getheader(filelist[ii])['EXPTIME']
        im = (fits.getdata(filelist[ii]) - master_bias) / exptime
        fits_stack[:,:,ii] = im

    # Take the median of the stack
    med_frame = np.median(fits_stack, axis = 2)

    # create output file directory if it does not exist
    if not os.path.exists(os.path.dirname(outputfile)):
        os.mkdir(os.path.dirname(outputfile))
    
    # write median combined output fits file
    fits.writeto(outputfile, med_frame, header=first_frame_header, output_verify='exception', overwrite=True, checksum=False)
    print(f"Master dark file saved as: {os.path.basename(outputfile)}")

    return med_frame



def master_flat(filelist, master_bias, master_dark, outputfile):
    '''
    Generates a master flat from a list of flat frames, writes the master file in .fit format

    Parameters:
    filelist (str): list of all dark frame fits files
    master_bias (array): the array of master bias
    master_dark (array): the array of master dark
    outputfile (str): a file path to save master flat file 

    Returns:
    None
    '''
    # number of files in the list
    n = len(filelist)

    # get first frame header
    first_frame_header = fits.getheader(filelist[0])

    # gets the first file from the list
    first_frame_data = fits.getdata(filelist[0])

    # saves the dimensions of the fits file
    imsize_y, imsize_x = first_frame_data.shape

    # Initializes a 3d array
    fits_stack = np.zeros((imsize_y, imsize_x , n))

    # subtracts bias,normalizes by exposure time, subtracts dark, normalizes by the pixel values, 
    #then Insert each frame into a three dimensional stack, one by one:
    
    for ii in range(0, n):
        exptime = fits.getheader(filelist[ii])['EXPTIME']
        im = ((fits.getdata(filelist[ii]) - master_bias) / exptime) - master_dark
        norm_im = im/np.median(im)
        fits_stack[:,:,ii] = norm_im
    
    # Gets the median value of the pixels across the 3rd dimension of the array
    med_frame = np.median(fits_stack, axis=2)

    # create output file directory if it does not exist
    if not os.path.exists(os.path.dirname(outputfile)):
        os.mkdir(os.path.dirname(outputfile))
    
    # write median combined output fits file
    fits.writeto(outputfile, med_frame, header=first_frame_header, output_verify='exception', overwrite=True, checksum=False)
    print(f"Master flat file saved as: {os.path.basename(outputfile)}")
    
    return


def reduce_image(img, master_bias, master_dark, flats_path, outputfile):
    '''
    Reduces the image and writes a new file with the reduced image.
    
    Parameters:
    img (str): path to the image to be reduced
    master_bias (array): the array of master bias
    master_dark (array): the array of master dark
    flats_path (str): the path to the master flat file
    outputfile (str): a file path to save the reduced image 

    Returns:
    None
    '''
    # frame header
    header = fits.getheader(img)
    
    # array for image
    data = fits.getdata(img)
    
    #Getting the master flat of same filter
    master_flat = fits.getdata(flats_path)

    # Reducing image
    reduced_img = (data - master_bias - header['EXPTIME']*master_dark) / master_flat

    # create output file directory if it does not exist
    if not os.path.exists(os.path.dirname(outputfile)):
        os.mkdir(os.path.dirname(outputfile))
    
    # write median combined output fits file
    output_filename = os.path.join(outputfile, 'fdb_' + os.path.basename(img))
    fits.writeto(output_filename, reduced_img, header=header, output_verify='exception', overwrite=True, checksum=False)
    print(f"Reduced image saved as: {os.path.basename(output_filename)}")
    
    return

def measure_offset(img, ref_img, fg, bg):
    
    '''
    Measure misalignment shift between a reference image and test image.

    Arguments:
        img (str): path to reference fits files
        ref_img (str): path to test fits file to align with referene
        bg (tuple): (x, y) center of background region
        fg (tuple): (x, y) center of foreground region

    Return:
        misalignment (tuple): (dx, dy) translation shift from reference to test image

    The background region should contain no stars, and the foreground
    region should only contain a single star.
    '''
    
    # load reference and test images from fits files
    img0 = fits.getdata(img)
    img1 = fits.getdata(ref_img)

    half_width = 25

    # slice background region from reference and test images
    bg0 = img0[bg[1] - half_width : bg[1] + half_width, bg[0] - half_width : bg[0] + half_width]
    bg1 = img1[bg[1] - half_width : bg[1] + half_width, bg[0] - half_width : bg[0] + half_width]

    # slice foreground region from reference and test images
    fg0 = img0[fg[1] - half_width : fg[1] + half_width, fg[0] - half_width : fg[0] + half_width]
    fg1 = img1[fg[1] - half_width : fg[1] + half_width, fg[0] - half_width : fg[0] + half_width]

    # determine significant pixel thresholds based on background regions
    thresh0 = np.median(bg0) + 3 * np.std(bg0)
    thresh1 = np.median(bg1) + 3 * np.std(bg1)

    # determine indices or pixels over threshold in foreground regions
    (y0, x0) = np.nonzero(fg0 > thresh0)
    (y1, x1) = np.nonzero(fg1 > thresh1)

    # determine values of pixels over threshold in foreground regions
    thres_arr0 = fg0[y0,x0]
    thres_arr1 = fg1[y1,x1]

    #Formula:
    #x_center = sum(x(R[x,y]-B)) / sum(R[x,y]-B)
    #y_center = sum(y(R[x,y]-B)) / sum(R[x,y]-B)

    # computed weighted centroids of pixels over threshold in ref img
    x_0 = np.sum(x0*(thres_arr0-np.median(bg0))) / np.sum(thres_arr0-np.median(bg0))
    y_0 = np.sum(y0*(thres_arr0-np.median(bg0))) / np.sum(thres_arr0-np.median(bg0))
    
    # computed weighted centroids of pixels over threshold in image to be aligned
    x_1 = np.sum(x1*(thres_arr1-np.median(bg1))) / np.sum(thres_arr1-np.median(bg1))
    y_1 = np.sum(y1*(thres_arr1-np.median(bg1))) / np.sum(thres_arr1-np.median(bg1))

    return y_0 - y_1, x_0 - x_1



def image_registraion(files, offsets, padding, outputfile):
    '''
    Pads, shifts, stacks, and median combines a list of same band images
    for a single object.

    Arguments:
        files (list of strings): list of fit files to process
        offsets (list of tuples): list of image misalignments (dx, dy)
        padding (int): number of pixels to pad before shifting
        outputfile (str): a file path to save the registered image

    Return:
        final image

    The individual padded and shifted images are saved to fit files with
    the prefix 's' added to the original fit file name.

    The median combined image is saved to a fit file with the prefix
    'median_s' added to the first fit file name in the list. 
    '''
    #initializing array of stacked images
    img = fits.getdata(files[0])
    (x, y) = img.shape
    n = len(files)
    stack = np.zeros((y + 2 * padding, x + 2 * padding, n))

    #Padding the images 
    for ii in range(n):
        file = files[ii]
        offset = offsets[ii]
        img = fits.getdata(file)
        hdr = fits.getheader(file)
        img[np.isnan(img)] = 0.0
        img[np.isinf(img)] = 0.0
        padded_img = np.pad(img, padding, 'constant', constant_values = -1)
        
        shifted_img = shift(padded_img, offset, cval = -1)
        shifted_img[np.isnan(shifted_img)] = 0.0
        shifted_img[np.isinf(shifted_img)] = 0.0
      
        # Insert each frame into a three dimensional stack, one by one:
        stack[:,:,ii] = shifted_img
    
        # write padded and shifted fits file
        outputfile_name = os.path.join(outputfile, 's_' + os.path.basename(file))
        fits.writeto(outputfile_name, shifted_img, header=hdr, output_verify='exception', overwrite=True, checksum=False)
        print(f"Shifted image saved as: {os.path.basename(outputfile_name)}")
        
    # Take the median of the stack
    med_img = np.median(stack, axis = 2)
    med_img[np.isnan(med_img)] = 0.0
    med_img[np.isinf(med_img)] = 0.0
    
    # write median combined output fits file
    output_filename = os.path.join(outputfile , 'Stacked_image.fit')
    fits.writeto(output_filename, med_img, header=hdr, output_verify='exception', overwrite=True, checksum=False)
    print(f"Stacked image saved as: {os.path.basename(output_filename)}")
    
    return

coordinates = {
 "target" : 
    {"Visual":{"fg":(922, 3010), "bg":(2752, 1594)},
    "Blue": {"fg":(733,3064), "bg":(2752, 1594)},
    "Red": {"fg":(943, 2989), "bg":(2752, 1594)}
    },
 "standard":
    {"Visual":{"fg":(1731, 1584), "bg":(2100, 2300)},
    "Blue": {"fg":(1774, 1490), "bg":(2100, 2300)},
    "Red": {"fg":(1756,1524), "bg":(2100, 2300)}
    }
}

def reduction(folder_path, object_name):
    '''
    Performs data reduction on images.

    Arguments:
        folder_path (string): path to the top level data folder
        object_name (string): name of object specific data subfolder

    Return:
        None
    '''
    
    # Create master bias file
    biasfiles = glob.glob(os.path.join(folder_path, 'Calibration', 'Bias', 'CSJ*.fit'))
    master_bias_path = os.path.join(folder_path, 'Calibration', 'Bias', 'Master_bias.fit')
    m_bias = master_bias(biasfiles, master_bias_path)
    
    # Create master dark file
    darkfiles = glob.glob(os.path.join(folder_path, 'Calibration', 'Dark','CSJ*.fit'))
    master_dark_path = os.path.join(folder_path, 'Calibration', 'Dark', 'Master_dark.fit')
    m_dark = master_dark(darkfiles, m_bias, master_dark_path) 
        
   # Create master flat file

    for filter in ['Red', 'Blue', 'Visual']:
        flatfiles = glob.glob(os.path.join(folder_path, 'Calibration', 'Flat', filter ,'PBH*.fit'))
        master_flat_path = os.path.join(folder_path, 'Calibration', 'Flat', filter , 'Master_flat.fit')
        master_flat(flatfiles, m_bias, m_dark, master_flat_path) 

    # Reduce object files
    for filter in ['Red', 'Blue', 'Visual']:
        master_flatfile = os.path.join(folder_path, 'Calibration', 'Flat', filter , 'Master_flat.fit')
        objfiles = glob.glob(os.path.join(folder_path, object_name, filter ,'CSJ*.fit'))

        for objfile in objfiles:
            outputfile = os.path.join(folder_path, object_name, filter)
            reduce_image(objfile, m_bias, m_dark, master_flatfile, outputfile)

    # Compute offsets between object images, and shift, pad, and stack them
    for filter in ['Red', 'Blue', 'Visual']:
        fg = coordinates[object_name][filter]['fg']
        bg = coordinates[object_name][filter]['bg']
        
        nref = 0 # Simple placeholder - Ideally choose img with best signal to noise ratio
        objfiles = glob.glob(os.path.join(folder_path, object_name, filter ,'fdb_*.fit'))
        offsets = []
        padding = 0
        for objfile in objfiles:
            (dx, dy) = measure_offset(objfile, objfiles[nref], fg, bg)
            offsets.append((dx, dy))
            print(f'{objfile} offset by ({dx}, {dy})')
            padding = int(math.ceil(max([padding, dx, dy])))
        outputfile = os.path.join(folder_path, object_name, filter)
        image_registraion(objfiles, offsets, padding, outputfile)
    
    # The following portion of the function Aligns stacked images of all three filters and writes the shifted file
    
    images = []
    for filter in ['Red', 'Blue', 'Visual']:
        images.append(os.path.join(folder_path, object_name, filter, 'Stacked_image.fit'))
    
    stacked_files = []
    sizes = []
    hdr = []
    
    # populating the declared lists above
    for i,img in enumerate(images):
        hdr.append(fits.getheader(images[i]))
        stacked_files.append(fits.getdata(img))
        sizes.append(stacked_files[i].shape[0])
        
    # Padding the stacks of each filter because they have different sizes each and writing the padded files
    for i in range(len(stacked_files)):
        padded_img = np.pad(stacked_files[i], (int)((np.max(sizes)-sizes[i])/2), 'constant', constant_values = -1)
        filter = hdr[i]['FILTER']
        outputfile_name = os.path.join(folder_path, object_name,'padded_'+filter+'_stacked_image.fit')
        fits.writeto(outputfile_name, padded_img, header=hdr[i], output_verify='exception', overwrite=True, checksum=False)

    # coordinates to align the stacked images
    fg_V = coordinates[object_name]["Visual"]['fg']
    bg_V = coordinates[object_name]["Visual"]['bg']

    # Shift the padded stacked files to align them
    files = glob.glob(os.path.join(folder_path, object_name, 'padded*.fit'))
    
    #Following 3 lines of code should be ideally used but we're manually inputing offsets to perfectly align our images
    #offsets = []
    #for img in files:
    #    offsets.append(measure_offset(img, files[0], fg_V, bg_V))

    offsets = [(60, -34), (34,-8), (0,0)]

    for ii in range(len(files)):
        file = files[ii]
        offset = offsets[ii]
        img = fits.getdata(file)
        hdr = fits.getheader(file)
        img[np.isnan(img)] = 0.0
        img[np.isinf(img)] = 0.0
        
        shifted_img = shift(img, offset, cval = -1)
        shifted_img[np.isnan(shifted_img)] = 0.0
        shifted_img[np.isinf(shifted_img)] = 0.0
        #Writing files for the shifted images
        outputfile_name = os.path.join(folder_path, object_name, 's_' + os.path.basename(file))
        fits.writeto(outputfile_name, shifted_img, header=hdr, output_verify='exception', overwrite=True, checksum=False)
        print(f"Shifted image saved as: {os.path.basename(outputfile_name)}")
    
    return

if __name__ == "__main__":
    datafolder = input("Folder path: ")
    objname = input("Object name: ")
    reduction(datafolder, objname)