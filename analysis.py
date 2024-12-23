# The standard fare, plus a few extra packages:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import os.path
import math

# Newer packages:
from astropy.stats import mad_std
from astropy.stats import sigma_clip
from photutils.utils import calc_total_error
import astropy.stats as stat

from photutils.aperture import aperture_photometry, ApertureStats, CircularAperture, CircularAnnulus
from photutils.detection import DAOStarFinder

from IPython.display import Image
import matplotlib.image as mpimg

# Star extraction function
def starExtractor(fitsfile, nsigma_value, fwhm_value):
    """
    Takes an image and outputs a list of all star coordinates above the defined threshold.
    Inputs:
        fitsfile (string): path to fits file
        nsigma_value (int): number of standard deviations away to use for a threshold
        fwhm_value (float): width of aperture for locating stars
    Outputs:
        xpos, ypos (arrays): arrays of centroid x and y positions of stars
    """

    # First, check if the region file exists yet, so it doesn't get overwritten
    regionfile = fitsfile.split(".")[0] + ".reg"

    if os.path.exists(regionfile) == True:
        print(regionfile, "already exists in this directory. Rename or remove the .reg file and run again.")
        return

    # get data of fitsfile
    image = fits.getdata(fitsfile)

    # *** Measure the median absolute standard deviation of the image: ***
    bkg_sigma = mad_std(image[~np.isnan(image)])

    # *** Define the parameters for DAOStarFinder ***
    daofind = DAOStarFinder(fwhm=fwhm_value, threshold=nsigma_value*bkg_sigma)

    # Apply DAOStarFinder to the image
    sources = daofind(image)
    nstars = len(sources)
    print("Number of stars found in ",fitsfile,":", nstars)

    # Define arrays of x-position and y-position
    xpos = np.array(sources['xcentroid'])
    ypos = np.array(sources['ycentroid'])

    # Write the positions to a .reg file based on the input file name
    if os.path.exists(regionfile) == False:
        f = open(regionfile, 'w')
        for i in range(0,len(xpos)):
            f.write('circle '+str(xpos[i])+' '+str(ypos[i])+' '+str(fwhm_value)+'\n')
        f.close()
        print("Wrote ", regionfile)
    
    return xpos, ypos # Return the x and y positions of each star as variables

#Aperture photometry
def bg_error_estimate(fitsfile):
    """
    Estimates background error in a fits file.
    Input:
        fitsfile (string): path to fits file
    Output:
        error_image (array): array of pixel errors
    """
    fitsdata = fits.getdata(fitsfile)
    hdr = fits.getheader(fitsfile)

    # What is happening in the next step? Read the docstring for sigma_clip.
    # Answer: removes data a certain number of standard deviations from the mean (ie keeps only the centered data)
    filtered_data = sigma_clip(fitsdata, sigma=3.,copy=False)

    # Summarize the following steps. Answer:
    # Fills empty indicies in array with Nans
    # Makes array were each pixel is square root of previous pixel of last array
    # Sets NaN values equal to the median of the non-NaN values
    bkg_values_nan = filtered_data.filled(fill_value=np.nan)
    bkg_error = np.sqrt(bkg_values_nan)
    bkg_error[np.isnan(bkg_error)] = np.nanmedian(bkg_error)

    print("Writing the background-only error image: ", fitsfile.split('.')[0]+"_bgerror.fit")
    fits.writeto(fitsfile.split('.')[0]+"_bgerror.fit", bkg_error, hdr, overwrite=True)

    effective_gain = 1.4 # electrons per ADU

    error_image = calc_total_error(fitsdata, bkg_error, effective_gain)

    print("Writing the total error image: ", fitsfile.split('.')[0]+"_error.fit")
    fits.writeto(fitsfile.split('.')[0]+"_error.fit", error_image, hdr, overwrite=True)

    return error_image

# Photometry function
def measurePhotometry(fitsfile, star_xpos, star_ypos, aperture_radius, sky_inner, sky_outer, error_array):
    """
    This function makes a table with star positions, sums for flux for stars, uncertainty in the sums, and bg subtracted star counts.
    
    Parameters:
    fitsfile (str): path to the image
    star_xpos (array): x positions of the stars
    star_ypos (array): y positions of the stars
    aperture_radius (float): aperture radius
    sky_inner (float): Inner radius of the sky annulus
    sky_outter (float): Outter radius of the sky annulus
    error_array (array): an array containing the total error
    
    Returns: a table containing photometry information
    """
    # *** Read in the data from the fits file:
    image = fits.getdata(fitsfile)

    star_pos = np.vstack([star_xpos, star_ypos]).T

    starapertures = CircularAperture(star_pos,r = aperture_radius)
    skyannuli = CircularAnnulus(star_pos, r_in = sky_inner, r_out = sky_outer)
    phot_apers = [starapertures, skyannuli]

    # What is new about the way we're calling aperture_photometry?
    # This is accounting for the annulus and including error
    phot_table = aperture_photometry(image, phot_apers, error=error_array)

    # Calculate mean background in annulus and subtract from aperture flux
    bkg_mean = phot_table['aperture_sum_1'] / skyannuli.area
    bkg_starap_sum = bkg_mean * starapertures.area
    final_sum = phot_table['aperture_sum_0']-bkg_starap_sum
    phot_table['bg_subtracted_star_counts'] = final_sum

    # Calculating mean and sum error for the background using annuli area and aperture area
    bkg_mean_err = phot_table['aperture_sum_err_1'] / skyannuli.area
    bkg_sum_err = bkg_mean_err * starapertures.area

    # Combining errors using error propagation
    phot_table['bg_sub_star_cts_err'] = np.sqrt((phot_table['aperture_sum_err_0']**2)+(bkg_sum_err**2))

    return phot_table

# calibrated dataframe function
def zpcalc(magzp, magzp_err, filtername, dataframe):

    '''
    Calibrates V and R fluxxes and their uncertainties in dataframe.
    Inputs:
        magzp (float): the zeropoint
        magzp_err (float): the error in zeropoint
        filtername (string): filter type
        dataframe (pandas dataframe): star dataframe
    Outputs:
        dataframe (pandas dataframe): dataframe with added calibrated magnitude and error columns
    '''

    #extract column data

    maginst = dataframe[filtername + 'inst']
    maginst_err = dataframe[filtername + 'inst_err']

    #calculate magnitude

    calibrated_mag = maginst + magzp
    calibrated_mag_err = np.sqrt(maginst_err**2 + magzp_err**2)

    #add to dataframe

    dataframe = dataframe.assign(**{filtername + 'mag' : calibrated_mag})
    dataframe = dataframe.assign(**{filtername + 'mag_err' : calibrated_mag_err})


    #dataframe.insert(0, filtername + 'mag', calibrated_mag)
    #dataframe.insert(1, filtername + 'mag_err', calibrated_mag_err)

    return dataframe


avg_fwhm = np.mean([9,9,10.5,11,9])

# Star position for the cluster
# we want around 500 stars so we choose nsignma=15
m52_fwhm = avg_fwhm
m52_nsigma = 15

#star position for std star
# we want around 500 stars, 4-sigma is too many, so we choose nsignma=5
std_fwhm = 15
std_nsigma = 5

def make_catalog(folder_path, m52_fwhm=m52_fwhm, m52_nsigma=m52_nsigma, std_fwhm=std_fwhm, std_nsigma=std_nsigma):
    '''
    Takes staked images of the cluster and the std star, performs aperture photometry and bg error analysis
    and writes csv files of the photometry tables.
    Parameters:
    folder_path(str) : path to the main folder
    (Optional) 
    (int) fwhmn of the cluster and std star
    (int) nsigma of the cluster and std star
    return: none
    '''

    v_stacked = os.path.join(folder_path, 'target/s_padded_Visual_stacked_image.fit')
    b_stacked = os.path.join(folder_path, 'target/s_padded_Blue_stacked_image.fit')
    r_stacked = os.path.join(folder_path, 'target/s_padded_Red_stacked_image.fit')
    
    std_v_stacked = os.path.join(folder_path, 'standard/s_padded_Visual_stacked_image.fit')
    std_b_stacked = os.path.join(folder_path, 'standard/s_padded_Blue_stacked_image.fit')
    std_r_stacked = os.path.join(folder_path, 'standard/s_padded_Red_stacked_image.fit')
    
    #Extracting cluster stars position
    x_pos, y_pos = starExtractor(v_stacked, m52_nsigma, m52_fwhm)
    #Extracting std star stars position
    std_x_pos, std_y_pos = starExtractor(std_v_stacked, std_nsigma, std_fwhm)

    # Extracting photometry for the standard star images
    # Measure the background of the image (V)
    std_V_bgerror = bg_error_estimate(std_v_stacked)
    std_V_phottable = measurePhotometry(std_v_stacked, star_xpos=std_x_pos, star_ypos=std_y_pos, \
                                        aperture_radius=15, sky_inner=18, sky_outer=23, error_array=std_V_bgerror)
    
    # Measure the background of the R image
    std_R_bgerror = bg_error_estimate(std_r_stacked)
    std_R_phottable = measurePhotometry(std_r_stacked, star_xpos=std_x_pos, star_ypos=std_y_pos, \
                                        aperture_radius=15, sky_inner=18, sky_outer=23, error_array=std_R_bgerror)
    
    # Measure the background of the B image
    std_B_bgerror = bg_error_estimate(std_b_stacked)
    std_B_phottable = measurePhotometry(std_b_stacked, star_xpos=std_x_pos, star_ypos=std_y_pos, \
                                        aperture_radius=15, sky_inner=18, sky_outer=23, error_array=std_B_bgerror)
    
    #Extracting photometry for the cluster images
    # Measure the background of the image (V)
    m52_V_bgerror = bg_error_estimate(v_stacked)
    m52_V_phottable = measurePhotometry(v_stacked, star_xpos=x_pos, star_ypos=y_pos, \
                                        aperture_radius=10, sky_inner=11, sky_outer=16, error_array=m52_V_bgerror)
    # Measure the background of the image (R)
    m52_B_bgerror = bg_error_estimate(b_stacked)
    m52_B_phottable = measurePhotometry(b_stacked, star_xpos=x_pos, star_ypos=y_pos, \
                                        aperture_radius=10, sky_inner=11, sky_outer=16, error_array=m52_B_bgerror)
    # Measure the background of the image (B)
    m52_R_bgerror = bg_error_estimate(r_stacked)
    m52_R_phottable = measurePhotometry(r_stacked, star_xpos=x_pos, star_ypos=y_pos, \
                                        aperture_radius=10, sky_inner=11, sky_outer=16, error_array=m52_R_bgerror)
    
    # Making photometry tables for the cluster and std star
    columns = ['id','xcenter', 'ycenter','Bflux','Bfluxerr','Vflux','Vfluxerr','Rflux','Rfluxerr']
    std_fluxtable = pd.DataFrame(
        {'id'      : std_V_phottable['id'],
         'xcenter' : std_V_phottable['xcenter'],
         'ycenter' : std_V_phottable['ycenter'],
         'Bflux'   : std_B_phottable['bg_subtracted_star_counts'],
         'Bfluxerr': std_B_phottable['bg_sub_star_cts_err'], 
         'Vflux'   : std_V_phottable['bg_subtracted_star_counts'],
         'Vfluxerr': std_V_phottable['bg_sub_star_cts_err'], 
         'Rflux'   : std_R_phottable['bg_subtracted_star_counts'],
         'Rfluxerr': std_R_phottable['bg_sub_star_cts_err']}, columns=columns)
    
    m52_fluxtable = pd.DataFrame(
        {'id'      : m52_V_phottable['id'],
         'xcenter' : m52_V_phottable['xcenter'],
         'ycenter' : m52_V_phottable['ycenter'],
         'Bflux'   : m52_B_phottable['bg_subtracted_star_counts'],
         'Bfluxerr': m52_B_phottable['bg_sub_star_cts_err'], 
         'Vflux'   : m52_V_phottable['bg_subtracted_star_counts'],
         'Vfluxerr': m52_V_phottable['bg_sub_star_cts_err'], 
         'Rflux'   : m52_R_phottable['bg_subtracted_star_counts'],
         'Rfluxerr': m52_R_phottable['bg_sub_star_cts_err']}, columns=columns)
    
    # Normalize by exp time
    header_m52V= fits.getheader(v_stacked)
    header_m52R = fits.getheader(r_stacked)
    header_m52B = fits.getheader(b_stacked)
    
    header_stdR = fits.getheader(std_r_stacked)
    header_stdV = fits.getheader(std_v_stacked)
    header_stdB = fits.getheader(std_b_stacked)
    
    # new m52 flux
    m52_fluxtable["Bflux_1sec"] = m52_fluxtable["Bflux"] / header_m52B['EXPTIME']
    m52_fluxtable["Vflux_1sec"] = m52_fluxtable["Vflux"] / header_m52V['EXPTIME']
    m52_fluxtable["Rflux_1sec"] = m52_fluxtable["Rflux"] / header_m52R['EXPTIME']
    # new uncert
    m52_fluxtable["Bflux_1sec_err"] = m52_fluxtable["Bfluxerr"] / header_m52B['EXPTIME']
    m52_fluxtable["Vflux_1sec_err"] = m52_fluxtable["Vfluxerr"] / header_m52V['EXPTIME']
    m52_fluxtable["Rflux_1sec_err"] = m52_fluxtable["Rfluxerr"] / header_m52R['EXPTIME']
    
    # new std flux
    std_fluxtable["Bflux_1sec"] = std_fluxtable["Bflux"] / header_stdB['EXPTIME']
    std_fluxtable["Vflux_1sec"] = std_fluxtable["Vflux"] / header_stdV['EXPTIME']
    std_fluxtable["Rflux_1sec"] = std_fluxtable["Rflux"] / header_stdR['EXPTIME']
    # new std uncert
    std_fluxtable["Bflux_1sec_err"] = std_fluxtable["Bfluxerr"] / header_stdB['EXPTIME']
    std_fluxtable["Vflux_1sec_err"] = std_fluxtable["Vfluxerr"] / header_stdV['EXPTIME']
    std_fluxtable["Rflux_1sec_err"] = std_fluxtable["Rfluxerr"] / header_stdR['EXPTIME']
    
    # Calculate instrumental magnitudes
    m52_fluxtable["Binst"] = -2.5*(np.log10(m52_fluxtable["Bflux_1sec"]))
    m52_fluxtable["Vinst"] = -2.5*(np.log10(m52_fluxtable["Vflux_1sec"]))
    m52_fluxtable["Rinst"] = -2.5*(np.log10(m52_fluxtable["Rflux_1sec"]))
    
    std_fluxtable["Binst"] = -2.5*(np.log10(std_fluxtable["Bflux_1sec"]))
    std_fluxtable["Vinst"] = -2.5*(np.log10(std_fluxtable["Vflux_1sec"]))
    std_fluxtable["Rinst"] = -2.5*(np.log10(std_fluxtable["Rflux_1sec"]))
    
    #Propagate errors
    m52_fluxtable["Binst_err"] = 2.5 * 0.434 * (m52_fluxtable["Bflux_1sec_err"]/m52_fluxtable["Bflux_1sec"])
    m52_fluxtable["Vinst_err"] = 2.5 * 0.434 * (m52_fluxtable["Vflux_1sec_err"]/m52_fluxtable["Vflux_1sec"])
    m52_fluxtable["Rinst_err"] = 2.5 * 0.434 * (m52_fluxtable["Rflux_1sec_err"]/m52_fluxtable["Rflux_1sec"])
    
    std_fluxtable["Binst_err"] = 2.5 * 0.434 * (std_fluxtable["Bflux_1sec_err"]/std_fluxtable["Bflux_1sec"])
    std_fluxtable["Vinst_err"] = 2.5 * 0.434 * (std_fluxtable["Vflux_1sec_err"]/std_fluxtable["Vflux_1sec"])
    std_fluxtable["Rinst_err"] = 2.5 * 0.434 * (std_fluxtable["Rflux_1sec_err"]/std_fluxtable["Rflux_1sec"])
    
    # used the following step to get standard star location on the table
    target_value = 1756
    tolerance = 10
    
    indices = std_fluxtable.index[(std_fluxtable['xcenter'] >= target_value - tolerance) & (std_fluxtable['xcenter'] <= target_value + tolerance)].tolist()
    
    loc = 75
    
    #standard star magnitudes from SIMBAD
    real_V_mag = 8.951
    real_R_mag = 8.475
    real_B_mag = 9.817
    
    #calculate the zeropoints for each band
    magzp_B = real_B_mag - std_fluxtable['Binst'][loc]
    magzp_B_error = np.sqrt((std_fluxtable['Binst_err'][loc]**2 + 0.004**2))
    magzp_V = real_V_mag - std_fluxtable['Vinst'][loc]
    magzp_V_error = np.sqrt((std_fluxtable['Vinst_err'][loc]**2 + 0.004**2))
    magzp_R = real_R_mag - std_fluxtable['Rinst'][loc]
    magzp_R_error = np.sqrt((std_fluxtable['Rinst_err'][loc]**2 + 0.004**2))
    
    # Apply calibrate df function to dataframes
    m52_fluxtable = zpcalc(magzp_B, magzp_B_error, "B", m52_fluxtable)
    m52_fluxtable = zpcalc(magzp_V, magzp_V_error, "V", m52_fluxtable)
    m52_fluxtable = zpcalc(magzp_R, magzp_R_error, "R", m52_fluxtable)
    
    # Add V-R column and V-R error column
    m52_fluxtable["V-R"] = m52_fluxtable["Vmag"] - m52_fluxtable["Rmag"]
    m52_fluxtable["V-R_err"] = np.sqrt((m52_fluxtable["Vmag_err"])**2 + (m52_fluxtable["Rmag_err"])**2)
    # Add B-V column and B-V error column
    m52_fluxtable["B-V"] = m52_fluxtable["Bmag"] - m52_fluxtable["Vmag"]
    m52_fluxtable["B-V_err"] = np.sqrt((m52_fluxtable["Vmag_err"])**2 + (m52_fluxtable["Bmag_err"])**2)
    
    # Finally, save both calibrated dataframes (standard and m52) here as .csv files;
    # These can later be read into Excel, Google Sheets, back into pandas, etc. for future use
    std_fluxtable.to_csv('std_star.csv')
    m52_fluxtable.to_csv('M52_photometry.csv')

    return

def make_cmd(m_52_photometry, path_to_isochrone):
    '''
    Makes V-R and B-V CMD for a cluster in a single figure with subplots
    Input 
    m_52_photometry(.csv): takes a csv file of the star catalog
    path_to_isochrone (str): path to the isochrone needed to fit
    returns none
    '''

    isochrone = pd.read_table(path_to_isochrone, sep='\\s+', skiprows=10)
    m52_stars = pd.read_csv(m_52_photometry)

    # Prepare V-R data
    m52_stars_cut_vr = (m52_stars[m52_stars['Vmag'] < 16])[m52_stars['V-R'] > -3]
    x_vr = m52_stars_cut_vr['V-R']
    y_vr = m52_stars_cut_vr['Vmag']

    # Prepare B-V data
    m52_stars_cut_bv = m52_stars[m52_stars['Vmag'] < 16]
    x_bv = m52_stars_cut_bv['B-V']
    y_bv = m52_stars_cut_bv['Vmag']

    # Create a single figure with subplots: 2x2 + 1
    fig, axs = plt.subplots(3, 2, figsize=(14, 18), constrained_layout=True)
    
    # ---- First Row ----
    # Plot 1: V-R Scatter Plot
    pts_vr = axs[0, 0].scatter(x_vr, y_vr, alpha=0.5, edgecolors='black', linewidths=0.5, c=x_vr, cmap='twilight_r')
    cb1 = fig.colorbar(pts_vr, ax=axs[0, 0])
    cb1.set_label('V-R', fontsize=14)
    axs[0, 0].invert_yaxis()
    axs[0, 0].set_xlabel('V-R')
    axs[0, 0].set_ylabel('V mag')
    axs[0, 0].set_title('M52 CMD (V-R Scatter)')

    # Plot 2: V-R Errorbar Plot
    axs[0, 1].errorbar(
        x_vr, y_vr, 
        xerr=m52_stars_cut_vr["V-R_err"],
        yerr=m52_stars_cut_vr["Vmag_err"], 
        marker='o', linestyle='None', alpha=0.5, 
        markerfacecolor='#d9b2fb', markeredgecolor='black',
        markeredgewidth=0.5, markersize=5, ecolor='#2b0c46', capsize=3
    )
    axs[0, 1].invert_yaxis()
    axs[0, 1].set_xlabel('V-R')
    axs[0, 1].set_ylabel('V mag')
    axs[0, 1].set_title('M52 CMD (V-R Errorbar)')

    # ---- Second Row ----
    # Plot 3: B-V Scatter Plot
    pts_bv = axs[1, 0].scatter(x_bv, y_bv, alpha=0.5, edgecolors='black', linewidths=0.5, c=x_bv, cmap='twilight_r')
    cb2 = fig.colorbar(pts_bv, ax=axs[1, 0])
    cb2.set_label('B-V', fontsize=14)
    axs[1, 0].invert_yaxis()
    axs[1, 0].set_xlabel('B-V')
    axs[1, 0].set_ylabel('V mag')
    axs[1, 0].set_title('M52 CMD (B-V Scatter)')

    # Plot 4: B-V Errorbar Plot
    axs[1, 1].errorbar(
        x_bv, y_bv, 
        xerr=m52_stars_cut_bv["B-V_err"],
        yerr=m52_stars_cut_bv["Vmag_err"], 
        marker='o', linestyle='None', alpha=0.5, 
        markerfacecolor='#b7d5ff', markeredgecolor='black',
        markeredgewidth=0.5, markersize=5, ecolor='#150966', capsize=3
    )
    axs[1, 1].invert_yaxis()
    axs[1, 1].set_xlabel('B-V')
    axs[1, 1].set_ylabel('V mag')
    axs[1, 1].set_title('M52 CMD (B-V Errorbar)')

    # ---- Third Row ----
    # Plot 5: Isochrone Fit CMD
    axs[2, 0].scatter(x_bv, y_bv, label='M52', alpha=0.5, edgecolors='black', linewidths=0.5, c=x_bv, cmap='twilight_r')
    axs[2, 0].plot(isochrone['B']-isochrone['V']+1.3, isochrone['V']+10.7, label='1e8 yr Isochrone', color='xkcd:tealish')
    axs[2, 0].invert_yaxis()
    axs[2, 0].set_xlabel('B-V')
    axs[2, 0].set_ylabel('V mag')
    axs[2, 0].set_title('M52 CMD with Isochrone Fit')
    axs[2, 0].legend()

    # Remove empty subplot in (2,1)
    fig.delaxes(axs[2, 1])

    # Show the figure
    plt.show()

    # ---- Chi^2 Calculation ----
    isochrone_shift_cut = isochrone[isochrone['V'] > 1.5]
    isochrone_shift_cut = isochrone_shift_cut.assign(B_V=isochrone_shift_cut['B'] - isochrone_shift_cut['V'])
    isochrone_shift_cut['B_V'] = isochrone_shift_cut['B_V'] + 1.3

    chi_sqfit = ((y_bv - isochrone_shift_cut['V']) / m52_stars_cut_bv['Vmag_err']) ** 2
    chi_sqfit = np.sum(chi_sqfit)
    print('$\chi^{2}$ for 1e8 MS is', chi_sqfit)

    # Distance calculation
    d = 10 ** ((10.7 + 5) / 5)
    print('The distance to M52 is', d)
    print('Actual value is 1533')
    percent_err = (d - 1533) / 1533 * 100
    print("Percent error in distance", percent_err, '%')

    return


if __name__ == "__main__":
    datafolder = input("Folder path: ")
    make_catalog(datafolder)
    make_cmd('M52_photometry.csv', os.path.join(datafolder, 'Isochrones/isochrones_marigo08_1e8yr.txt'))