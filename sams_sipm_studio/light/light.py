import glob 
import os, sys, h5py, json, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from uncertainties import ufloat, unumpy
import warnings
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
from tqdm import tqdm
tqdm.pandas() # suppress annoying FutureWarning
import random
from scipy.signal import find_peaks
import numba
import time
import scipy.signal as signal 

from sams_sipm_studio.dsp.adc_to_current import current_converter
from sams_sipm_studio.dsp.qpe_peak_finding import fit_peaks, gaussian, guess_peaks_no_width , fit_peak
from sams_sipm_studio.dsp.current_to_charge import integrate_current, rando_integrate_current
from sams_sipm_studio.dsp.gain_processors import normalize_charge, my_gain, slope_fit_gain
from sams_sipm_studio.dsp.charge_to_photons import sipm_photons, apd_photons 

import argparse 

# parser = argparse.ArgumentParser(description='Calculate gain of SiPMs')
# parser.add_argument('-c', '--core_num', help='number of cores to distribute files over', type=int, default=1)
# parser.add_argument('-f', '--file', help='json spec file', type=str, default="/home/sjborden/sams_sipm_studio/examples/default_gain.json")

# args = parser.parse_args()
# num_processors = int(args.core_num)
# json_file_path = str(args.file)

STD_FIT_WIDTH = 100 # number of bins to fit a gaussian to a std histogram 
SAMPLE_TIME = 2e-9 # convert the DT5730 sampling rate to seconds
BL_START_IDX = 4000 
BL_END_IDX = 4250 # index to start and stop calculating mean values for the baseline 
LIGHT_QPE_HEIGHT = 30
DARK_QPE_HEIGHT = 40


def calculate_light(input_file, bias, device_name, vpp, gain_file, light_window_start_idx, light_window_end_idx, dark_window_start_idx, dark_window_end_idx, output_file):
    """
    For a given file, read in the waveforms and convert them to current waveforms. Then, integrate the waveforms. 
    Apply a peak finding routine to the charge spectrum. Then calculate the gain with a number of calculators. Save the results. 
    
    The peak finding routine works like this: first, find and fit only the tallest peak in the QPE. Use its fit sigma to set 
    the minimum width of peaks to look for in the QPE, and use 3 times the std to set the inter-peak spacing in the search.
    """
    # Grab the output path from the filename 
    out_path = "/" + os.path.join(*output_file.split("/")[:-1])
    print(out_path)
    
    # Read in the gain for the SiPM 
    f = h5py.File(gain_file, 'r')
    n_gains=(f[f'{bias}/gain'][()])
    n_gains_u = ufloat(n_gains[0],n_gains[1])
    f.close()


    
    # Read in the file and convert the waveforms to currents 
    sipm_file = h5py.File(input_file, "r")
    
    waveformies = sipm_file['raw/waveforms'][()]
    bls = sipm_file['raw/baselines'][()]
    
    sipm_file.close()
    
    
    # dark data is usually only taken with second stage amplification, but allow for device specification 
    waveformies = current_converter(waveformies, vpp=vpp, n_bits=14, device=device_name)
    bls = current_converter(bls, vpp=vpp, n_bits=14, device=device_name)
    
    print("performing baseline cuts") 
    # Make sure the baseline standard deviations are ok, select two std from the peak
    std = [] 

    for x in waveformies: 
        std.append(np.std(x[-1000:]))

    n, bins = np.histogram(std, bins=100) 
    
    try:
        params, errors = fit_peaks(n, bins, [np.argmax(n)], [bins[0], bins[1]], [np.amax(n)], fit_width =  STD_FIT_WIDTH)
    except: 
        raise ValueError("Standard Deviation fit failed gracefully")
        
    cut_value = params[0][1] + 5 * params[0][2]  # cut from the median plus 5 standard deviations

    bls = np.array(bls)
    std_cut_bls = bls[(std < cut_value)]
    std_cut_wfs = waveformies[(std < cut_value)]
    
    
    # Calculate the baselines 
    if device_name == 'apd' or device_name == 'apd_goofy':
        bls = []
        for x in waveformies:
            bl = np.min([np.mean(x[BL_START_IDX:BL_END_IDX])])
            bls.append(np.full_like(x, bl))

    if device_name == 'sipm':
        bls = []
        for x in waveformies:
            bl = np.min([np.mean(x[0:90]), np.mean(x[BL_START_IDX:BL_END_IDX])])
            bls.append(np.full_like(x, bl))

    if device_name == 'sipm_1st' or device_name == 'sipm_1st_low_gain' or device_name == 'sipm_1st_low_gain_goofy':
        bls = []
        for x in waveformies:
            bl = np.min(np.mean(x[BL_START_IDX:BL_END_IDX]))
            bls.append(np.full_like(x, bl))


    wf_bl = np.array(waveformies)-np.array(bls)
    bls = np.array(bls, dtype=np.uint16)
    wf_bl = np.array(wf_bl)


    # Integrate over the windows provided 
    c_wfs = wf_bl

    window = [light_window_start_idx, light_window_end_idx]

    print("saving figure of light window")
    fig = plt.figure(figsize = (12,8))
    plt.plot(np.mean(c_wfs, axis=0))
    plt.title(f'Average {device_name} Waveform')
    plt.axvline(x=window[0])
    plt.axvline(x=window[1])
    plt.xlim([window[0]-100,window[1]+100])
    plt.ylabel("Current [A]")
    plt.xlabel("Samples")
    fig_path = out_path + "/light_window_superpulse_" + str(bias) + "_" + str(device_name) + ".png" 
    plt.savefig(fig_path, dpi=fig.dpi)

    dark_window = [dark_window_start_idx, dark_window_end_idx]

    print("saving figure of dark window")
    fig = plt.figure(figsize = (12,8))
    plt.plot(np.mean(c_wfs, axis=0))
    plt.title(f'Average {device_name} Waveform')
    plt.axvline(x=dark_window[0])
    plt.axvline(x=dark_window[1])
    plt.xlim([dark_window[0]-100,dark_window[1]+100])
    fig_path = out_path + "/dark_window_superpulse_" + str(bias) + "_" + str(device_name) +".png" 
    plt.savefig(fig_path, dpi=fig.dpi)

    qs = integrate_current(c_wfs, lower_bound=window[0], upper_bound=window[1])
    qs_dark = integrate_current(c_wfs, lower_bound=dark_window[0], upper_bound=dark_window[1])

    # Histogram and peak find the light and dark counts 
    
    plt.figure()
    n, bins, patches = plt.hist(qs, bins=500, histtype="step")
    n, bins, patches = plt.hist(qs, bins=500, histtype="stepfilled", alpha=0.15)
    plt.xlabel("Integrated Charge (C)")
    plt.ylabel("Counts")
    plt.yscale("log")

    plt.figure()
    n_dark, bins_dark, patches_dark = plt.hist(qs_dark, bins=500, histtype="step")
    n_dark, bins_dark, patches_dark = plt.hist(qs_dark, bins=500, histtype="stepfilled", alpha=0.15)
    plt.xlabel("Integrated Charge (C)")
    plt.ylabel("Counts")
    plt.yscale("log")
    
    
    try:
        peaks, peak_locs, amplitudes = guess_peaks_no_width(n, bins, LIGHT_QPE_HEIGHT, 1e-2)
        peaks_dark, peak_locs_dark, amplitudes_dark = guess_peaks_no_width(n_dark, bins_dark, DARK_QPE_HEIGHT, 2e-12)
        
    except:
        raise ValueError("Peak Finding failed")
        
    # Fit with a Gaussian 
    try:
        gauss_params, gauss_errors = fit_peak(n, bins, peaks, peak_locs, amplitudes, fit_width=10000)
        gauss_params_dark, gauss_errors_dark = fit_peak(n_dark, bins_dark, peaks_dark, peak_locs_dark, amplitudes_dark, fit_width=1000)
    
    except: 
        raise ValueError("Peak Fitting Routine Failed.")
    
    # Save the Gaussian fits for visual inspection 
    x = np.linspace(bins[0], bins[-1], 500)
    x_dark = np.linspace(bins_dark[0], bins_dark[-1], 500)
    
    fig=plt.figure(figsize=(12,8))
    for i, params in enumerate(gauss_params):
        plt.plot(x, gaussian(x, *params))
    n, bins, patches = plt.hist(qs, bins=500)
    n, bins, patches = plt.hist(qs, bins=500, histtype="stepfilled",  alpha=0.5)
    plt.xlabel("Integrated Charge (C)")
    plt.ylabel("Counts")
    plt.ylim(1, np.amax(n))
    plt.yscale("log")
    plt.title(f"Histogram of Light Charges for {bias}V in {device_name}")
    fig_path = out_path + "/light_fit_histogram_" + str(bias) + "_" + str(device_name) +".png" 
    plt.savefig(fig_path, dpi=fig.dpi)


    fig=plt.figure(figsize=(12,8))
    for i, params in enumerate(gauss_params_dark):
        plt.plot(x_dark, gaussian(x_dark, *params))
    n_dark, bins_dark, patches_dark = plt.hist(qs_dark, bins=500)
    n_dark, bins_dark, patches_dark = plt.hist(qs_dark, bins=500, histtype="stepfilled",  alpha=0.5)
    plt.xlabel("Integrated Charge (C)")
    plt.ylabel("Counts")
    plt.ylim(1, np.amax(n_dark))
    plt.yscale("log")
    plt.title(f"Histogram of Dark Charges for {bias}V in {device_name}")
    fig_path = out_path + "/dark_fit_histogram_" + str(bias) + "_" + str(device_name) +".png" 
    plt.savefig(fig_path, dpi=fig.dpi)
    
    # Expand the gaussian parameters into their own arrays 
    centroid = []
    centroid_err = []
    for i,x in enumerate(gauss_params):
        centroid.append(x[1]) # the median from [amplitude, median, sigma]
        centroid_err.append(gauss_errors[i][1]) # this is the pcov error on the median
    centroid_dark = []
    centroid_err_dark = []
    for i,x in enumerate(gauss_params_dark):
        centroid_dark.append(x[1])
        centroid_err_dark.append(gauss_errors[i][1])
        
    # Convert from charge to number of photons
    if device_name == 'sipm' or device_name == 'sipm_1st' or device_name == 'sipm_1st_low_gain' or device_name == 'sipm_1st_low_gain_goofy':
        n_q = sipm_photons(centroid, centroid_err, n_gains_u)

        n_q_dark = sipm_photons(centroid_dark, centroid_err_dark, n_gains_u)

        n_q = np.array([n_q.n, n_q.s])

        n_q_dark = np.array([n_q_dark.n, n_q_dark.s])

        print(n_q, n_q_dark)

    if device_name == 'apd' or device_name == 'apd_goofy':
        n_photons = apd_photons(centroid, centroid_err)
        n_photons_dark = apd_photons(centroid_dark, centroid_err_dark)
        n_photons = np.array([n_photons.n, n_photons.s])
        n_photons_dark = np.array([n_photons_dark.n, n_photons_dark.s])

        print('apd', n_photons, n_photons_dark)
        
        
    # Save it all to a file 
    f = h5py.File(output_file, 'a')

    if device_name == 'sipm' or device_name == 'sipm_1st' or device_name == 'sipm_1st_low_gain' or device_name == 'sipm_1st_low_gain_goofy':

        device = "sipm"

        if f'{device}/{bias}/n_charge' in f:
            f[f'{device}/{bias}/n_charge'][:] = n_q
            f[f'{device}/{bias}/charges'][:] = qs
            f[f'{device}/{bias}/window'][:] = window

            f[f'{device}/dark/{bias}/n_charge'][:] = n_q_dark
            f[f'{device}/dark/{bias}/charges'][:] = qs_dark
            f[f'{device}/dark/{bias}/window'][:] = dark_window

        else: 
            dset = f.create_dataset(f'{device}/{bias}/n_charge', data=n_q)
            dset = f.create_dataset(f'{device}/{bias}/charges', data=qs)
            dset = f.create_dataset(f'{device}/{bias}/window', data=window)

            dset = f.create_dataset(f'{device}/dark/{bias}/n_charge', data=n_q_dark)
            dset = f.create_dataset(f'{device}/dark/{bias}/charges', data=qs_dark)
            dset = f.create_dataset(f'{device}/dark/{bias}/window', data=dark_window)


    if device_name == 'apd':
        device = "apd"

        if f'{device}/{bias}/n_photons' in f:
            f[f'{device}/{bias}/n_photons'][:] = n_photons
            f[f'{device}/{bias}/charges'][:] = qs
            f[f'{device}/{bias}/window'][:] = window
            f[f'{device}/dark/{bias}/n_photons'][:] = n_photons_dark
            f[f'{device}/dark/{bias}/charges'][:] = qs_dark
            f[f'{device}/dark/{bias}/window'][:] = dark_window

        else:
            dset = f.create_dataset(f'{device}/{bias}/n_photons', data=n_photons)
            dset = f.create_dataset(f'{device}/{bias}/charges', data=qs)
            dset = f.create_dataset(f'{device}/{bias}/window', data=window)
            # Read in dark info
            dset = f.create_dataset(f'{device}/dark/{bias}/n_photons', data=n_photons_dark)
            dset = f.create_dataset(f'{device}/dark/{bias}/charges', data=qs_dark)
            dset = f.create_dataset(f'{device}/dark/{bias}/window', data=dark_window)

    f.close()
