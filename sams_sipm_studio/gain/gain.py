import os, h5py, json
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat, unumpy
import warnings
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

from sams_sipm_studio.dsp.adc_to_current import current_converter
from sams_sipm_studio.dsp.qpe_peak_finding import fit_peaks, guess_peaks, gaussian
from sams_sipm_studio.dsp.current_to_charge import integrate_current, rando_integrate_current
from sams_sipm_studio.dsp.gain_processors import normalize_charge, my_gain, slope_fit_gain

import argparse 

## Create some parser arguments for when the pool process is called. 

# parser = argparse.ArgumentParser(description='Calculate gain of SiPMs')
# parser.add_argument('-c', '--core_num', help='number of cores to distribute files over', type=int, default=1)
# parser.add_argument('-f', '--file', help='json spec file', type=str, default="/home/sjborden/sams_sipm_studio/examples/default_gain.json")

# args = parser.parse_args()
# num_processors = int(args.core_num)
# json_file_path = str(args.file)

# Define some global parameters used to automatically clean the data, and detect and fit peaks.

STD_FIT_WIDTH = 100 # number of bins to fit a gaussian to a std histogram 
SAMPLE_TIME = 2e-9 # convert the DT5730 sampling rate to seconds
num_peaks = 4 # maximum number of peaks to find in a QPE
QPE_FIT_WIDTH = 10 # number of bins to fit around the tallest peak in a QPE to guess the stddev
BIN_STD = 100 # number to multiply the bin width by in the tallest peak fitting to get a convergent fit
MIN_QPE_HEIGHT = 7 # minimum height in counts to find a QPE peak
QPE_FIT_WIDTH = 20 # number of bins to fit the QPE gaussian peaks with 
SIGMA_DISTANCE = 5 # number of standard deviations to set as the minimum distance between QPE peaks
SIGMA_WIDTH = 2 # the number of sigma for minimum peak width in QPE peak finding


def calculate_gain(input_file: str, bias: float, device_name: str, vpp: float, start_idx: int, end_idx: int, output_file: str) -> None:
    """
    For a given file, read in the waveforms and convert them to current waveforms. Then, integrate the waveforms. 
    Apply a peak finding routine to the charge spectrum. Then calculate the gain with a number of calculators. Save the results. 
    
    The peak finding routine works like this: first, find and fit only the tallest peak in the QPE. Use its fit sigma to set 
    the minimum width of peaks to look for in the QPE, and use 3 times the std to set the inter-peak spacing in the search.

    Parameters 
    ----------
    input_file 
        A raw-tier CoMPASS file that contains `raw/waveforms` and `raw/baselines` as keys 
    bias 
        The bias the SiPM was set at during the run
    device_name 
        The name of the device this file corresponds to, one of:
        `sipm`, `sipm_1st`, `sipm_1st_low_gain`, or `apd`
    vpp 
        The voltage division CoMPASS was set to record the DAC at 
    start_idx 
        The start of the trigger set to capture the SiPM pulses, in samples 
    end_idx
        The sample index of where 90% of a SiPM pulse as recovered to baseline 
    output_file 
        The name of the file in which to place the output of the calculated gains

    Notes 
    -----
    The inputs of this function are all determined by :func:`.util.parse_json_config.parse_gain_json`
    """
    # Grab the output path from the filename 
    out_path = "/" + os.path.join(*output_file.split("/")[:-1])
    print(out_path)
    
    
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
    std = np.array(np.std(bls, axis=1, dtype=np.float64))
    
    n, bins = np.histogram(std, bins=100) 
    
    try:
        params, errors = fit_peaks(n, bins, [np.argmax(n)], [bins[0], bins[1]], [np.amax(n)], fit_width =  STD_FIT_WIDTH)
    except: 
        raise ValueError("Standard Deviation fit failed gracefully")
        
    cut_value = params[0][1] + 2 * params[0][2]  # cut from the median plus 2 standard deviations

    bls = np.array(bls)
    std_cut_bls = bls[(std < cut_value)]
    std_cut_wfs = waveformies[(std < cut_value)]
    
    nice_wfs = std_cut_wfs - std_cut_bls
    
    print("saving figure of pulses")
    # Save a figure of the time aligned pulses for visual inspection after analysis
    fig = plt.figure(figsize=(12, 8))
    time = np.arange(0, len(nice_wfs[0]), 1)
    for i in range(0, 500):
        plt.plot(time*SAMPLE_TIME, nice_wfs[i], linewidth=0.1, alpha=0.95)
    plt.xlim([0,(end_idx+150)*SAMPLE_TIME])
    plt.axvline(x=start_idx*SAMPLE_TIME)
    plt.axvline(x=end_idx*SAMPLE_TIME)
    plt.xlabel("Time [ns]")
    plt.ylabel("Current [A]")
    plt.title(f"Current Waveforms for Gain Analysis at {bias}V Reverse Bias")
    fig_path = out_path + "/gain_pulses_" + str(bias) + ".png" 
    plt.savefig(fig_path, dpi=fig.dpi)
    
    # Integrate the pulses 
    qs = integrate_current(nice_wfs, lower_bound=start_idx, upper_bound=end_idx, sample_time=SAMPLE_TIME)
    
    print("saving raw QPE") 
    # Save a figure of the raw QPE spectrum in case it looks bad and needs manual inspection 
    fig = plt.figure(figsize=(12, 8))
    plt.hist(qs, bins=1000)
    plt.yscale('log')
    plt.xlabel('Charge [C]')
    plt.ylabel('Counts')
    plt.title(f"QPE Spectrum for {bias}V Reverse Bias")
    fig_path = out_path + "/QPE_" + str(bias) + ".png" 
    plt.savefig(fig_path, dpi=fig.dpi)
    
    # Here's the hard part: actually do the fitting. First, make the QPE 
    ramge = [-0.02e-12, np.amax(qs)]
    fig = plt.figure(figsize=(12,8))
    n, bins, patches = plt.hist(qs, bins=1000, range=ramge, histtype="step")
    n, bins, patches = plt.hist(qs, bins=1000, range=ramge, histtype="stepfilled", alpha=0.15)
    
    print("about to try fitting tallest peak") 
    # Guess the peak locations 
    # Try fitting a gaussian to the tallest peak 
    try:
        params, errors = fit_peaks(n, bins, [np.argmax(n)], [bins[0], BIN_STD*bins[1]], [np.amax(n)], fit_width =  QPE_FIT_WIDTH)
    except: 
        raise ValueError("Tallest Peak Fitting Failed, exiting gracefully")

    sigma_guess = np.abs(params[0][2]) # take the standard deviation as the minimum width of a qpe peak
    min_distance_guess = SIGMA_DISTANCE*sigma_guess # take 3 times the standard deviation as the minimum distance to find peaks
    mind_width_guess = SIGMA_WIDTH*sigma_guess
    # Guess the rest of the peak locations
    try:
        peaks, peak_locs, amplitudes = guess_peaks(n, bins, MIN_QPE_HEIGHT, min_distance_guess, mind_width_guess)
    except: 
        raise ValueError("Finding the rest of the QPE peaks failed, exiting.")
    
    print("success on finding all the peak locations") 
    peaks = peaks[:num_peaks]
    peak_locs = peak_locs[:num_peaks]
    amplitudes = amplitudes[:num_peaks]
    
    # Now try fitting the rest of the peaks 
    try: 
        gauss_params, gauss_errors = fit_peaks(n, bins, peaks, peak_locs, amplitudes, fit_width=QPE_FIT_WIDTH)
    except:
        raise ValueError("Fitting the rest of the QPE peaks failed, exiting.")
    
    print("success on fitting all the peaks, about to save figure") 
    # Save the fit QPE 
    x = np.linspace(-0.02e-12, ramge[1], 1000)
    fig = plt.figure(figsize=(12,8))
    for i, params in enumerate(gauss_params):
        plt.plot(x, gaussian(x, *params))
    n, bins, patches = plt.hist(qs, bins=1000, range=ramge)
    n, bins, patches = plt.hist(qs, bins=1000, range=ramge, histtype="stepfilled",  alpha=0.5)
    plt.xlabel("Integrated Charge (C)")
    plt.ylabel("Counts")
    plt.xlim(ramge)
    plt.ylim(1, np.amax(n))
    plt.yscale("log")
    plt.title(f"Gaussian Fits to QPE for {bias}V Reverse Bias")
    fig_path = out_path + "/QPE_fit_" + str(bias) + ".png" 
    plt.savefig(fig_path, dpi=fig.dpi)
    
    
    # Now calculate the gain
    qs = np.array(qs)
    centroid = []
    centroid_err = []
    for i, x in enumerate(gauss_params):
        centroid.append(x[1])
        centroid_err.append(gauss_errors[i][1])

    nick_gain, n_qs, nick_gain_u = normalize_charge(qs, centroid, centroid_err)
    
    # Save the normalized QPE
    fig = plt.figure(figsize=(12,8))
    plt.hist(n_qs, bins=1000,alpha=0.5)
    plt.axvline(x=0)
    plt.axvline(x=1)
    plt.axvline(x=2)
    plt.axvline(x=3)
    plt.axvline(x=4)
    plt.axvline(x=5)
    plt.axvline(x=6)
    plt.yscale('log')
    plt.xlim([-1,5])

    fig_path = out_path + "/QPE_normalized_" + str(bias) + ".png" 
    plt.savefig(fig_path, dpi=fig.dpi)

    nick_gain = [nick_gain, nick_gain_u]
   
   # Calculate the gain another way 
    my_gains = my_gain(centroid,centroid_err) 
    slope_gain = slope_fit_gain(centroid)
   
   
   # Save it all to a file 

    f = h5py.File(output_file, 'a')

    window = [start_idx, end_idx] # save the parameters used to integrate with

    if f'{bias}/gain' in f: 
        f[f'{bias}/gain'][:] = nick_gain
        f[f'{bias}/my_gain'][:] = my_gains
        f[f'{bias}/window'][:] = window
        f[f'{bias}/slope_gain'][:] = slope_gain
        del f[f'{bias}/charges']
        dset = f.create_dataset(f'{bias}/charges', data=n_qs)

    else: 
        dset = f.create_dataset(f'{bias}/gain', data=nick_gain)
        dset = f.create_dataset(f'{bias}/my_gain', data=my_gains)
        dset = f.create_dataset(f'{bias}/charges', data=n_qs)
        dset = f.create_dataset(f'{bias}/window', data=window)
        dset = f.create_dataset(f'{bias}/slope_gain', data=slope_gain)
    f.close()