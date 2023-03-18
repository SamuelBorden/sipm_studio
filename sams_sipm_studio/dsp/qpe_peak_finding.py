"""
Peak finding and fitting routines to a charge spectrum, usually obtained through current_to_charge.py functions
"""

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


def gaussian(x, A, mu, sigma):
    return A*np.exp(-(x - mu)**2/(2*sigma**2))

# +
def guess_peaks(n, bins, min_height, min_dist, min_width):
    bin_width = bins[1] - bins[0]
    min_bin_dist = min_dist / bin_width
    min_bin_width = min_width/bin_width
    peaks, amplitudes = find_peaks(n, height=min_height, distance=min_bin_dist, width = min_bin_width)
    return peaks, bins[peaks], amplitudes["peak_heights"]
    
def guess_peaks_no_width(n, bins, min_height, min_dist):
    bin_width = bins[1] - bins[0]
    min_bin_dist = min_dist / bin_width
    peaks, amplitudes = find_peaks(n, height=min_height, distance=min_bin_dist)
    return peaks, bins[peaks], amplitudes["peak_heights"]


# +
def fit_peaks(n, bins, peaks, peak_locs, amplitudes, fit_width=15):
    gauss_params = []
    gauss_errors = []
    bin_centers = (bins[1:] + bins[:-1]) / 2
    sigma_guess = (peak_locs[1] - peak_locs[0])
    for i, peak in enumerate(peaks):
        left = peak-fit_width
        right = peak+fit_width
        if left < 0:
            left = 0
        coeffs, covs = curve_fit(gaussian, bin_centers[left: right], n[left: right], p0=[amplitudes[i], peak_locs[i], sigma_guess])
        gauss_params.append(coeffs)
        gauss_errors.append(np.sqrt(np.diag(covs)))
    return gauss_params, gauss_errors
    
    
def fit_peak(n, bins, peaks, peak_locs, amplitudes, fit_width=15):
    """
    fit one peak only, used in light analysis 
    """
    gauss_params = []
    gauss_errors = []
    bin_centers = (bins[1:] + bins[:-1]) / 2
    sigma_guess = (np.amax(bins))/2
#     sigma_guess = 1e-11
    print(sigma_guess)
    for i, peak in enumerate(peaks):
        left = peak-fit_width
        right = peak+fit_width
        if left < 0:
            left = 0
        coeffs, covs = curve_fit(gaussian, bin_centers[left: right], n[left: right], p0=[amplitudes[i], peak_locs[i], sigma_guess])
        gauss_params.append(coeffs)
        gauss_errors.append(np.sqrt(np.diag(covs)))
    return gauss_params, gauss_errors


# -

def write_to_json(json_name, h5_file_name, gauss_params, gauss_errors):
    head, tail = os.path.split(h5_file_name)
    reproc_dict = None
    with open(json_name, "r") as json_file:
        reproc_dict = json.load(json_file)
        
    gauss_params_np = np.array(gauss_params).T
    gauss_errors_np = np.array(gauss_errors).T
    
    reproc_dict["processes"]["normalize_charge"][tail]["peak_locs"] = list(gauss_params_np[1])
    reproc_dict["processes"]["normalize_charge"][tail]["peak_errors"] = list(gauss_errors_np[1])
    
    with open(json_name, "w") as json_file:
        json.dump(reproc_dict, json_file, indent=4)
