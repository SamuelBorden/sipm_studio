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
from uncertainties import ufloat
from uncertainties.umath import *

e_charge = 1.6e-19

def normalize_charge(charges, peak_locs, peak_errors):
    x0 = peak_locs[0]
    peak_locs = np.array(peak_locs)[1:]
    

    
    peak_locs_u = []
    for i in range(len(peak_locs)):
        peak_locs_u.append(ufloat(peak_locs[i],peak_errors[i]))
        
        
    peak_locs_u  = np.array(peak_locs_u)

    peak_errors = np.array(peak_errors)[1:]
    
    peak_diffs = peak_locs[1:] - peak_locs[:-1]
    peak_diffs_u = peak_locs_u[1:] - peak_locs_u[:-1]

    
    
    diff_errors = np.sqrt((peak_errors[1:] + peak_errors[:-1])**2)
    gain = np.sum(peak_diffs * diff_errors) / np.sum(diff_errors)
    gain_e = np.sum(diff_errors) / len(diff_errors)

    
    gain_u = np.sum(peak_diffs_u)/len(peak_diffs_u)
    return gain/e_charge, (charges - x0) / gain, gain_e/e_charge


def my_gain(peak_locs, peak_err):
    
    peak_locs_u = []
    for i in range(len(peak_locs)):
        peak_locs_u.append(ufloat(peak_locs[i],peak_err[i]))
    peak_locs_u = np.array(peak_locs_u)
    peak_diffs = np.diff(peak_locs)
    peak_diffs_u = np.diff(peak_locs_u)
    gain = np.sum(peak_diffs)/len(peak_diffs)
    gain_u = np.sum(peak_diffs_u)/len(peak_diffs_u)
    return np.array([gain_u.n/e_charge,gain_u.s/e_charge])


from scipy.stats import linregress
def line(x, m, b):
    return m*x+b

def slope_fit_gain(peak_locs, pedestal = False):
    if pedestal:
        npe = np.arange(0, len(peak_locs))
    else: 
        npe = np.arange(1, len(peak_locs)+1)

    slope, intercept, r, p, se = linregress(npe, peak_locs)

    return np.array([slope/e_charge, se/e_charge])    
