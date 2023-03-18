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


### Create a harcoded dictionary of physical hardware settings to let the user select from
setting_dict = {
    "amplifier": {
    "sipm":{
            "functions": ["trans_amp", "voltage_divider", "non_invert_amp", "invert_amp", "voltage_divider"],
            "values": [{"R1": 1.5e3}, {"R1": 49.9, "R2": 49.9}, {"R1": 680, "R2": 75}, {"R1": 750, "R2": 90.9}, {"R1": 49.9, "R2": 49.9}]
        }, 
    "sipm_1st":{
            "functions": ["trans_amp", "voltage_divider"],
            "values": [{"R1": 1.5e3}, {"R1": 49.9, "R2": 49.9}]
        },
    "sipm_1st_low_gain":{
            "functions": ["trans_amp", "voltage_divider"],
            "values": [{"R1": 150}, {"R1": 49.9, "R2": 49.9}] 
        },
    "sipm_low_gain":{
            "functions": ["trans_amp", "voltage_divider", "non_invert_amp", "invert_amp", "voltage_divider"],
            "values": [{"R1": 150}, {"R1": 49.9, "R2": 49.9}, {"R1": 680, "R2": 75}, {"R1": 750, "R2": 90.9}, {"R1": 49.9, "R2": 49.9}]
        }, 
    "apd":{
            "functions": ["trans_amp", "voltage_divider", "non_invert_amp", "invert_amp", "voltage_divider"],
            "values": [{"R1": 2.49e3}, {"R1": 49.9, "R2": 49.9}, {"R1": 499, "R2": 69.8}, {"R1": 499, "R2": 100}, {"R1": 49.9, "R2": 49.9}]
        }
    },

"v_range": {
        "sipm": 2,
        "apd" : 0.5
    }

}

class pc: 
    def voltage_divider(R1, R2):
        return (R2 / (R1 + R2))


    def trans_amp(R1):
        return - R1


    def non_invert_amp(R1, R2):
        return (1 + (R1 / R2))


    def invert_amp(R1, R2):
        return - R1 / R2


# ## create function to compute the amplification of a specific hardware configuration

def _compute_amplification(settings, channel):
    full_amp = 1
    for i, func in enumerate(settings["amplifier"][channel]["functions"]):
        amp_func = getattr(pc, func)
        full_amp *= amp_func(**settings["amplifier"][channel]["values"][i])
    return full_amp

apd_amp = _compute_amplification(setting_dict, "apd")
sipm_amp = _compute_amplification(setting_dict, "sipm")
sipm_amp_1st_stage = _compute_amplification(setting_dict, "sipm_1st")
sipm_amp_1st_stage_low_gain = _compute_amplification(setting_dict, "sipm_1st_low_gain")
sipm_amp_low_gain = _compute_amplification(setting_dict, "sipm_low_gain")

# ## create funciton that uses the above amplifications to convert to current 

def current_converter(waveforms, vpp=2, n_bits=14, device="sipm"):
    amp = _compute_amplification(setting_dict, device)
    return waveforms * (vpp / 2 ** n_bits) * (1 / amp)
