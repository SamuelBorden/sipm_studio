"""
Processors for finding the baseline of input waveforms
"""
from scipy.stats import linregress
import matplotlib.pyplot as plt
import numpy as np

from sipm_studio.dsp.qpe_peak_finding import (
    fit_peak,
    gaussian,
    guess_peaks_no_width,
)

# define physical constants
e_charge = 1.6e-19
h = 6.64e-34
c = 3e8

# Define some fit global parameters that help peak finding converge

BL_HIST_MIN_HEIGHT = 100  # minimum height in the bl slope histogram to look for a peak
BL_HIST_MIN_DISTANCE = (
    50  # minimum distance in the bl slope histogram to look for peaks
)
BL_HIST_FIT_WIDTH = 10  # width of the peak in the bl slope histogram to fit

INTERCEPT_HIST_MIN_HEIGHT = 9  # height of the peak in the intercept histogram to fit
INTERCEPT_HIST_MIN_DISTANCE = (
    0.3e-5  # minimum distance in the intercept histogram to fit
)
INTERCEPT_HIST_FIT_WIDTH = 30  # width of the peak in the intercept histogram to fit


def find_bl(wfs_in: np.array, bl_idx: int) -> np.array:
    """
    On an input set of waveforms, calculate the baseline, and then subtract the baseline from the waveforms.
    This works by finding the waveforms whose first bl_idx samples have roughly 0 slope, then treats
    the average of these waveform's y-intercepts as the average baseline to subtract off from all waveforms

    Parameters
    ----------
    wfs_in
        An array of arrays containing waveform data
    bl_idx
        An integer up to which to evaluate the slope of each waveform
    """

    # Find the slope of the first bl_idx samples
    bl_slopes = []
    bl_intercepts = []

    for wf in wfs_in:
        slope, intercept, *_ = linregress(np.arange(0, bl_idx), wf[:bl_idx])
        bl_slopes.append(slope)
        bl_intercepts.append(intercept)

    bl_slopes = np.array(bl_slopes)
    bl_intercepts = np.array(bl_intercepts)
    n, bins, _ = plt.hist(bl_slopes, bins=200)
    plt.clf()

    # guess and fit the 0 slope peak
    peaks, peak_locs, amplitudes = guess_peaks_no_width(
        n, bins, BL_HIST_MIN_HEIGHT, BL_HIST_MIN_DISTANCE
    )
    peaks = peaks[:1]
    peak_locs = peak_locs[:1]
    amplitudes = amplitudes[:1]

    gauss_params, gauss_errors = fit_peak(
        n, bins, peaks, peak_locs, amplitudes, fit_width=BL_HIST_FIT_WIDTH
    )

    x = np.linspace(bins[0], bins[-1], 500)

    plt.figure(figsize=(12, 8))
    for i, params in enumerate(gauss_params):
        plt.plot(x, gaussian(x, *params))
    n, bins, patches = plt.hist(bl_slopes, bins=200)
    n, bins, patches = plt.hist(bl_slopes, bins=200, histtype="stepfilled", alpha=0.5)
    plt.xlabel("Integrated Charge (C)")
    plt.ylabel("Counts")
    plt.ylim(1, np.amax(n))
    plt.yscale("log")
    plt.show()

    # Now we have the location of 0 slope, we can pull waveforms. We want slope 0 waveform's y-intercept to use as baseline
    avg_slope = gauss_params[0][1]
    slope_std = np.abs(gauss_params[0][2])

    std_cut = 3  # 99% of waveforms with 0 slope should be in this cut...

    bl_values = bl_intercepts[
        (bl_slopes <= avg_slope + std_cut * slope_std)
        & (bl_slopes >= avg_slope - std_cut * slope_std)
    ]
    n, bins, _ = plt.hist(bl_values, bins=200)
    plt.clf()

    # guess and fit the baseline peak
    peaks, peak_locs, amplitudes = guess_peaks_no_width(
        n, bins, INTERCEPT_HIST_MIN_HEIGHT, INTERCEPT_HIST_MIN_DISTANCE
    )

    peaks = peaks[:2]
    peak_locs = peak_locs[:2]
    amplitudes = amplitudes[:2]

    gauss_params, gauss_errors = fit_peak(
        n, bins, peaks, peak_locs, amplitudes, fit_width=INTERCEPT_HIST_FIT_WIDTH
    )

    x = np.linspace(bins[0], bins[-1], 500)

    plt.figure(figsize=(12, 8))
    for i, params in enumerate(gauss_params):
        plt.plot(x, gaussian(x, *params))
    n, bins, patches = plt.hist(bl_values, bins=200)
    n, bins, patches = plt.hist(bl_values, bins=200, histtype="stepfilled", alpha=0.5)
    plt.xlabel("Integrated Charge (C)")
    plt.ylabel("Counts")
    plt.ylim(1, np.amax(n))
    plt.yscale("log")
    plt.show()

    avg_bl = gauss_params[0][1]
    bl_std = np.abs(gauss_params[0][2])
    if len(gauss_params) > 1:

        avg_bl_2 = gauss_params[1][1]
        bl_std_2 = np.abs(gauss_params[1][2])

    # This code is used in the case of processing LN data where there is very little baseline jitter
    # std_cut = 3
    # if device in ["sipm", "sipm_mini", "sipm_1st"]:
    # if False:
    #     bls = []
    #     for x in wfs_in:
    #         bls.append(np.full_like(x,np.mean(x[:bl_idx])))
    # #             bls.append(np.full_like(x,bl))

    #     wf_bl = np.array(wfs_in)-np.array(bls)

    # else:
    #     bls = []
    #     for x in wfs_in:
    #         bl = np.min([np.mean(x[:bl_idx]), np.mean(x[-bl_idx:])])
    #         if (bl >= avg_bl + std_cut*bl_std) or (bl <= avg_bl - std_cut*bl_std):
    #             noisy_counts += 1
    #             bls.append(np.full_like(x,avg_bl))

    #         else:
    # #             bls.append(np.full_like(x,avg_bl_2))
    #             bls.append(np.full_like(x,bl))

    ## Try subtracting off the average baseline and see what happens
    wf_bl = np.array(wfs_in) - avg_bl
    #     wf_bl = np.array(wfs_in)- np.array(bls)
    ## what happens if we don't even do baseline subtraction
    # wf_bl = np.array(wfs_in)
    wf_bl = np.array(wf_bl)
    return wf_bl
