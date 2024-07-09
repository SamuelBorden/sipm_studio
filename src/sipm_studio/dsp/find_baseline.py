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
    tallest_peak_loc_sigma,
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
    bl_stds = []

    for wf in wfs_in:
        slope, intercept, *_ = linregress(np.arange(0, bl_idx), wf[:bl_idx])
        bl_slopes.append(slope)
        bl_intercepts.append(intercept)
        bl_stds.append(np.std(wf[:bl_idx]))

    bl_slopes = np.array(bl_slopes)
    bl_intercepts = np.array(bl_intercepts)
    bl_std = np.abs(np.mean(bl_stds))
    n, bins, _ = plt.hist(bl_slopes, bins=500, range=(-bl_std / 4, bl_std / 4))
    plt.clf()

    peak, sigma = tallest_peak_loc_sigma(bins, n)
    peaks = np.array([np.argmax(n)])
    peak_locs = np.array([peak])
    amplitudes = np.array([np.amax(n)])

    sigma = int(sigma / (bins[1] - bins[0]))  # convert to units of bins

    if len(amplitudes) < 1:
        raise ValueError(f"{len(amplitudes)} length")
    if sigma <= 5:  # make sure sigma is larger than the number of free parameters
        sigma = 6
    #         raise ValueError(f'{sigma}, {amplitudes}, {peaks}, {peak_locs} in baseline first pass failing')

    gauss_params, gauss_errors = fit_peak(
        n, bins, peaks, peak_locs, amplitudes, fit_width=sigma
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
    n, bins, _ = plt.hist(bl_values, bins=500)
    plt.clf()

    # guess and fit the baseline peak
    peak, sigma = tallest_peak_loc_sigma(bins, n)
    peaks = np.array([np.argmax(n)])
    peak_locs = np.array([peak])
    amplitudes = np.array([np.amax(n)])
    sigma_int = int(sigma / (bins[1] - bins[0]))  # convert to units of bins

    if len(amplitudes) < 1:
        raise ValueError(f"{len(amplitudes)} length")

    if sigma_int > 5:
        gauss_params, gauss_errors = fit_peak(
            n, bins, peaks, peak_locs, amplitudes, fit_width=sigma_int
        )
    elif (
        sigma_int <= 5
    ):  # make sure sigma is larger than the number of free parameters, otherwise use the central value
        gauss_params = [[amplitudes[0], peak, sigma]]

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

    ## Try subtracting off the average baseline and see what happens
    wf_bl = np.array(wfs_in) - avg_bl

    wf_bl = np.array(wf_bl)
    return wf_bl
