import os, h5py, json
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat, unumpy
import warnings
from scipy.stats import linregress

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

from sipm_studio.dsp.adc_to_current import current_converter
from sipm_studio.dsp.find_baseline import find_bl
from sipm_studio.dsp.qpe_peak_finding import (
    fit_peaks,
    gaussian,
    guess_peaks_no_width,
    fit_peak,
    fit_peaks_no_sigma_guess,
    tallest_peak_loc_sigma,
)
from sipm_studio.dsp.current_to_charge import (
    integrate_current,
    rando_integrate_current,
)
from sipm_studio.dsp.gain_processors import (
    normalize_charge,
    my_gain,
    slope_fit_gain,
)
from sipm_studio.dsp.charge_to_photons import sipm_photons, apd_photons

import argparse
from scipy.ndimage import gaussian_filter1d

# Set up some global variables so that automatic Gaussian fitting can work properly

STD_FIT_WIDTH = 100  # number of bins to fit a gaussian to a std histogram
SAMPLE_TIME = 2e-9  # convert the DT5730 sampling rate to seconds
bl_idx = 100  # index to calculate the baseline preceding the led pulse

LIGHT_QPE_HEIGHT = 30  # Minimum height in counts to idnetify a peak in the QPE spectrum
DARK_QPE_HEIGHT = 80

LIGHT_QPE_WIDTH = 2e-13  # minimum peak width to look for
DARK_QPE_WIDTH = 1e-12
NUM_BINS_FIT = 50


NBINS = 2000  # number of bins to make charge histogram from
# KETEK_PDE = 0.2587
# KETEK_PDE_ERROR = 0.2587 * 0  # have a separate error band for the systematic errors...
# # BROADCOM_PDE = 0.2457870367786596
# BROADCOM_PDE_ERROR = 0.2457870367786596 * 0
PDE_ERROR = 0
e_charge = 1.6e-19

NUM_SIGMA_AWAY = 5
GAUSSIAN_FILTERED_QPE_THRESHOLD = 1

SAVE_SUPERPULSE = False


def calculate_pulse_pde(
    input_file: str,
    temperature: str,
    PDE: float,
    bias: float,
    device_name: str,
    vpp: float,
    light_window_start_idx: int,
    light_window_end_idx: int,
    dark_window_start_idx: int,
    dark_window_end_idx: int,
    output_file: str,
    lock,
) -> None:
    """
    For a given file, read in the waveforms and convert them to current waveforms. Then, integrate the waveforms.
    Apply a peak finding routine to the charge spectrum. Convert the median charge and its error to number of photons;
    either use the gain of the SiPM as calculated earlier in the analysis chain, or use the known resposivity of the APD.

    Parameters
    ----------
    input_file
        A raw-tier CoMPASS file that contains `raw/waveforms` and `raw/baselines` as keys
    temperature
        RT or LN. If LN, then only 2 peaks are fit to accommodate the high AP rates. If RT, it will attempt to fit 3 peaks
    PDE
        The PDE of the reference diode used for converting data
    bias
        The bias the SiPM was set at during the run
    device_name
        The name of the device this file corresponds to, one of:
        `sipm`, `sipm_1st`, `sipm_1st_low_gain`, or `apd`
    vpp
        The voltage division CoMPASS was set to record the DAC at
    light_window_start_idx
        The start of the trigger set to capture the triggered-on illuminated SiPM/APD pulses, in samples
    light_window_end_idx
        The end of the trigger set to capture the triggered-on illuminated SiPM/APD pulses, in samples
    dark_window_start_idx
        The index at which to start integrating a current waveform under dark conditions -- must be suitably far away from the triggered light pulse
    dark_window_end_idx
        The index at which to end integrating a current waveform under dark conditions -- the window must be the same width as the light window for error free analysis
    output_file
        The name of the file in which to place the output of the calculated photon rates
    lock
        The lock to prevent writing to the same file at the same time

    Notes
    -----
    The inputs of this function are all determined by :func:`.util.parse_json_config.parse_light_json`
    The `light_window_start_idx` is determined by the `pretrigger` parameter set in CoMPASS, and the `light_window_end_idx` is determined by the width of the LED pulse
    """
    # Grab the output path from the filename
    out_path = "/" + os.path.join(*output_file.split("/")[:-1])
    print(out_path)

    # Read in the file and convert the waveforms to currents
    sipm_file = h5py.File(input_file, "r")

    waveformies = sipm_file["raw/waveforms"][()]

    sipm_file.close()

    # dark data is usually only taken with second stage amplification, but allow for device specification
    waveformies = current_converter(waveformies, vpp=vpp, n_bits=14, device=device_name)

    # Calculate the baseline
    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    print("Performing baseline routine")

    try:
        wf_bl = find_bl(waveformies, bl_idx)
    except:
        raise ValueError(
            f"{bias}, {device_name} something wrong in baseline estimation"
        )

    # Integrate over the windows provided
    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    c_wfs = wf_bl

    window = [light_window_start_idx, light_window_end_idx]
    dark_window = [dark_window_start_idx, dark_window_end_idx]
    fig, axs = plt.subplots(
        3, 3, figsize=(14, 10)
    )  # create the figure that will hold our subplots
    fig.suptitle(str(bias) + "V " + str(device_name))

    if SAVE_SUPERPULSE:
        print("saving figure of light window")
        fig2 = plt.figure(figsize=(12, 8))
        plt.plot(np.mean(c_wfs, axis=0))
        plt.title(f"Average {device_name} Waveform")
        plt.axvline(x=window[0])
        plt.axvline(x=window[1])
        plt.xlim([window[0] - 100, window[1] + 100])
        plt.ylabel("Current [A]")
        plt.xlabel("Samples")
        fig_path = (
            out_path
            + "/light_window_superpulse_"
            + str(bias)
            + "_"
            + str(device_name)
            + ".png"
        )
        plt.savefig(fig_path, dpi=fig.dpi)

        print("saving figure of dark window")
        fig2 = plt.figure(figsize=(12, 8))
        plt.plot(np.mean(c_wfs, axis=0))
        plt.title(f"Average {device_name} Waveform")
        plt.axvline(x=dark_window[0])
        plt.axvline(x=dark_window[1])
        plt.xlim([dark_window[0] - 100, dark_window[1] + 100])
        fig_path = (
            out_path
            + "/dark_window_superpulse_"
            + str(bias)
            + "_"
            + str(device_name)
            + ".png"
        )
        plt.savefig(fig_path, dpi=fig.dpi)

    qs = integrate_current(c_wfs, lower_bound=window[0], upper_bound=window[1])
    qs_dark = integrate_current(
        c_wfs, lower_bound=dark_window[0], upper_bound=dark_window[1]
    )

    # Histogram and peak find the pedestal in light counts so as to seed gain fit
    # ------------------------------------------------------------------------------------------------------------------------------------------------------

    n, bins, patches = axs[0, 0].hist(qs, bins=NBINS, histtype="step")
    n, bins, patches = axs[0, 0].hist(qs, bins=NBINS, histtype="stepfilled", alpha=0.15)
    axs[0, 0].set_xlabel("Integrated Charge (C)")
    axs[0, 0].set_ylabel("Counts")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_title("Light QPE")

    n_dark, bins_dark, patches_dark = axs[1, 0].hist(
        qs_dark, bins=NBINS, histtype="step"
    )
    n_dark, bins_dark, patches_dark = axs[1, 0].hist(
        qs_dark, bins=NBINS, histtype="stepfilled", alpha=0.15
    )
    axs[1, 0].set_xlabel("Integrated Charge (C)")
    axs[1, 0].set_ylabel("Counts")
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_title("Dark QPE")

    # Find the location of the tallest peak and rough distance to next peak by taking derivatives
    # ----------------------------------------------------------------------------------------------------------------------------

    light_inflection_flag = True
    dark_inflection_flag = True

    # First smooth the derivative of the QPE with a Gaussian filter, find the index of when the maximum peak crosses 0 again
    # Use this index as a hold off to look for the next time the QPE derivative crosses a threshold
    # The threshold crossing index is used as the distance between peaks, and half of its value is the fit width for each peak
    diffs = np.diff(n)
    diffs = gaussian_filter1d(diffs, 5)
    light_max_idx = np.argmax(diffs)
    offset_idx = zero_crosser(diffs, np.argmax(diffs))
    pol, trig = bi_level_zero_crossing_time_points(
        diffs,
        GAUSSIAN_FILTERED_QPE_THRESHOLD,
        -1 * GAUSSIAN_FILTERED_QPE_THRESHOLD,
        1000,
        0,
        np.array([0]),
        np.zeros(10),
        np.zeros(10),
    )
    trig = trig[np.where(~np.isnan(trig))[0]]
    trig = np.array(trig, dtype=int)
    if len(trig) == 0:
        trig = np.array([0, 0])
    elif len(trig) == 1:
        trig = np.array([0, trig[0]])

    light_pedestal_fit_width = 2 * (offset_idx - light_max_idx)
    light_peak_distance = int(np.mean(np.diff(trig)))

    # Do the same for the dark peaks
    diffs = np.diff(n_dark)
    diffs = gaussian_filter1d(diffs, 5)
    dark_max_idx = np.argmax(diffs)
    offset_idx = zero_crosser(diffs, np.argmax(diffs))

    pol, trig = bi_level_zero_crossing_time_points(
        diffs,
        GAUSSIAN_FILTERED_QPE_THRESHOLD,
        -1 * GAUSSIAN_FILTERED_QPE_THRESHOLD,
        1000,
        0,
        np.array([0]),
        np.zeros(10),
        np.zeros(10),
    )
    trig = trig[np.where(~np.isnan(trig))[0]]
    trig = np.array(trig, dtype=int)
    if len(trig) == 0:
        trig = np.array([0, 0])
    elif len(trig) == 1:
        trig = np.array([0, trig[0]])

    dark_pedestal_fit_width = 2 * (offset_idx - dark_max_idx)
    dark_peak_distance = int(np.mean(np.diff(trig)))

    # Try finding the tallest peak and fitting it so as to seed eventual fit parameters for the whole charge spectrum
    # ----------------------------------------------------------------------------------------------------------------------------
    try:
        peak, sigma = tallest_peak_loc_sigma(bins, n)
        peaks = np.array([np.argmax(n)])
        peak_locs = np.array([peak])
        amplitudes = np.array([np.amax(n)])
        sigma = int(sigma / (bins[1] - bins[0]))  # convert to units of bins
        print("Found light peaks")

        peak_dark, sigma_dark = tallest_peak_loc_sigma(bins_dark, n_dark)
        peaks_dark = np.array([np.argmax(n_dark)])
        peak_locs_dark = np.array([peak_dark])
        amplitudes_dark = np.array([np.amax(n_dark)])
        sigma_dark = int(
            sigma_dark / (bins_dark[1] - bins_dark[0])
        )  # convert to units of bins

    except:
        raise ValueError(f"Peak Finding failed {bias}, {device_name}")

    # If no threshold crossings were found, or if the distance between peaks is negative, or if the threshold crossing is too close to the maximum value
    # use the default bin width to fit
    if light_peak_distance <= 0:
        if sigma > 10:  # can't be too small to fit!
            light_inflection_flag = False
            light_pedestal_fit_width = sigma

    if dark_peak_distance <= 0:
        if sigma_dark > 10:
            dark_inflection_flag = False
            dark_pedestal_fit_width = sigma_dark

    # Fit with a Gaussian
    try:
        gauss_params, gauss_errors = fit_peak(
            n, bins, peaks, peak_locs, amplitudes, fit_width=light_pedestal_fit_width
        )

        gauss_params_dark, gauss_errors_dark = fit_peak(
            n_dark,
            bins_dark,
            peaks_dark,
            peak_locs_dark,
            amplitudes_dark,
            fit_width=dark_pedestal_fit_width,
        )

    except:
        raise ValueError(
            f"Peak Fitting Routine Failed. {dark_pedestal_fit_width} and {light_pedestal_fit_width} used as fit width, {bias}, {device_name}, {2 * (offset_idx - light_max_idx)}, {light_inflection_flag}"
        )

    # Save the Gaussian fits for visual inspection
    x = np.linspace(bins[0], bins[-1], NBINS)
    x_dark = np.linspace(bins_dark[0], bins_dark[-1], NBINS)

    for i, params in enumerate(gauss_params):
        axs[0, 1].plot(x, gaussian(x, *params))
    n, bins, patches = axs[0, 1].hist(
        qs,
        bins=NBINS,
        label=f"{light_pedestal_fit_width} bins fit, {light_peak_distance} distance",
    )
    n, bins, patches = axs[0, 1].hist(qs, bins=NBINS, histtype="stepfilled", alpha=0.5)
    axs[0, 1].set_xlabel("Integrated Charge (C)")
    axs[0, 1].set_ylabel("Counts")
    axs[0, 1].set_ylim(1, np.amax(n))
    axs[0, 1].set_yscale("log")
    axs[0, 1].set_title(f"Light QPE Pedestal Fit")
    axs[0, 1].legend()

    for i, params in enumerate(gauss_params_dark):
        axs[1, 1].plot(x_dark, gaussian(x_dark, *params))
    n_dark, bins_dark, patches_dark = axs[1, 1].hist(
        qs_dark,
        bins=NBINS,
        label=f"{dark_pedestal_fit_width} bins fit, {dark_peak_distance} distance",
    )
    n_dark, bins_dark, patches_dark = axs[1, 1].hist(
        qs_dark, bins=NBINS, histtype="stepfilled", alpha=0.5
    )
    axs[1, 1].set_xlabel("Integrated Charge (C)")
    axs[1, 1].set_ylabel("Counts")
    axs[1, 1].set_ylim(1, np.amax(n_dark))
    axs[1, 1].set_yscale("log")
    axs[1, 1].set_title("Dark QPE Pedestal Fit")
    axs[1, 1].legend()

    # Expand the gaussian parameters into their own arrays
    centroid = []
    centroid_std = []
    for i, x in enumerate(gauss_params):
        centroid.append(x[1])  # the median from [amplitude, median, sigma]
        centroid_std.append(np.abs(x[2]))  # this is the width of the peak

    centroid_dark = []
    centroid_std_dark = []
    for i, x in enumerate(gauss_params_dark):
        centroid_dark.append(x[1])  # the median from [amplitude, median, sigma]
        centroid_std_dark.append(np.abs(x[2]))  # this is the width of the peak

    centroid = np.array(centroid)
    centroid_std = np.array(centroid_std)

    centroid_dark = np.array(centroid_dark)
    centroid_std_dark = np.array(centroid_std_dark)
    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    # Calculate the gain and number of photons
    bin_width = bins[1] - bins[0]
    fit_width = int(2.3 * centroid_std[0] / bin_width)  # fit the FWQM of the pedestal

    if not light_inflection_flag:
        light_peak_distance = 2.355 * NUM_SIGMA_AWAY * centroid_std[0] / bin_width

    gain = calculate_gain(
        temperature,
        bias,
        device_name,
        qs,
        light_peak_distance * bin_width,
        fit_width,
        axs,
    )
    print(gain)

    # Normalize the charge distributions
    qs_normal = (qs - centroid) / gain[0] / e_charge
    qs_dark_normal = (qs_dark - centroid_dark) / gain[0] / e_charge

    # Calculate the number of photons using the pedestal method
    NPE_CUT = 0.5
    N_gamma = calculate_photons(
        bias,
        device_name,
        qs_normal,
        qs_dark_normal,
        NPE_CUT,
        0,
        NPE_CUT,
        0,
        3,
        axs,
    )
    print(N_gamma)

    if device_name == "reference":
        n_photons = N_gamma[0] / PDE
        n_photons_u = n_photons * np.sqrt(
            (N_gamma[1] / N_gamma[0]) ** 2 + (PDE_ERROR / PDE) ** 2
        )
        n_q = np.array([n_photons, n_photons_u])

        print(n_q)

    else:
        n_q = N_gamma  # dark subtraction is already accounted for
        print(n_q)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    # Save it all to a file
    fig_path = (
        out_path + "/monitoring_plots_" + str(bias) + "_" + str(device_name) + ".png"
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=fig.dpi)

    lock.acquire()
    f = h5py.File(output_file, "a")

    if (
        device_name == "sipm"
        or device_name == "sipm_1st"
        or device_name == "sipm_1st_low_gain"
        or device_name == "sipm_1st_low_gain_goofy"
    ):

        device = "sipm"

        if f"{device}/{bias}/n_photons" in f:
            f[f"{device}/{bias}/n_photons"][:] = n_q
            f[f"{device}/{bias}/charges"][:] = qs
            f[f"{device}/{bias}/window"][:] = window
            f[f"{device}/{bias}/gain"][:] = gain
            f[f"{device}/dark/{bias}/charges"][:] = qs_dark
            f[f"{device}/dark/{bias}/window"][:] = dark_window

        else:
            dset = f.create_dataset(f"{device}/{bias}/n_photons", data=n_q)
            dset = f.create_dataset(f"{device}/{bias}/charges", data=qs)
            dset = f.create_dataset(f"{device}/{bias}/window", data=window)
            dset = f.create_dataset(f"{device}/{bias}/gain", data=gain)
            dset = f.create_dataset(f"{device}/dark/{bias}/charges", data=qs_dark)
            dset = f.create_dataset(f"{device}/dark/{bias}/window", data=dark_window)

    if device_name == "reference" or device_name == "apd_goofy" or device_name == "apd":
        device = "reference"

        if f"{device}/{bias}/n_photons" in f:
            del f[f"{device}/{bias}/n_photons"]
            dset = f.create_dataset(f"{device}/{bias}/n_photons", data=n_q)
            #         f[f'{device}/{bias}/n_photons'][:] = n_q
            del f[f"{device}/{bias}/charges"]
            dset = f.create_dataset(f"{device}/{bias}/charges", data=qs)
            f[f"{device}/{bias}/window"][:] = window
            f[f"{device}/{bias}/gain"][:] = gain
            del f[f"{device}/dark/{bias}/charges"]
            dset = f.create_dataset(f"{device}/dark/{bias}/charges", data=qs_dark)
            f[f"{device}/dark/{bias}/charges"][:] = qs_dark
            f[f"{device}/dark/{bias}/window"][:] = dark_window

        else:
            dset = f.create_dataset(f"{device}/{bias}/n_photons", data=n_q)
            dset = f.create_dataset(f"{device}/{bias}/charges", data=qs)
            dset = f.create_dataset(f"{device}/{bias}/window", data=window)
            dset = f.create_dataset(f"{device}/{bias}/gain", data=gain)
            dset = f.create_dataset(f"{device}/dark/{bias}/charges", data=qs_dark)
            dset = f.create_dataset(f"{device}/dark/{bias}/window", data=dark_window)

    f.close()
    lock.release()


def calculate_gain(
    temperature: str,
    bias: float,
    device_name: str,
    qs_light: list,
    peak_distance: float,
    fit_width=15,
    axs=None,
) -> list:
    """
    Parameters
    ----------
    temperature
        RT or LN. If LN, then only 2 peaks are fit to accommodate the high AP rates. If RT, it will attempt to fit 3 peaks
    bias
        The bias the SiPM was set at during the run
    device_name
        The name of the device this file corresponds to, one of:
        `sipm`, `sipm_1st`, `sipm_1st_low_gain`, or `apd`
    qs_light
        Array of integrated waveforms that give a QPE spectrum
    peak_distance
        The distance between peaks in the spectrum to calculate gain
    fit_width
        The number of bins to fit around a peak
    axs
        A `matplotlib.pyplot` axes object to hold the output plots


    Returns
    -------
    Gain
        The gain calculated from the pulsed light spectrum

    Notes
    -----
    This function works by first attempting to find peaks with an inter-peak distance of `peak_distance` passed to the function.
    Then, if more than 1 peak is found, the peaks are fitted. There are stop-gaps to ensure fitting works: if the fit_width is too small,
    then the distance between the found peaks is used to supplement the fit_width guess. Finally the gain is calculated. If 3 peaks are found,
    then the slope of the NPE vs Q graph is used; otherwise, the pedestal is subtracted off and the first peak sets the gain.
    """
    # Now calculate the gain from the light histogram
    # It's ok to use the light bc the p.e. peaks don't shift, just relative ratios
    fig = plt.figure(figsize=(12, 8))
    n, bins, patches = plt.hist(qs_light, bins=NBINS, histtype="step")
    n, bins, patches = plt.hist(qs_light, bins=NBINS, histtype="stepfilled", alpha=0.15)
    plt.title("Charge Spectrum Under Illumination")
    plt.xlabel("Integrated Charge (C)")
    plt.ylabel("Counts")
    plt.yscale("log")
    plt.clf()

    peaks, peak_locs, amplitudes = guess_peaks_no_width(n, bins, 4, peak_distance)

    # TODO:
    # Need some logic here to handle cases when there is high AP and only 2 distinguishable peaks
    if len(peaks) == 2:
        max_peaks = 2
    else:
        max_peaks = 3

    if temperature == "LN":
        max_peaks = 2  # Used if there aren't enough peaks as in the case of dark LN data sometimes due to signal degradation from afterpulsing

    peaks = peaks[:max_peaks]
    peak_locs = peak_locs[:max_peaks]
    amplitudes = amplitudes[:max_peaks]

    n, bins, patches = axs[2, 0].hist(qs_light, bins=NBINS, histtype="step")
    n, bins, patches = axs[2, 0].hist(
        qs_light, bins=NBINS, histtype="stepfilled", alpha=0.15
    )

    axs[2, 0].scatter(peak_locs, amplitudes)
    axs[2, 0].set_title("Light QPE Peaks Found")
    axs[2, 0].set_xlabel("Integrated Charge (C)")
    axs[2, 0].set_ylabel("Counts")
    axs[2, 0].set_yscale("log")

    if len(peaks) > 1:
        # If the found fit width is too small, there might be fit convergence errors
        # So try finding FWHM of the second peak
        FWQM_2nd_peak = 0

        for i in range(int(peaks[1]), len(n) - 1):
            if n[i] >= 3 * amplitudes[1] / 4 > n[i + 1]:
                FWQM_2nd_peak = 2 * (i - peaks[1])
            else:
                pass

        if fit_width <= 0.000001 * FWQM_2nd_peak:
            fit_width = FWQM_2nd_peak

        if fit_width <= 10:
            fit_width = (peaks[1] - peaks[0]) // 3

        try:
            gauss_params, gauss_errors = fit_peaks_no_sigma_guess(
                n, bins, peaks, peak_locs, amplitudes, fit_width=fit_width
            )
        except:
            raise ValueError(
                f"{bias}, {device_name}, {peak_locs}, {fit_width}, failing, could try {FWQM_2nd_peak}"
            )

    # Logic if there is only one peak -- look 5 sigma beyond one peak and then take the median value as peak 2?
    else:
        gauss_params, gauss_errors = fit_peaks_no_sigma_guess(
            n, bins, peaks, peak_locs, amplitudes, fit_width=fit_width
        )
        peak_locs = np.append(peak_locs, np.median(bins[bins > peak_distance]))
        peaks = np.append(peaks, np.argmin(np.abs(bins - peak_locs[1])))
        amplitudes = np.append(amplitudes, n[peaks[1]])

        gauss_params.append(
            [amplitudes[1], peaks[1], np.std(bins[bins > peak_distance])]
        )
        gauss_errors.append(
            [
                np.sqrt(amplitudes[1]),
                np.sqrt(peaks[1]),
                np.std(bins[bins > peak_distance]),
            ]
        )

    x = np.linspace(bins[0], bins[-1], NBINS)

    for i, params in enumerate(gauss_params):
        axs[2, 1].plot(x, gaussian(x, *params))
    n, bins, patches = axs[2, 1].hist(qs_light, bins=NBINS)
    n, bins, patches = axs[2, 1].hist(
        qs_light, bins=NBINS, histtype="stepfilled", alpha=0.5
    )
    axs[2, 1].set_xlabel("Integrated Charge (C)")
    axs[2, 1].set_ylabel("Counts")
    axs[2, 1].set_ylim(1, np.amax(n))
    axs[2, 1].grid(True)
    axs[2, 1].set_title("Light QPE Peaks Fit")
    axs[2, 1].set_yscale("log")

    # Do the actual gain calculation
    qs_light = np.array(qs_light)
    centroid = []
    centroid_err = []
    for i, x in enumerate(gauss_params):
        centroid.append(x[1])
        centroid_err.append(gauss_errors[i][1])

    if max_peaks > 2:
        nick_gain, n_qs, nick_gain_u = normalize_charge(
            qs_light, centroid, centroid_err
        )

        axs[2, 2].hist(n_qs, bins=NBINS, alpha=0.5)

        axs[2, 2].axvline(x=0)
        axs[2, 2].axvline(x=1)
        axs[2, 2].axvline(x=2)
        axs[2, 2].axvline(x=3)
        axs[2, 2].axvline(x=4)
        axs[2, 2].axvline(x=5)
        axs[2, 2].axvline(x=6)

        axs[2, 2].set_yscale("log")
        axs[2, 2].set_xlim([-1, 5])
        axs[2, 2].set_xlabel("N.P.E.")
        axs[2, 2].set_ylabel("Counts")
        axs[2, 2].set_title("Normalized Light Charge Spectrum")

        nick_gain = [nick_gain, nick_gain_u]
        print(np.array(nick_gain) / 1e6)
        print(nick_gain_u / 1e6)

    my_gainer = my_gain(centroid, centroid_err)
    print(my_gainer / 1e6)

    slope_gain = slope_fit_gain(centroid)
    print(slope_gain / 1e6)

    if max_peaks == 2:  # NOTE: this assumes that the first peak is the pedestal peak!
        plt.clf()
        axs[2, 2].hist(
            (qs_light - centroid[0]) / (my_gainer[0]) / e_charge,
            range=(-1, 5),
            bins=NBINS,
        )
        axs[2, 2].set_yscale("log")
        axs[2, 2].axvline(x=0)
        axs[2, 2].axvline(x=1)
        axs[2, 2].axvline(x=2)
        axs[2, 2].set_title("Normalized Light Charge Spectrum")
        axs[2, 2].set_xlim([-1, 5])

    return np.array(my_gainer)


def calculate_photons(
    bias: float,
    device_name: str,
    qs_light: list,
    qs_dark: list,
    pedestal_light: float,
    pedestal_light_std: float,
    pedestal_dark: float,
    pedestal_dark_std: float,
    std_cut: int,
    axs=None,
) -> list:
    """
    Parameters
    ----------
    bias
        The bias the SiPM was set at during the run
    device_name
        The name of the device this file corresponds to, one of:
        `sipm`, `sipm_1st`, `sipm_1st_low_gain`, or `apd`
    qs_light
        Array of integrated waveforms during the light trigger
    qs_dark
        Array of integrated waveforms (integrated over the same length window!) under dark conditions
    pedestal_light
        The location of the centroid of the pedestal from illumination, in Coulombs
    pedestal_light_std
        The width of the pedestal in light conditions, in Coulombs
    pedestal_dark
        The location of the centroid of the pedestal from dark conditinos, in Coulombs
    pedestal_dark_std
        The width of the pedestal in dark conditions, in Coulombs
    std_cut
        The number of standard deviations to sum up the Gaussian fit to the peak
    axs
        A `matplotlib.pyplot` axes object to hold the output plots

    Returns
    -------
    N_gamma
        The number of photons, and its error from error propagation

    Notes
    -----
    In order to prevent biasing from correlated noise, the number of events in the pedestal can be used to determine the
    mean number of photons. If the detected light is assumed to be Poissonian, then the probability of detecting 0 photons is
    P(0) = e^(-N), so we can find N = -log(N_pedestal/N_counted)
    """
    sigma_cutoff = std_cut  # this gets like 99% of events in the pedestal, as long as it doesn't overlap with 1p.e.

    n, bins, _ = axs[0, 2].hist(qs_light, bins=500)
    axs[0, 2].axvline(x=pedestal_light + sigma_cutoff * pedestal_light_std)
    axs[0, 2].set_yscale("log")
    axs[0, 2].set_xlabel("Number of Photoelectrons")
    axs[0, 2].set_ylabel("Counts")
    axs[0, 2].set_title("Light Photon Distribution")

    N_light = qs_light[qs_light < pedestal_light + sigma_cutoff * pedestal_light_std]

    n, bins, _ = axs[1, 2].hist(qs_dark, bins=500)
    axs[1, 2].axvline(x=pedestal_dark + sigma_cutoff * pedestal_dark_std)
    axs[1, 2].set_yscale("log")
    axs[1, 2].set_xlabel("Number of Photoelectrons")
    axs[1, 2].set_ylabel("Counts")
    axs[1, 2].set_title("Dark Photon Distribution")

    N_dark = qs_dark[qs_dark < pedestal_dark + sigma_cutoff * pedestal_dark_std]

    f_light = len(N_light) / len(qs_light)
    f_dark = len(N_dark) / len(qs_dark)

    N_gamma = np.log(f_dark / f_light)
    N_gamma_u = np.sqrt((1 / len(N_light)) + (1 / len(N_dark)))

    print("Dark Photons", -np.log(f_dark))
    print("Light Photons", np.log(1 / f_light))

    return np.array([N_gamma, N_gamma_u])


def thresh_crosser(w_in, a_threshold_in, t_start_in):
    """
    Find the index where the waveform value crosses 0 after crossing the threshold.
    Useful for finding pileup events with the CR-RC^2 filter. Or for finding the next peak
    in the derivative of a QPE spectrum.

    Parameters
    ----------
    w_in
        the input waveform.
    a_threshold_in
        the threshold value.
    t_start_in
        the starting index.

    Returns
    -------
    t_trig_times_out
        the first index where the waveform value has crossed the threshold and returned to 0.
    """
    # prepare output
    t_trig_times_out = 0

    # Check everything is ok
    if np.isnan(w_in).any() or np.isnan(a_threshold_in) or np.isnan(t_start_in):
        return

    # Perform the processing
    is_above_thresh = False
    for i in range(int(t_start_in), len(w_in) - 1, 1):
        if w_in[i] <= a_threshold_in < w_in[i + 1]:
            is_above_thresh = True
        if is_above_thresh and (w_in[i] >= 0 > w_in[i + 1]):
            t_trig_times_out = i
            is_above_thresh = False
            break
    return t_trig_times_out


def zero_crosser(w_in, t_start_in):
    """
    Find the index where the waveform value crosses 0.
    Useful for finding pileup events with the CR-RC^2 filter.

    Parameters
    ----------
    w_in
        the input waveform.
    t_start_in
        the starting index.

    Returns
    -------
    t_trig_times_out
        the first index where the waveform value has crossed the threshold and returned to 0.
    """
    # prepare output
    t_trig_times_out = 0

    # Check everything is ok
    if np.isnan(w_in).any() or np.isnan(t_start_in):
        return

    # Perform the processing
    is_above_thresh = False
    for i in range(int(t_start_in), len(w_in) - 1, 1):
        if w_in[i] >= 0 > w_in[i + 1]:
            is_above_thresh = True
            t_trig_times_out = i
            break
    return t_trig_times_out


def bi_level_zero_crossing_time_points(
    w_in: np.ndarray,
    a_pos_threshold_in: float,
    a_neg_threshold_in: float,
    gate_time_in: int,
    t_start_in: int,
    n_crossings_out: np.array,
    polarity_out: np.array,
    t_trig_times_out: np.array,
) -> None:
    """
    Find the indices where a waveform value crosses 0 after crossing the threshold and reaching the next threshold within some gate time.
    Works on positive and negative polarity waveforms.
    Useful for finding pileup events with the RC-CR^2 filter.

    Parameters
    ----------
    w_in
        the input waveform.
    a_pos_threshold_in
        the positive threshold value.
    a_neg_threshold_in
        the negative threshold value.
    gate_time_in
        The number of samples that the next threshold crossing has to be within in order to count a 0 crossing
    t_start_in
        the starting index.
    n_crossings_out
        the number of zero-crossings found. Note: if there are more zeros than elements in output arrays, this will continue to increment but the polarity and trigger time will not be added to the output buffers
    polarity_out
        An array holding the polarity of identified pulses. 0 for negative and 1 for positive
    t_trig_times_out
        the indices where the waveform value has crossed the threshold and returned to 0.
        Arrays of fixed length (padded with `numpy.nan`) that hold the
        indices of the identified trigger times.

    JSON Configuration Example
    --------------------------

    .. code-block :: json

        "trig_times_out": {
            "function": "multi_trigger_time",
            "module": "dspeed.processors",
            "args": ["wf_rc_cr2", "5", "-10", 0, "n_crossings", "polarity_out(20)", "trig_times_out(20)"],
            "unit": "ns"
        }
    """
    # prepare output
    t_trig_times_out[:] = np.nan
    polarity_out[:] = np.nan

    # Check everything is ok
    if (
        np.isnan(w_in).any()
        or np.isnan(a_pos_threshold_in)
        or np.isnan(a_neg_threshold_in)
        or np.isnan(t_start_in)
    ):
        return

    if np.floor(t_start_in) != t_start_in:
        raise ValueError("The starting index must be an integer")

    if int(t_start_in) < 0 or int(t_start_in) >= len(w_in):
        raise ValueError("The starting index is out of range")

    if len(polarity_out) != len(t_trig_times_out):
        raise ValueError("The output arrays are of different lengths.")

    gate_time_in = int(gate_time_in)  # make sure this is an integer!
    # Perform the processing
    is_above_thresh = False
    is_below_thresh = False
    crossed_zero = False
    n_crossings_out[0] = 0
    for i in range(int(t_start_in), len(w_in) - 1, 1):
        if is_below_thresh and (w_in[i] <= 0 < w_in[i + 1]):
            crossed_zero = True
            neg_trig_time_candidate = i

        # Either we go above threshold
        if w_in[i] <= a_pos_threshold_in < w_in[i + 1]:
            if crossed_zero and is_below_thresh:
                if i - is_below_thresh < gate_time_in:
                    if n_crossings_out[0] < len(polarity_out):
                        t_trig_times_out[n_crossings_out[0]] = neg_trig_time_candidate
                        polarity_out[n_crossings_out[0]] = 0
                    n_crossings_out[0] += 1
                else:
                    is_above_thresh = i

                is_below_thresh = False
                crossed_zero = False
            else:
                is_above_thresh = i

        if is_above_thresh and (w_in[i] >= 0 > w_in[i + 1]):
            crossed_zero = True
            pos_trig_time_candidate = i

        # Or we go below threshold
        if w_in[i] >= a_neg_threshold_in > w_in[i + 1]:
            if crossed_zero and is_above_thresh:
                if i - is_above_thresh < gate_time_in:
                    if n_crossings_out[0] < len(polarity_out):
                        t_trig_times_out[n_crossings_out[0]] = pos_trig_time_candidate
                        polarity_out[n_crossings_out[0]] = 1
                    n_crossings_out[0] += 1
                else:
                    is_below_thresh = i
                is_above_thresh = False
                crossed_zero = False
            else:
                is_below_thresh = i

    return polarity_out, t_trig_times_out
