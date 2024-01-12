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

SAVE_SUPERPULSE = False
SAVE_GAIN_PLOTS = True


def calculate_pulse_pde(
    input_file: str,
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
    wf_bl = find_bl(waveformies, bl_idx)

    # Integrate over the windows provided
    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    c_wfs = wf_bl

    window = [light_window_start_idx, light_window_end_idx]
    dark_window = [dark_window_start_idx, dark_window_end_idx]

    if SAVE_SUPERPULSE:
        print("saving figure of light window")
        fig = plt.figure(figsize=(12, 8))
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
        fig = plt.figure(figsize=(12, 8))
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

    fig = plt.figure(figsize=(12, 8))
    n, bins, patches = plt.hist(qs, bins=NBINS, histtype="step")
    n, bins, patches = plt.hist(qs, bins=NBINS, histtype="stepfilled", alpha=0.15)
    plt.xlabel("Integrated Charge (C)")
    plt.ylabel("Counts")
    plt.yscale("log")
    fig_path = out_path + "/light_qpe_" + str(bias) + "_" + str(device_name) + ".png"
    if False:
        plt.savefig(fig_path, dpi=fig.dpi)

    fig = plt.figure(figsize=(12, 8))
    n_dark, bins_dark, patches_dark = plt.hist(qs_dark, bins=NBINS, histtype="step")
    n_dark, bins_dark, patches_dark = plt.hist(
        qs_dark, bins=NBINS, histtype="stepfilled", alpha=0.15
    )
    plt.xlabel("Integrated Charge (C)")
    plt.ylabel("Counts")
    plt.yscale("log")
    fig_path = out_path + "/dark_qpe_" + str(bias) + "_" + str(device_name) + ".png"
    if False:
        plt.savefig(fig_path, dpi=fig.dpi)

    try:
        peaks, peak_locs, amplitudes = guess_peaks_no_width(
            n, bins, LIGHT_QPE_HEIGHT, LIGHT_QPE_WIDTH
        )
        print("Found light peaks")

        peaks_dark, peak_locs_dark, amplitudes_dark = guess_peaks_no_width(
            n_dark, bins_dark, DARK_QPE_HEIGHT, DARK_QPE_WIDTH
        )
        # only care about the location of the pedestal
        peaks = peaks[:1]
        peak_locs = peak_locs[:1]
        amplitudes = amplitudes[:1]

        peaks_dark = peaks_dark[:1]
        peak_locs_dark = peak_locs_dark[:1]
        amplitudes_dark = amplitudes_dark[:1]
    except:
        raise ValueError("Peak Finding failed")

    # Fit with a Gaussian
    try:
        gauss_params, gauss_errors = fit_peak(
            n, bins, peaks, peak_locs, amplitudes, fit_width=NUM_BINS_FIT
        )

        gauss_params_dark, gauss_errors_dark = fit_peak(
            n_dark,
            bins_dark,
            peaks_dark,
            peak_locs_dark,
            amplitudes_dark,
            fit_width=NUM_BINS_FIT,
        )

    except:
        raise ValueError("Peak Fitting Routine Failed.")

    # Save the Gaussian fits for visual inspection
    x = np.linspace(bins[0], bins[-1], NBINS)

    fig = plt.figure(figsize=(12, 8))
    for i, params in enumerate(gauss_params):
        plt.plot(x, gaussian(x, *params))
    n, bins, patches = plt.hist(qs, bins=NBINS)
    n, bins, patches = plt.hist(qs, bins=NBINS, histtype="stepfilled", alpha=0.5)
    plt.xlabel("Integrated Charge (C)")
    plt.ylabel("Counts")
    plt.ylim(1, np.amax(n))
    plt.yscale("log")
    plt.title(f"Histogram of Light Charges for {bias}V in {device_name}")
    fig_path = (
        out_path + "/light_fit_histogram_" + str(bias) + "_" + str(device_name) + ".png"
    )
    plt.savefig(fig_path, dpi=fig.dpi)

    fig = plt.figure(figsize=(12, 8))
    for i, params in enumerate(gauss_params_dark):
        plt.plot(x, gaussian(x, *params))
    n_dark, bins_dark, patches_dark = plt.hist(qs_dark, bins=NBINS)
    n_dark, bins_dark, patches_dark = plt.hist(
        qs_dark, bins=NBINS, histtype="stepfilled", alpha=0.5
    )
    plt.xlabel("Integrated Charge (C)")
    plt.ylabel("Counts")
    plt.ylim(1, np.amax(n_dark))
    plt.yscale("log")
    plt.title(f"Histogram of Dark Charges for {bias}V in {device_name}")
    fig_path = (
        out_path + "/dark_fit_histogram_" + str(bias) + "_" + str(device_name) + ".png"
    )
    plt.savefig(fig_path, dpi=fig.dpi)

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
    fit_width = int(2.355 * centroid_std[0] / bin_width)  # fit the FWHM of the pedestal

    if device_name not in ["reference"]:
        gain = calculate_gain(
            out_path,
            bias,
            device_name,
            qs,
            qs_dark,
            centroid[0],
            2.355 * 2 * centroid_std[0],
            fit_width,
        )
        print(gain)

    # Use this block if the reference diode is changing bias voltage, like for calculating the gain of the reference diode
    #     else:
    #         gain = calculate_gain(out_path, bias, device_name, qs, qs_dark, centroid[0], 2.355*6*centroid_std[0], fit_width)
    #         print(gain)

    # good if the reference diode is fixed
    else:
        gain = calculate_gain(
            out_path, bias, device_name, qs, qs_dark, centroid[0], 0.5e-12, fit_width
        )
        print(gain)

    # Normalize the charge distributions
    qs_normal = (qs - centroid) / gain[0] / e_charge
    qs_dark_normal = (qs_dark - centroid_dark) / gain[0] / e_charge

    # Calculate the number of photons using the pedestal method
    NPE_CUT = 0.5
    N_gamma = calculate_photons(
        out_path,
        bias,
        device_name,
        qs_normal,
        qs_dark_normal,
        NPE_CUT,
        0,
        NPE_CUT,
        0,
        std_cut=3,
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
    out_path: str,
    bias: float,
    device_name: str,
    qs_light: list,
    qs_dark: list,
    centroid,
    peak_distance: float,
    fit_width=15,
) -> list:
    """
    Parameters
    ----------
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
        The numbers of standard deviations to sum the area under the pedestal Gaussian
    peak_distance
        The distance between peaks in the dark spectrum to calculate gain


    Returns
    -------
    Gain
        The gain calculated from the pulsed light spectrum
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

    # Find the distance between the first two peaks by chopping off the histogram from the pedestal onward
    if device_name not in ["reference"]:
        qs_light_above_pedestal = n[bins[1:] >= centroid + peak_distance]
        bins_above_pedestal = bins[bins >= centroid + peak_distance]
        next_peak = bins_above_pedestal[np.argmax(qs_light_above_pedestal)]
        peak_distance = next_peak - centroid

    peaks, peak_locs, amplitudes = guess_peaks_no_width(n, bins, 4, peak_distance)

    max_peaks = 3
    #     max_peaks = 2 # Used if there aren't enough peaks as in the case of dark LN data sometimes due to signal degradation from afterpulsing

    peaks = peaks[:max_peaks]
    peak_locs = peak_locs[:max_peaks]
    amplitudes = amplitudes[:max_peaks]

    fig = plt.figure(figsize=(12, 8))
    n, bins, patches = plt.hist(qs_light, bins=NBINS, histtype="step")
    n, bins, patches = plt.hist(qs_light, bins=NBINS, histtype="stepfilled", alpha=0.15)

    if SAVE_GAIN_PLOTS:
        plt.scatter(peak_locs, amplitudes)
        plt.title("Charge Spectrum Under Illumination")
        plt.xlabel("Integrated Charge (C)")
        plt.ylabel("Counts")
        plt.yscale("log")
        plt.show()
        fig_path = (
            out_path + "/light_qpe_peaks_" + str(bias) + "_" + str(device_name) + ".png"
        )
        plt.savefig(fig_path, dpi=fig.dpi)

    gauss_params, gauss_errors = fit_peaks_no_sigma_guess(
        n, bins, peaks, peak_locs, amplitudes, fit_width=fit_width
    )
    x = np.linspace(bins[0], bins[-1], NBINS)

    if SAVE_GAIN_PLOTS:
        fig = plt.figure(figsize=(12, 8))
        for i, params in enumerate(gauss_params):
            plt.plot(x, gaussian(x, *params))
        n, bins, patches = plt.hist(qs_light, bins=NBINS)
        n, bins, patches = plt.hist(
            qs_light, bins=NBINS, histtype="stepfilled", alpha=0.5
        )
        plt.xlabel("Integrated Charge (C)")
        plt.ylabel("Counts")
        plt.ylim(1, np.amax(n))
        plt.grid(True)
        plt.title("Charge Histogram Under Illumination")
        plt.yscale("log")
        fig_path = (
            out_path
            + "/light_qpe_peaks_fit_"
            + str(bias)
            + "_"
            + str(device_name)
            + ".png"
        )
        plt.savefig(fig_path, dpi=fig.dpi)

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

        if SAVE_GAIN_PLOTS:
            fig = plt.figure(figsize=(12, 8))
            plt.hist(n_qs, bins=NBINS, alpha=0.5)

            plt.axvline(x=0)
            plt.axvline(x=1)
            plt.axvline(x=2)
            plt.axvline(x=3)
            plt.axvline(x=4)
            plt.axvline(x=5)
            plt.axvline(x=6)

            plt.yscale("log")
            plt.xlim([-1, 5])
            plt.xlabel("N.P.E.")
            plt.ylabel("Counts")
            plt.title("Normalized Charge Spectrum")

            fig_path = (
                out_path
                + "/light_qpe_normalized_"
                + str(bias)
                + "_"
                + str(device_name)
                + ".png"
            )
            plt.savefig(fig_path, dpi=fig.dpi)

        nick_gain = [nick_gain, nick_gain_u]
        print(np.array(nick_gain) / 1e6)
        print(nick_gain_u / 1e6)

    my_gainer = my_gain(centroid, centroid_err)
    print(my_gainer / 1e6)

    slope_gain = slope_fit_gain(centroid)
    print(slope_gain / 1e6)

    if max_peaks == 2:
        plt.clf()
        fig = plt.figure(figsize=(12, 8))
        plt.hist(
            (qs_light - centroid[0]) / (my_gainer[0]) / e_charge, range=(0, 4), bins=500
        )
        plt.yscale("log")
        plt.axvline(x=0)
        plt.axvline(x=1)
        plt.axvline(x=2)
        plt.title("Normalized Spectrum")
        plt.xlim([-1, 3])
        plt.show()

    return np.array(my_gainer)


def calculate_photons(
    out_path: str,
    bias: float,
    device_name: str,
    qs_light: list,
    qs_dark: list,
    pedestal_light: float,
    pedestal_light_std: float,
    pedestal_dark: float,
    pedestal_dark_std: float,
    std_cut: int,
) -> list:
    """
    Parameters
    ----------
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

    Returns
    -------
    N_gamma
        The number of photons, and its error from error propagation

    Notes
    -----
    In order to prevent biasing from correlated noise, the number of events in the pedestal can be used to determine the
    mean number of photons. If the detected light is assumed to be Poissonian, then the probability of detecting 0 photons is
    P(0) = e^(-N), so we can find N = -log(N_pedestal/N_counted)

    Ideas
    -----
    Maybe want to just take the midpoint between 0p.e. and 1p.e. to accommodate for different sigma sizes due to different
    amplifier response functions
    """
    sigma_cutoff = std_cut  # this gets like 99% of events in the pedestal, as long as it doesn't overlap with 1p.e.

    if SAVE_GAIN_PLOTS:
        fig = plt.figure(figsize=(12, 8))
        n, bins, _ = plt.hist(qs_light, bins=500)
        plt.axvline(x=pedestal_light + sigma_cutoff * pedestal_light_std)
        plt.yscale("log")
        plt.title("Light Distribution")
        fig_path = (
            out_path
            + "/light_qpe_photons_"
            + str(bias)
            + "_"
            + str(device_name)
            + ".png"
        )
        plt.savefig(fig_path, dpi=fig.dpi)

    N_light = qs_light[qs_light < pedestal_light + sigma_cutoff * pedestal_light_std]

    if SAVE_GAIN_PLOTS:
        fig = plt.figure(figsize=(12, 8))
        n, bins, _ = plt.hist(qs_dark, bins=500)
        plt.axvline(x=pedestal_dark + sigma_cutoff * pedestal_dark_std)
        plt.yscale("log")
        plt.title("Dark Distribution")
        fig_path = (
            out_path
            + "/dark_qpe_photons_"
            + str(bias)
            + "_"
            + str(device_name)
            + ".png"
        )
        plt.savefig(fig_path, dpi=fig.dpi)

    N_dark = qs_dark[qs_dark < pedestal_dark + sigma_cutoff * pedestal_dark_std]

    f_light = len(N_light) / len(qs_light)
    f_dark = len(N_dark) / len(qs_dark)

    N_gamma = np.log(f_dark / f_light)
    N_gamma_u = np.sqrt((1 / len(N_light)) + (1 / len(N_dark)))

    print("Dark Photons", -np.log(f_dark))
    print("Light Photons", np.log(1 / f_light))

    return np.array([N_gamma, N_gamma_u])
