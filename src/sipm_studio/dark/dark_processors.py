import os, sys, h5py, json, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
from uncertainties import ufloat, unumpy
import warnings
from tqdm import tqdm
import random
from scipy.signal import find_peaks
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import time
from scipy.stats import chisquare

import matplotlib as mpl

font = {"size": 10}

mpl.rc("font", **font)

plt.rcParams["figure.autolayout"] = True


from sipm_studio.dsp.adc_to_current import current_converter
from sipm_studio.dsp.current_to_charge import (
    integrate_current,
    rando_integrate_current,
)
from sipm_studio.dsp.qpe_peak_finding import (
    tallest_peak_loc_sigma,
    fit_peaks_no_sigma_guess,
    gaussian,
    guess_peaks_no_width,
)
from sipm_studio.light.pde_pulse_method import zero_crosser, thresh_crosser

NUM_BINS_FIT = 50
NUM_SIGMA_AWAY = 5
GAUSSIAN_FILTERED_QPE_THRESHOLD = 0


def number_peaks_return_to_bl(wf, peaks, trigger_amp):
    """
    Starting from the first peak in a waveform (i.e. the trigger waveform for ), make sure it is 1p.e. peak,
    find the sample where the wf returns to baseline, and then count the number of peaks in that window.

    Should give you a measure of the "train" or how many pulses per pulse?
    """
    if len(peaks) == 0:
        return 0
    if (trigger_amp > 0.5) & (trigger_amp < 1.5):
        bl = np.min(
            [np.mean(wf[:20]), np.mean(wf[-100:])]
        )  # there really shouldn't be any peaks at the start or end of a waveform at ln temps...
        bl_std = np.min([np.std(wf[:20]), np.std(wf[-100:])])
        return_idx = peaks[0]
        while return_idx < len(wf):
            if (wf[return_idx] <= bl + bl_std) & (wf[return_idx] >= bl - bl_std):
                break
            else:
                return_idx += 1
        return len(peaks[peaks <= return_idx])
    else:
        return 0


def best_guess_peak_distances(n):
    """
    Parameters
    ----------
    n
        The count values from a histogram of a charge or pulse height spectrum that is roughly sums of Gaussians


    Returns
    -------
    light_inflection_flag
        A boolean, if true means that a good inflection point in the spectrum was found and guesses are accurate
    light_pedestal_fit_width
        A guess of the width of each peak to fit, in bin distances
    light_peak_distance
        A guess of the distance to the next peak

    Notes
    -----
    This function works by taking the derivative of a spectrum (reducing the noise, while keeping it still Gaussian)
    and then applying a Gaussian filter to smooth it even more. Then a threshold crosser is applied. It works like
    a trigger algorithm like an RC-CR^2 filter.
    """
    light_inflection_flag = True

    diffs = np.diff(n)
    diffs = gaussian_filter1d(diffs, 10)

    light_max_idx = np.argmax(diffs)
    offset_idx = zero_crosser(diffs, np.argmax(diffs))
    light_next_peak_idx = thresh_crosser(
        diffs,
        a_threshold_in=GAUSSIAN_FILTERED_QPE_THRESHOLD,
        t_start_in=2 * offset_idx - np.argmax(diffs),
    )
    light_pedestal_fit_width = int((light_next_peak_idx - light_max_idx) // 2.2)
    light_peak_distance = light_next_peak_idx - offset_idx

    #     plt.plot(np.arange(len(diffs)), diffs)
    #     plt.scatter(offset_idx, diffs[offset_idx])
    #     plt.scatter(light_next_peak_idx, diffs[light_next_peak_idx])
    #     plt.scatter(2 * offset_idx - np.argmax(diffs), diffs[2 * offset_idx - np.argmax(diffs)], c='r')
    #     plt.show()
    # If no threshold crossings were found, or if the distance between peaks is negative, or if the threshold crossing is too close to the maximum value
    # use the default bin width to fit
    if (
        (light_next_peak_idx == 0)
        or (light_peak_distance < 0)
        or (light_peak_distance < 2 * (offset_idx - np.argmax(diffs)))
    ):
        light_inflection_flag = False
        light_pedestal_fit_width = NUM_BINS_FIT

    return light_inflection_flag, light_pedestal_fit_width, light_peak_distance


def plot_dts(ax, dts, bins=1000, x_range=None):
    ax.hist(dts, bins=bins, range=x_range, histtype="stepfilled", alpha=0.15, color="b")
    n, bins, patches = ax.hist(
        dts, bins=bins, range=x_range, histtype="step", color="b"
    )
    ax.set_xlabel("Inter-times (s)")
    ax.set_ylabel("Counts")
    return n, bins


def fit_exp(x, y):
    x_new = x[y > 0]
    y_new = y[y > 0]
    ln_y = np.log(y_new)
    slope, intercept, r, p, stderr = linregress(x_new, ln_y)
    #     scale = (1 / (1e-9*1e3))
    scale = 1
    return abs(slope) * scale, stderr * scale, slope, intercept


def otte_DCR(scaled_waves, scaled_heights):
    # right now this assumes all the waves are the same length
    time_per_sample = 2e-9
    count = 0
    for x in tqdm(scaled_heights, total=len(scaled_heights)):
        for height in x:
            if height > 0.4:
                count += 1

    return count / (len(scaled_waves) * len(scaled_waves[0]) * time_per_sample)


def hamamatsu_dcr(norm_q_max_tensor, wf_length_time):
    DCR_array = []
    for q_array in norm_q_max_tensor:
        pulses = 0
        for q in q_array:
            if q > 0.5:
                pulses += 1
        DCR_array.append(pulses / wf_length_time)
    return np.mean(DCR_array)


def cross_talk_frac(heights, min_height=0.5, max_height=1.50):
    one_pulses = 0
    other_pulses = 0
    for height_set in tqdm(heights, total=len(heights)):
        if len(height_set) > 0:
            if (height_set[0] > min_height) & (height_set[0] < max_height):
                one_pulses += 1
            else:
                other_pulses += 1
        else:
            continue
    return other_pulses / (one_pulses + other_pulses)


def ap_frac(norm_q_maxes):
    norm_q_maxes = np.array(norm_q_maxes)
    ap = norm_q_maxes[norm_q_maxes < 0.5]
    return len(ap) / len(norm_q_maxes)


def run_dark(
    input_file: str,
    bias: float,
    device_name: str,
    vpp: float,
    temperature: str,
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
    bias
        The bias the SiPM was set at during the run
    device_name
        The name of the device this file corresponds to, one of:
        `sipm`, `sipm_1st`, `sipm_1st_low_gain`, or `apd`
    vpp
        The voltage division CoMPASS was set to record the DAC at
    temperature
        Either RT or LN depending on the measurement type
    output_file
        The name of the file in which to place the output of the calculated photon rates
    lock
        The lock to prevent writing to the same file at the same time

    Notes
    -----
    """
    # Initialize and read in the data ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Grab the output path from the filename
    out_path = "/" + os.path.join(*output_file.split("/")[:-1])
    print(out_path)

    # Create the output plot for this file
    fig, axs = plt.subplots(
        3, 3, figsize=(14, 10)
    )  # create the figure that will hold our subplots
    fig.suptitle(str(bias) + "V " + str(device_name) + " Dark " + str(temperature))

    # Read in the file and convert the waveforms to currents
    sipm_file = h5py.File(input_file, "r")

    waveformies = sipm_file["raw/waveforms"][()]
    t0 = sipm_file["raw/timetag"][()]  # need the timestamps to find inter-event times

    sipm_file.close()

    # Convert the waveforms from adc to current, then plot them
    waveformies = current_converter(waveformies, vpp=vpp, n_bits=14, device="sipm")
    wf_length = len(waveformies[0])
    device = device_name

    time = np.arange(0, wf_length, 1)
    for i in range(0, min(500, len(waveformies))):
        axs[0, 0].plot(time, waveformies[i], linewidth=0.1, alpha=0.95)
    axs[0, 0].set_xlim([0, 600])
    axs[0, 0].set_xlabel("Time [sample]")
    axs[0, 0].set_ylabel("Current [A]")
    axs[0, 0].set_title(
        f"Traces of {device_name} at {bias}V Bias, {temperature} Temperature"
    )

    #### Create a superpulse from 1 p.e. waveforms --------------------------------------------------------------------------------------------------------------------------------------------------------------------
    INTEGRATION_WINDOW_START = 0
    INTEGRATION_WINDOW_END = 300
    DIGITZER_RATE = 2e-9  # s/samples
    NBINS = 500
    SIGMA_CUT = 3
    PRE_TRIG_IDX = 50
    NOISE_START_IDX = 1000  # long enough after trigger to start randomizing the noise

    # Find the one p.e. peak, which should be the tallest if we are triggering correctly on it
    qs = []

    for x in waveformies:
        qs.append(
            np.sum(x[INTEGRATION_WINDOW_START:INTEGRATION_WINDOW_END]) * DIGITZER_RATE
        )

    n, bins = plt.hist(qs, bins=NBINS)  # NOTE: should we use the autorange?
    peak, sigma = tallest_peak_loc_sigma(bins[1:], n)
    qs = np.array(qs)
    one_pe = waveformies[
        (qs < peak + SIGMA_CUT * sigma) & (qs > peak - SIGMA_CUT * sigma)
    ]

    # Now, remove events from the one p.e. peak that have some dark counts spoiling after the trigger

    h = []
    for x in one_pe:
        h.append(np.amax(x))
    n, bins, _ = plt.hist(h, bins=NBINS)

    h = np.array(h)

    peak_height, sigma_height = tallest_peak_loc_sigma(bins[1:], n)
    cut_one_pe = one_pe[
        (h < peak_height + SIGMA_CUT * sigma_height)
        & (h > peak_height - SIGMA_CUT * sigma_height)
    ]

    # Make a further cut on the pre-trigger part of the waveform, to make sure there are no soft-pile-up events
    bl_qs = []

    for x in cut_one_pe:
        bl_qs.append(np.sum(x[:PRE_TRIG_IDX]) * DIGITZER_RATE)
    n, bins, _ = plt.hist(bl_qs, bins=100)
    plt.yscale("log")

    peak, sigma = tallest_peak_loc_sigma(bins[1:], n)

    # finish the cut
    bl_qs = np.array(bl_qs)
    pe_wfs = cut_one_pe[
        (bl_qs < peak + SIGMA_CUT * sigma) & (bl_qs > peak - SIGMA_CUT * sigma)
    ]

    super_pulse = np.mean(pe_wfs, axis=0)
    super_pulse = super_pulse - np.mean(super_pulse[:10])

    fit_pulse = super_pulse

    # Plot the superpulse on top of the one pe wfs we have selected
    time = np.arange(0, wf_length, 1)
    for i in range(0, min(50, len(pe_wfs))):
        axs[0, 1].plot(time, pe_wfs[i], linewidth=0.1, alpha=0.95)
    axs[0, 1].set_xlabel("Time [Sample]")
    axs[0, 1].set_ylabel("Current [A]")
    axs[0, 1].plot(time, super_pulse, c="k")
    axs[0, 1].set_xlim([0, 500])

    # ------------- Find noise waveforms --------------------------------------------------------------------------------------------------
    noise_wfs = []
    for wf in waveformies:
        peaks, _ = find_peaks(wf, height=peak_height - sigma_height)
        if len(peaks) <= 2:
            noise_wfs.append(wf)

    # Do a while loop
    random_idx = 0

    while random_idx <= 100:  # give up after 100 times

        noise_waveform = np.random.choice(
            noise_wfs[random_idx][NOISE_START_IDX:], len(waveformies[random_idx])
        )

        # ensure that the noise_wf is good enough
        if np.amax(noise_waveform) < peak_height - 3 * sigma_height:
            break
        else:
            random_idx += 1  # try another waveform
            noise_waveform = []

    if len(noise_waveform) == 0:
        raise ValueError("Could not find a suitable noise waveform")

    axs[0, 2].plot(time, noise_waveform)
    axs[0, 2].set_xlabel("Time [Sample]")
    axs[0, 2].set_ylabel("Current [A]")

    cut_t0s = t0

    # trying another Wiener filter method:
    c_wfs = waveformies
    superpulse = fit_pulse

    dub = np.zeros_like(c_wfs[0])

    argpulse = np.argmax(superpulse)

    dub[argpulse] = np.amax(superpulse)

    PSF = np.fft.ifft(np.fft.fft(superpulse) / np.fft.fft(dub))[: len(c_wfs[0])]

    data = superpulse

    noise_form = noise_waveform

    f_data = np.fft.fft(data)

    f_pulse = np.fft.fft(superpulse)
    f_psf = np.fft.fft(PSF)
    f_data = np.fft.fft(data)
    f_noise = np.fft.fft(noise_form)

    p_signal = f_data * np.conj(f_data) / len((f_data)) ** 2
    p_noise = f_noise * np.conj(f_noise) / len((f_noise)) ** 2
    p_pulse = f_pulse * np.conj(f_pulse) / len((f_pulse)) ** 2

    w_f = (np.conj(f_psf)) / ((f_psf * np.conj(f_psf)) + (p_noise / p_pulse))

    f_pulse = (f_pulse) / (len(f_pulse))

    f_deconvolution = f_data * w_f
    deconvolution = np.fft.ifft(f_deconvolution)
    # plt.figure()
    # plt.plot(deconvolution)
    # plt.show()

    print(np.amax(data) / np.amax(deconvolution))

    w_f_scale = np.amax(data) / np.amax(deconvolution)

    w_height = np.amax(deconvolution)
    print(w_height)

    # plt.figure()
    # plt.plot(data)
    # plt.show()

    # start = time.time()

    # Set the peak finding threshold, width, and prominence for good SNR waveforms

    threshold = 0.4
    wiener_width = 3
    wiener_prominence = 0.3

    # Set the SNR to divide between good and bad waveforms

    SNR = 5

    # Set the peak finding parameters for the bad SNR waveforms

    bad_snr_threshold = 1
    bad_snr_wiener_width = 11
    bad_snr_wiener_prominence = 1.1

    # Create the arrays to store the data

    d_tp0s = []  # store the tp0s of each peak in all waveform
    d_q_maxes = []  # store the max value of each peak in all waveforms
    max_window = 5  # the window to look for the maximum in

    # Create the Wiener filter ----------------------------------------------------------------------------------------------------------------------------------

    # Create the impulse response, i.e. the spike at tp0

    dub = np.zeros_like(c_wfs[0])
    argpulse = np.argmax(superpulse)
    dub[argpulse] = np.amax(superpulse)

    # Create the point spread function

    PSF = np.fft.ifft(np.fft.fft(superpulse) / np.fft.fft(dub))[: len(c_wfs[0])]

    noise_form = noise_waveform

    f_pulse = np.fft.fft(superpulse)
    f_psf = np.fft.fft(PSF)
    f_noise = np.fft.fft(noise_form)

    p_noise = f_noise * np.conj(f_noise) / len((f_noise)) ** 2
    p_pulse = f_pulse * np.conj(f_pulse) / len((f_pulse)) ** 2

    # Create the Wiener filter

    w_f = (np.conj(f_psf)) / ((f_psf * np.conj(f_psf)) + (p_noise / p_pulse))

    # Code in the difference between actual tp0 and the height of the waveform, should be constant across pulse shapes

    # peak_shift = 6 #Good for Hamamatsu
    peak_shift = 4

    q_array = []
    ts_array = []
    r_tp0s = []

    counter = 0

    dt = 2e-9

    num_analyzed = 0

    plot = False

    time_axis = np.arange(0, len(c_wfs[0])) * 2

    pulses_per_pulse = []

    for data in c_wfs[:]:
        """
        Returns
        -------
        d_q_maxes: flat array containing all the pulse heights of found peaks
        q_array: array of arrays, each sub array contains the pulse heights found in that given waveform
        ts_array: redundant, flat array, this is identical to cut_t0s, just the timestamp of each waveform analyzed
        d_tp0s: an array of arrays, each array contains the sample that a peak was found within that waveform
        r_tp0s: an array of arrays, each array contains the real time that a peak was found at (waveform timestamp+peak position)
        """

        # Perform the wiener filter convolution, then deconvolve it
        f_data = np.fft.fft(data)
        f_deconvolution = f_data * w_f
        corr = np.fft.ifft(f_deconvolution)

        # Scale the deconvolved wiener filtered data by the maximum height of the deconvolved 1pe superpulse

        corr_scale = w_height
        corr = corr / corr_scale

        # This scaling makes it so that one p.e. waveforms have a height of 1, and the threshold can be set to 0.4 for free

        q_temp = []

        ts_array.append(cut_t0s[counter])
        counter += 1

        if (np.amax(np.real(corr)) / np.std(corr)) > SNR:
            peaks = find_peaks(
                corr, height=threshold, width=wiener_width, prominence=wiener_prominence
            )[0]

            real_peaks = []

            ### This block makes sure we aren't just adding noise peaks by accident
            ### It works by integrating the next 100 or so samples and making sure the area
            ### is at least half as much as the area of a 1 p.e. pulse...
            for n, i in enumerate(peaks):
                if len(peaks) > 15:  # crazy noise events removed
                    real_peaks = []
                    break
                if (
                    corr[i] >= 0.69
                ):  # we are pretty sure that these are not afterpulses and are real
                    real_peaks.append(i)
                elif (len(real_peaks) >= 1) and (
                    peaks[n] - real_peaks[-1] < 20
                ):  # ignore the really short time noise bursts
                    pass
                elif (
                    (len(real_peaks) >= 1)
                    and (peaks[n] - real_peaks[-1] < 500)
                    and (np.sum(data[i : i + 150]) * 2e-9 > 0.05e-12)
                ):  # make sure it is an afterpulse...
                    real_peaks.append(i)
                #             elif (len(real_peaks)>=1) and (peaks[n]-real_peaks[-1]<500) and (np.sum(data[i:i+150])*2e-9 < 0.2e-12): # make sure it is an afterpulse...
                #                 pass
                #             elif (np.sum(data[i:i+150])*2e-9 > 0.01e-12):
                #                 real_peaks.append(i)
                else:
                    pass
            peaks = np.array(real_peaks)

            if len(peaks > 0):
                pp = number_peaks_return_to_bl(data, peaks, corr[peaks[0]])
                pulses_per_pulse.append(pp)
                if pp >= 55:
                    plt.plot(data[:2000])
                    plt.show()
            else:
                continue

            #         ### This block is for if there are noise pulses caught by mistake
            #         for i in peaks:
            #             if corr[i]>=1:
            #                 real_peaks.append(i)
            #             if (corr[i]<1) & (np.sum(data[i:i+100])*2e-9 > 0.8e-13):
            #                 real_peaks.append(i)
            #             else:
            #                 pass
            #         peaks = real_peaks

            #         peaks = peaks[peaks>2*peak_shift]
            # d_q_maxes.append(data[peaks])

            err = np.amax(np.real(corr)) / np.std(corr)
            corr = corr * corr_scale

            d_q_maxes.extend(
                np.real(corr[peaks] * w_f_scale)
            )  # we rescale the data by the scaling factor the superpulse had after deconvolution, this is a decent approximation
            q_temp.append(np.real(corr[peaks] * w_f_scale))

            if (
                False
                and (np.real(corr[peaks] / corr_scale < 0.55)).any()
                and (np.min(np.diff(peaks)) < 100)
            ):  # Check to see what noise we are catching
                print("bad noise")

                peaks = np.array(peaks) - peak_shift
                figures, (ax_orig, ax_corr) = plt.subplots(
                    2, 1, sharex=True, figsize=(12, 12)
                )
                ax_orig.plot(time_axis, data)
                # ax_orig.scatter(tp0s,data[tp0s], c='r')
                ax_orig.scatter(time_axis[peaks], data[peaks], s=90, c="r", label="tp0")
                ax_orig.set_xlabel("Samples")
                ax_orig.set_ylabel("ADC")
                ax_orig.set_title("Waveform")
                ax_orig.legend()

                print(peaks, data[peaks])

                peaks = np.array(peaks) + peak_shift
                # ax_corr.plot(corr*w_f_scale)
                # ax_corr.scatter(peaks,corr[peaks]*w_f_scale, label='E')
                ax_corr.plot(time_axis, (corr / corr_scale))
                ax_corr.scatter(time_axis[peaks], corr[peaks] / corr_scale, label=err)
                ax_corr.set_title("Time Domain Wiener filtered waveform")
                ax_corr.set_xlabel("Time [ns]")
                ax_corr.set_ylabel("ADC")
                ax_corr.legend()
                plt.xlim([0, 2000])

                plt.show()

            peaks = (
                np.array(peaks) - peak_shift
            )  # justified by assuming standard pulse shape a priori

            d_tp0s.append(peaks)
            r_tp0s.append(
                cut_t0s[counter - 1] * 1e-12 + peaks * 2e-9
            )  # time stamp is in picoseconds, peak position in samples
            q_array.extend(q_temp)

            num_analyzed += 1

            if plot:
                figures, (ax_orig, ax_corr) = plt.subplots(
                    2, 1, sharex=True, figsize=(12, 12)
                )
                ax_orig.plot(time_axis, data)
                # ax_orig.scatter(tp0s,data[tp0s], c='r')
                ax_orig.scatter(time_axis[peaks], data[peaks], c="r", label="tp0")
                ax_orig.set_xlabel("Time [ns]")
                ax_orig.set_ylabel("Current [A]")
                ax_orig.set_title(f"Hamamatsu at {bias} V Bias LN Waveform")
                ax_orig.legend()

                print(peaks, data[peaks])

                peaks = peaks + peak_shift
                # ax_corr.plot(corr*w_f_scale)
                # ax_corr.scatter(peaks,corr[peaks]*w_f_scale, label='E')
                ax_corr.plot(time_axis, corr / corr_scale)
                ax_corr.scatter(time_axis[peaks], corr[peaks] / corr_scale, label=err)
                ax_corr.set_title("Time Domain Wiener filtered waveform")
                ax_corr.set_xlabel("Time [ns]")
                ax_corr.set_ylabel("Current [A]")
                #             ax_corr.legend()

                plt.xlim([0, 10000])

                plt.show()

    #     else:
    #         peaks = find_peaks(corr, height=bad_snr_threshold, width=bad_snr_wiener_width, prominence=bad_snr_wiener_prominence)[0]
    #         # d_q_maxes.append(data[peaks])
    #         err = 1/np.std(corr)
    #         corr = corr*corr_scale

    #         d_q_maxes.extend(np.real(corr[peaks]*w_f_scale))
    #         q_temp.append(np.real(corr[peaks]*w_f_scale))

    #         peaks = peaks-peak_shift # justified by assuming standard pulse shape a priori

    #         d_tp0s.append(peaks)
    #         r_tp0s.append(cut_t0s[counter-1]*1e-12+peaks*2e-9)

    #         q_array.extend(q_temp)

    #         fig, (ax_orig, ax_corr) = plt.subplots(2, 1, sharex=True,figsize=(12,12))
    #         ax_orig.plot(data)
    #         # ax_orig.scatter(tp0s,data[tp0s], c='r')
    #         ax_orig.scatter(peaks,data[peaks], c='r', label=err)
    #         ax_orig.legend()

    #         peaks=peaks+peak_shift
    # #         ax_corr.plot(corr)
    # #         ax_corr.scatter(peaks,corr[peaks])

    #         ax_corr.plot(corr/corr_scale)
    #         ax_corr.scatter(peaks,corr[peaks]/corr_scale, label=err)

    # print(d_q_maxes)
    # end = time.time()
    # print(end-start)

    # Do the dark analysis ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Normalize the PHS
    n, bins, _ = axs[1, 0].hist(d_q_maxes, bins=NBINS)
    axs[1, 0].set_yscale("log")
    axs[1, 0].set_title("PHS from LN Teststand Data after Wiener filter")
    axs[1, 0].set_xlabel("Pulse Height [ADC]")
    axs[1, 0].set_ylabel("Counts")

    #### First find the FWHM of the tallest peak
    peak, sigma = tallest_peak_loc_sigma(bins[1:], n)

    (
        light_inflection_flag,
        light_pedestal_fit_width,
        light_peak_distance,
    ) = best_guess_peak_distances(n)

    bin_width = bins[1] - bins[0]
    fit_width = int(2.355 * sigma / bin_width)  # fit the FWHM of the pedestal

    if not light_inflection_flag:
        light_peak_distance = 2.355 * NUM_SIGMA_AWAY * sigma

    else:
        light_peak_distance *= bin_width

    peak_distance = light_peak_distance

    print(peak_distance)

    print((peak_distance) * (bins[1] - bins[0]))
    peaks, peak_locs, amplitudes = guess_peaks_no_width(n, bins, 4, peak_distance)

    if len(peaks) == 2:
        max_peaks = 2
    else:
        max_peaks = 3

    peaks = peaks[:max_peaks]
    peak_locs = peak_locs[:max_peaks]
    amplitudes = amplitudes[:max_peaks]

    gauss_params, gauss_errors = fit_peaks_no_sigma_guess(
        n, bins, peaks, peak_locs, amplitudes, fit_width=fit_width
    )

    peaks = []

    for i, x in enumerate(gauss_params):
        peaks.append(x[1])

    x = np.linspace(bins[0], bins[-1], NBINS)

    for i, params in enumerate(gauss_params):
        axs[1, 1].plot(x, gaussian(x, *params))

    axs[1, 1].step(bins[1:], n)
    axs[1, 1].set_xlabel("Integrated Charge (C)")
    axs[1, 1].set_ylabel("Counts")
    axs[1, 1].set_ylim(1, np.amax(n))
    axs[1, 1].grid(True)
    axs[1, 1].set_title("Light QPE Peaks Fit")
    axs[1, 1].set_yscale("log")

    npe = np.arange(0, len(peaks[:]) + 1)
    slope, intercept, *_ = linregress(npe, [0, *peaks[:]])

    def line(x, m, b):
        return m * x + b

    axs[1, 2].scatter(npe, [0, *peaks[:]])
    axs[1, 2].plot(npe, line(npe, slope, intercept))
    axs[1, 2].set_ylabel("Amplitude [A]")
    axs[1, 2].set_xlabel("NPE")

    norm_q_maxes = ((d_q_maxes) - intercept) / slope
    norm_q_array = (q_array - intercept) / slope

    norm_ps_peaks = (peaks - intercept) / slope

    axs[2, 0].hist(norm_q_maxes, bins=NBINS, range=(0, 5))
    axs[2, 0].set_yscale("log")
    axs[2, 0].set_title(f"PHS for {temperature} {device} at {bias} V Bias")
    axs[2, 0].set_xlabel("NPE")
    axs[2, 0].set_ylabel("Counts")

    # Create the time-to-next event arrays

    r_tp0s = np.array(r_tp0s)
    flat_r_tp0s = np.hstack(r_tp0s)

    norm_q_array = np.array(norm_q_array)
    q_flat = np.hstack(norm_q_array)

    # set the one_pe sensitivity

    one_pe_sensitivity = 0.5

    # Create the dt array using the timestamps of each waveform added to the values
    # consider delay times between waveforms
    all_dts = []
    all_dt_amps = []

    previous_flag = False
    for i, tp0 in enumerate(flat_r_tp0s):
        if (
            (q_flat[i] > 1 - one_pe_sensitivity)
            & (q_flat[i] < 1 + one_pe_sensitivity)
            & (not previous_flag)
            & (i + 1 < len(flat_r_tp0s))
        ):
            all_dt_amps.append(q_flat[i + 1])
            all_dts.append(flat_r_tp0s[i + 1] - flat_r_tp0s[i])
            previous_flag = True
        else:
            previous_flag = False

    all_scaled = np.array(all_dts) / DIGITZER_RATE
    # plt.hist2d(all_scaled,all_dt_amps, bins=100, range=[[0,200],[0,2]])
    # plt.title('Inter-event Arrival Times')
    # plt.xlabel('Time [Sample]')
    # plt.ylabel('Signal [p.e.]')
    # plt.colorbar()
    # plt.show()

    all_scaled = np.array(all_dts)

    n = 100**2
    # set the max and min of the log plot we will make
    x_bins = np.logspace(np.log10(2e-9), np.log10(15), int(np.sqrt(n)))  # time in s
    y_bins = np.linspace(0, 3, int(np.sqrt(n)))  # amp in p.e.
    H, xedges, yedges = np.histogram2d(all_scaled, all_dt_amps, bins=[x_bins, y_bins])

    # fig = plt.figure(figsize=(12,8))
    # ax1 = fig.add_subplot(111)
    # ax1.plot(all_scaled, all_dt_amps, 'o')
    # ax1.set_xscale('log')

    new = axs[2, 1].pcolormesh(xedges, yedges, H.T, norm=mpl.colors.LogNorm())
    plt.colorbar(new, ax=axs[2, 1], label="Counts")
    axs[2, 1].set_xscale("log")
    axs[2, 1].set_xlabel("Time From Previous Pulse [s]")
    axs[2, 1].set_ylabel("Amplitude [p.e.]")

    # ax2.axvline(10000*2e-9)
    axs[2, 1].axvline(10e-9)  # AP lower time limit
    # ax2.axvline(250e-9) # AP upper time limit for hamamatsu
    axs[2, 1].axvline(2000e-9)  # AP upper time limit for ketek
    axs[2, 1].axhline(0.8)  # upper AP limit
    axs[2, 1].set_title(f"{temperature} Inter-event Times\n at {bias} V Reverse Bias")

    # Compute the DCR ------------------------------------------------------------------------------------------------------------------------------
    restrictive_d_times = np.array(all_dts)
    restrictive_amps = np.array(all_dt_amps)

    lower_time = 5e-2
    upper_time = 6e-1
    dt_cut = (restrictive_d_times > lower_time) & (restrictive_d_times < upper_time)

    lower_amp = 0.5
    upper_amp = 10
    amp_cut = (restrictive_amps > lower_amp) & (restrictive_amps < upper_amp)

    n, bins = plot_dts(
        axs[2, 2],
        restrictive_d_times[amp_cut & dt_cut],
        x_range=[lower_time, upper_time],
        bins=100,
    )
    axs[2, 2].set_yscale("log")

    bin_centers = (bins[1:] + bins[:-1]) / 2
    rate, error, slope, intercept = fit_exp(bin_centers, n)

    inter_times = np.linspace(lower_time, upper_time, 100)
    exp_func = np.exp(intercept + slope * inter_times)
    label = "Hz"

    n, bins = plot_dts(
        axs[2, 2],
        restrictive_d_times[amp_cut & dt_cut],
        x_range=[lower_time, upper_time],
        bins=100,
    )
    if label == "kHz":
        axs[2, 2].plot(
            inter_times,
            exp_func,
            color="red",
            alpha=0.5,
            label=rf"DCR = {round(rate/1e3, 1)}  $\pm$ {round(error/1e3, 2)} kHz",
        )
    elif label == "Hz":
        axs[2, 2].plot(
            inter_times,
            exp_func,
            color="red",
            alpha=0.5,
            label=rf"{round(rate, 1)} Hz $\pm$ {round(error, 2)}",
        )
    axs[2, 2].legend()
    axs[2, 2].set_yscale("log")

    axs[2, 2].set_title(
        f"Inter-event Times for {device}\n at {bias} V Bias at LN Temperature"
    )

    dcr = otte_DCR(c_wfs, norm_q_array)

    dcr = hamamatsu_dcr(norm_q_array, wf_length_time=10000 * 2e-9)

    dcr = rate
    dcr_err = error

    cross_talk_frac(norm_q_array, min_height=0.5, max_height=1.50)

    max_charge = 1.5
    min_charge = 0.5
    restricted_dt_amps = np.array(restrictive_amps)
    cross_events = (
        np.array(restricted_dt_amps)[restricted_dt_amps > max_charge]
    ).shape[0]
    total_events = (
        np.array(restricted_dt_amps)[restricted_dt_amps > min_charge]
    ).shape[0]

    error = np.sqrt(
        (cross_events / total_events**2) + (cross_events**2 / total_events**3)
    )
    print(cross_events / total_events, error)

    CT = cross_events / total_events
    CT_err = error

    ap_frac(norm_q_maxes)

    max_charge = 0.8
    min_charge = 0.0

    ap_start = 10 * 2e-9
    # ap_end = 250*2e-9 # for hamamatsu
    ap_end = 2000e-9  # for ketek

    restricted_dt_amps = np.array(restrictive_amps)
    restricted_dts = np.array(restrictive_d_times)
    ap_events = (
        np.array(restricted_dt_amps)[
            (restricted_dt_amps < max_charge)
            & (restricted_dts < ap_end)
            & (restricted_dts > ap_start)
        ]
    ).shape[0]
    total_events = (
        np.array(restricted_dt_amps)[restricted_dt_amps > min_charge]
    ).shape[0]

    error = np.sqrt(
        (ap_events / total_events**2) + (ap_events**2 / total_events**3)
    )
    print(ap_events / total_events, error)

    AP = ap_events / total_events
    AP_err = error

    # ------------------------------------------------------------------------------------------------------------------------------------------------------
    # Save it all to a file
    fig_path = (
        out_path + "/monitoring_plots_" + str(bias) + "_" + str(device_name) + ".png"
    )
    fig.tight_layout()
    fig.savefig(fig_path, dpi=fig.dpi)

    lock.acquire()
    f = h5py.File(output_file, "a")

    if f"{device}/{bias}/tp0s" in f:
        f[f"{device}/{bias}/tp0s"][:] = flat_r_tp0s
        f[f"{device}/{bias}/norm_qs"][:] = q_flat
        f[f"{device}/{bias}/DCR"][:] = dcr
        f[f"{device}/{bias}/DCR_err"][:] = dcr_err
        f[f"{device}/{bias}/CT"][:] = CT
        f[f"{device}/{bias}/CT_err"][:] = CT_err
        f[f"{device}/{bias}/AP"][:] = AP
        f[f"{device}/{bias}/AP_err"][:] = AP_err

    else:
        dset = f.create_dataset(f"{device}/{bias}/tp0s", data=flat_r_tp0s)
        dset = f.create_dataset(f"{device}/{bias}/norm_qs", data=q_flat)
        dset = f.create_dataset(f"{device}/{bias}/DCR", data=dcr)
        dset = f.create_dataset(f"{device}/{bias}/DCR_err", data=dcr_err)
        dset = f.create_dataset(f"{device}/{bias}/CT", data=CT)
        dset = f.create_dataset(f"{device}/{bias}/CT_err", data=CT_err)
        dset = f.create_dataset(f"{device}/{bias}/AP", data=AP)
        dset = f.create_dataset(f"{device}/{bias}/AP_err", data=AP_err)

    f.close()
    lock.release()
