"""
Peak finding and fitting routines to a charge spectrum, usually obtained through current_to_charge.py functions
"""
import numpy as np
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
from scipy.signal import find_peaks


def gaussian(x: np.array, A: float, mu: float, sigma: float) -> np.array:
    """
    Return a Gaussian, used to fit parts of a charge spectrum

    Parameters
    ----------
    x
        Input data
    A
        Amplitude of Gaussian
    mu
        Median of Gaussian
    sigma
        Standard deviation of Gaussian
    """
    return A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def guess_peaks(
    n: np.array, bins: np.array, min_height: float, min_dist: float, min_width: float
) -> tuple[np.array, np.array, np.array]:
    """
    Routine for guessing the location of peaks in a charge spectrum. Uses `scipy`'s `find_peaks` with height, distance, and width parameters.

    Parameters
    ----------
    n
        The counts in each bin of a charge histogram
    bins
        The bins of a charge histogram
    min_height
        The minimum height to look for peaks in the histogram
    min_dist
        The minimum distance between peaks to look for, NOT in number of bins
    min_width
        The minimum width for a valid peak to search for, NOT in number of bins
    """
    bin_width = bins[1] - bins[0]
    min_bin_dist = min_dist / bin_width
    min_bin_width = min_width / bin_width
    peaks, amplitudes = find_peaks(
        n, height=min_height, distance=min_bin_dist, width=min_bin_width
    )
    return peaks, bins[peaks], amplitudes["peak_heights"]


def guess_peaks_no_width(
    n: np.array, bins: np.array, min_height: float, min_dist: float
) -> tuple[np.array, np.array, np.array]:
    """
    Routine for guessing the location of peaks in a charge spectrum. Uses `scipy`'s `find_peaks` with height and distance parameters.

    Parameters
    ----------
    n
        The counts in each bin of a charge histogram
    bins
        The bins of a charge histogram
    min_height
        The minimum height to look for peaks in the histogram
    min_dist
        The minimum distance between peaks to look for, NOT in number of bins
    """
    bin_width = bins[1] - bins[0]
    min_bin_dist = min_dist / bin_width
    peaks, amplitudes = find_peaks(n, height=min_height, distance=min_bin_dist)
    return peaks, bins[peaks], amplitudes["peak_heights"]


def fit_peaks(
    n: np.array,
    bins: np.array,
    peaks: np.array,
    peak_locs: np.array,
    amplitudes: np.array,
    fit_width: float = 15,
) -> tuple[np.array, np.array]:
    """
    Routine for fitting identified peaks in a charge spectrum with a Gaussian function.

    Parameters
    ----------
    n
        The counts in each bin of a charge histogram
    bins
        The bins of a charge histogram
    peaks
        The bin location of a peak in a charge histogram
    peak_locs
        The location of a peak in Coulombs in a charge histogram
    amplitudes
        The amplitude, in counts, of a peak in the charge histogram
    fit_width
        The number of bins which to fit around each peak

    Returns
    -------
    gauss_params
        A list of three tuples, each tuple is of the form (amplitude, mu, sigma)
    gauss_errors
        A list of three tuples, each tuple is of the form (amplitude error, mu error, sigma error)
    """
    gauss_params = []
    gauss_errors = []
    bin_centers = (bins[1:] + bins[:-1]) / 2
    sigma_guess = peak_locs[1] - peak_locs[0]
    for i, peak in enumerate(peaks):
        left = peak - fit_width
        right = peak + fit_width
        if left < 0:
            left = 0
        coeffs, covs = curve_fit(
            gaussian,
            bin_centers[left:right],
            n[left:right],
            p0=[amplitudes[i], peak_locs[i], sigma_guess],
        )
        gauss_params.append(coeffs)
        gauss_errors.append(np.sqrt(np.diag(covs)))
    return gauss_params, gauss_errors


def fit_peak(
    n: np.array,
    bins: np.array,
    peaks: np.array,
    peak_locs: np.array,
    amplitudes: np.array,
    fit_width: float = 15,
) -> tuple[np.array, np.array]:
    """
    fit one peak only, used in light analysis. Routine for fitting identified peaks in a charge spectrum with a Gaussian function.

    Parameters
    ----------
    n
        The counts in each bin of a charge histogram
    bins
        The bins of a charge histogram
    peaks
        The bin location of a peak in a charge histogram
    peak_locs
        The location of a peak in Coulombs in a charge histogram
    amplitudes
        The amplitude, in counts, of a peak in the charge histogram
    fit_width
        The number of bins which to fit around each peak

    Returns
    -------
    gauss_params
        A list of three tuples, each tuple is of the form (amplitude, mu, sigma)
    gauss_errors
        A list of three tuples, each tuple is of the form (amplitude error, mu error, sigma error)
    """
    gauss_params = []
    gauss_errors = []
    bin_centers = (bins[1:] + bins[:-1]) / 2
    sigma_guess = bins[np.argmax(n)] - bins[0]
    for i, peak in enumerate(peaks):
        left = peak - fit_width
        right = peak + fit_width
        if left < 0:
            left = 0
        coeffs, covs = curve_fit(
            gaussian,
            bin_centers[left:right],
            n[left:right],
            p0=[amplitudes[i], peak_locs[i], sigma_guess],
        )
        gauss_params.append(coeffs)
        gauss_errors.append(np.sqrt(np.diag(covs)))
    return gauss_params, gauss_errors


def fit_peaks_no_sigma_guess(n, bins, peaks, peak_locs, amplitudes, fit_width=15):
    gauss_params = []
    gauss_errors = []
    bin_centers = (bins[1:] + bins[:-1]) / 2
    sigma_guess = (np.amax(bins)) / 10
    for i, peak in enumerate(peaks):
        left = peak - fit_width
        right = peak + fit_width
        if left < 0:
            left = 0
        coeffs, covs = curve_fit(
            gaussian,
            bin_centers[left:right],
            n[left:right],
            p0=[amplitudes[i], peak_locs[i], sigma_guess],
        )
        gauss_params.append(coeffs)
        gauss_errors.append(np.sqrt(np.diag(covs)))
    return gauss_params, gauss_errors
