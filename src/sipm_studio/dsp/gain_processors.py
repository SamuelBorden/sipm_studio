"""
Processors to calculate gain from a SiPM charge histogram
"""

import numpy as np
from scipy.stats import linregress
from uncertainties import ufloat, unumpy
from uncertainties.umath import *

# define physical constants
e_charge = 1.6e-19


def normalize_charge(
    charges: np.array, peak_locs: np.array, peak_errors: np.array
) -> tuple(float, np.array, float):
    """
    Given an array of charges, return the gain, normalized charges, or, number of photoelectrons, and the gain error

    Parameters
    ----------
    charges
        Array of SiPM charges, in Coulombs
    peak_locs
        The locations of photoelectron peaks in the charge histogram
    peak_errors
        The errors associated with the peak locations
    """
    x0 = peak_locs[0]
    peak_locs = np.array(peak_locs)[1:]

    peak_locs_u = []
    for i in range(len(peak_locs)):
        peak_locs_u.append(ufloat(peak_locs[i], peak_errors[i]))

    peak_locs_u = np.array(peak_locs_u)

    peak_errors = np.array(peak_errors)[1:]

    peak_diffs = peak_locs[1:] - peak_locs[:-1]
    peak_diffs_u = peak_locs_u[1:] - peak_locs_u[:-1]

    diff_errors = np.sqrt((peak_errors[1:] + peak_errors[:-1]) ** 2)
    gain = np.sum(peak_diffs * diff_errors) / np.sum(diff_errors)
    gain_e = np.sum(diff_errors) / len(diff_errors)

    gain_u = np.sum(peak_diffs_u) / len(peak_diffs_u)
    return gain / e_charge, (charges - x0) / gain, gain_e / e_charge


def my_gain(peak_locs: np.array, peak_err: np.array) -> np.array:
    """
    The differs from :func:`normalize_charge` by not subtracting the pedestal; it only takes the differences of all peaks found into account

    Parameters
    ----------
    peak_locs
        The locations of photoelectron peaks in the charge histogram
    peak_errors
        The errors associated with the peak locations
    """

    peak_locs_u = []
    for i in range(len(peak_locs)):
        peak_locs_u.append(ufloat(peak_locs[i], peak_err[i]))
    peak_locs_u = np.array(peak_locs_u)
    peak_diffs = np.diff(peak_locs)
    peak_diffs_u = np.diff(peak_locs_u)
    gain = np.sum(peak_diffs) / len(peak_diffs)
    gain_u = np.sum(peak_diffs_u) / len(peak_diffs_u)
    return np.array([gain_u.n / e_charge, gain_u.s / e_charge])


def line(x: np.array, m: float, b: float) -> np.array:
    """
    Return a line, used to fit the peak locations from a charge histogram
    """
    return m * x + b


def slope_fit_gain(peak_locs: np.array, pedestal: bool = False) -> np.array:
    """
    Returns the gain by fitting a line to the peak locations from a charge histogram and returning its slope

    Parameters
    ----------
    peak_locs
        The locations of photoelectron peaks in the charge histogram
    pedestal
        Set to true if there is a pesestal recorded in the charge spectrum
    """
    if pedestal:
        npe = np.arange(0, len(peak_locs))
    else:
        npe = np.arange(1, len(peak_locs) + 1)

    slope, intercept, r, p, se = linregress(npe, peak_locs)

    return np.array([slope / e_charge, se / e_charge])
