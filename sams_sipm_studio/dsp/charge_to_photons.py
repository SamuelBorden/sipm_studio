"""
Processors for converting charge to photons for APDs and SiPMs
"""
from uncertainties import ufloat, unumpy
import numpy as np

# define physical constants
e_charge = 1.6e-19
h = 6.64e-34
c = 3e8


def sipm_photons(charge: np.array, charge_err: np.array, gain: ufloat) -> float:
    """
    Convert a given charge collected from a SiPM into number of incident photons, using the provided gain.
    The inputs can be the location of a peak and its associated error from a charge histogram.

    Parameters
    ----------
    charge
        An array containing only the charge collected at an APD
    charge_err
        An array containing only the error on the charge collected at an APD
    gain
        The gain of the SiPM

    TODO: change charge type from array to just a float, need to change in the actual processor calls
    """

    peak_loc_u = ufloat(charge[0], charge_err[0])
    return peak_loc_u / (gain * e_charge)


def apd_photons(
    charge: np.array, charge_err: np.array, photosensitivity: float = 24, l: float = 560
) -> float:
    """
    Convert a given charge collected in an APD into a number of incident photons, using the provided photosensitivity.
    The inputs can be the location of a peak and its associated error from a charge histogram.

    Parameters
    ----------
    charge
        An array containing only the charge collected at an APD
    charge_err
        An array containing only the error on the charge collected at an APD
    photosensitivity
        The photosensitivity of the APD in A/W
    l
        The wavelength of incident light in nm
    """
    charge = ufloat(charge[0], charge_err[0])
    photosensitivity = ufloat(photosensitivity, 1)
    l = ufloat(l, 10)
    # this conversion factor is determined by the integration over the LED spectrum
    conversion_factor = 1.1880412517513259e17  # in units of gamma/coulomb
    return ((charge / photosensitivity) * (l * 1e-9)) / (h * c)


#     return charge*conversion_factor


def apd_charge_photons(
    light_charge: np.array,
    light_charge_err: np.array,
    dark_charge: np.array,
    dark_charge_err: np.array,
) -> float:
    """
    Convert a given charge collected in an APD into a number of incident photons, using the APD's quantum efficiency.
    The inputs can be the location of a peak and its associated error from a charge histogram.

    Parameters
    ----------
    light_charge
        An array containing only the charge collected at an APD during illumination
    light_charge_err
        An array containing only the error on the charge collected at an APD during illumination
    dark_charge
        An array containing only the charge collected at an APD during dark conditions
    dark_charge_err
        An array containing only the error on the charge collected at an APD during dark conditions
    """
    light = ufloat(light_charge[0], light_charge_err[0])
    dark = ufloat(dark_charge[0], dark_charge_err[0])
    M = ufloat(50, 1)
    QE = ufloat(0.7, 0.1)
    return (light - dark) / (QE * M * e_charge)
