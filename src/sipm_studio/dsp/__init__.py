"""
Submodule for tools dedicated to processing SiPM waveforms
"""

from sipm_studio.dsp.adc_to_current import current_converter
from sipm_studio.dsp.charge_to_photons import sipm_photons, apd_photons
from sipm_studio.dsp.current_to_charge import rando_integrate_current, integrate_current
from sipm_studio.dsp.gain_processors import normalize_charge, my_gain, slope_fit_gain
from sipm_studio.dsp.qpe_peak_finding import (
    gaussian,
    guess_peaks,
    guess_peaks_no_width,
    fit_peaks,
    fit_peak,
)

__all__ = [
    "current_converter",
    "sipm_photons",
    "apd_photons",
    "rando_integrate_current",
    "integrate_current",
    "normalize_charge",
    "my_gain",
    "slope_fit_gain",
    "gaussian",
    "guess_peaks",
    "guess_peaks_no_width",
    "fit_peaks",
    "fit_peak",
]
