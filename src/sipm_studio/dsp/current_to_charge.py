"""
Processors to integrate current waveforms to convert them to an array of charges
"""
import numpy as np


def rando_integrate_current(
    current_forms: np.array, width: int, sample_time: float = 2e-9
) -> np.array:
    """
    Integrate a randomly section of a current waveform using a window of provided width.
    Useful if there is a high afterpulsing rate.

    Parameters
    ----------
    current_forms
        An array of current waveforms
    width
        The width of integration window
    sample_time
        The sampling rate of the ADC
    """
    start_range = width
    stop_range = current_forms.shape[1] - width - 1
    start = np.random.randint(start_range, stop_range)
    stop = start + width
    return np.sum(current_forms.T[start:stop].T, axis=1) * sample_time


def integrate_current(
    current_forms: np.array,
    lower_bound: int = 0,
    upper_bound: int = 200,
    sample_time: float = 2e-9,
) -> np.array:
    """
    Integrate each current waveform from the lower bound (in samples) to the upper bound, and multiply
    by the sampling rate to convert from samples to seconds. Could play around with numerically integrating
    the current waveforms, but a simple sum works well enough to convert to charge.

    Parameters
    ----------
    current_forms
        An array of current waveforms
    lower_bound
        The lower index of the integration window
    upper_bound
        The upper index of the integration window
    sample_time
        The sampling rate of the ADC
    """
    return np.sum(current_forms.T[lower_bound:upper_bound].T, axis=1) * sample_time
