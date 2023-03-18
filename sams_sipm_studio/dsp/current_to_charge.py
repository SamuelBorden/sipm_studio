import numpy as np 

def rando_integrate_current(current_forms, width, sample_time=2e-9):
    """
    integrate a randomly section of a current waveform using a window of provided width. 
    Useful if there is a high afterpulsing rate. 
    """
    start_range = width
    stop_range = current_forms.shape[1] - width - 1
    start = np.random.randint(start_range, stop_range)
    stop = start + width
    return np.sum(current_forms.T[start:stop].T, axis=1)*sample_time

def integrate_current(current_forms, lower_bound=0, upper_bound=200, sample_time=2e-9):
    """
    Integrate each current waveform from the lower bound (in samples) to the upper bound, and multiply
    by the sampling rate to convert from samples to seconds. Could play around with numerically integrating
    the current waveforms, but a simple sum works well enough to convert to charge. 
    """
    return np.sum(current_forms.T[lower_bound:upper_bound].T, axis=1)*sample_time
