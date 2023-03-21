import pytest
import numpy as np

from sipm_studio.dsp.gain_processors import (
    normalize_charge,
    my_gain,
    line,
    slope_fit_gain,
)

# define physical constants
e_charge = 1.6e-19


def test_normalize_charge():
    gain = 1000
    peak_locs = [0, 1000, 2000, 3000]  # start the pedestal at 0
    peak_locs_errors = [0.01, 10, 20, 30]
    charges = np.arange(0, 3000)

    gain_e, normalized_charges, gain_error = normalize_charge(
        charges, peak_locs, peak_locs_errors
    )

    assert np.array_equal(normalized_charges, charges / 1000)
    assert pytest.approx(gain_e * e_charge, 1e-1) == 1000
    assert pytest.approx(gain_error * e_charge, 1e-1) == 40


def test_my_gain():
    gain = 1000
    peak_locs = [0, 1000, 2000, 3000]  # start the pedestal at 0
    peak_locs_errors = [0.01, 10, 20, 30]
    charges = np.arange(0, 3000)

    gain_array = my_gain(peak_locs, peak_locs_errors)
    assert gain_array[0] * e_charge == gain
    assert pytest.approx(gain_array[1] * e_charge, 1e-1) == 10


def test_line():
    xs = np.ones(10)
    m = 5
    b = 0
    ys = line(xs, m, b)
    assert np.array_equal(ys, np.full_like(ys, 5))


def test_slope_fit_gain():
    gain = 1000
    peak_locs = [0, 1000, 2000, 3000]  # start the pedestal at 0
    peak_locs_errors = [0.01, 10, 20, 30]
    charges = np.arange(0, 3000)

    gain_array = slope_fit_gain(peak_locs, pedestal=True)
    assert gain_array[0] * e_charge == gain
    assert pytest.approx(gain_array[1] * e_charge, 1e-1) == 0

    gain_array = slope_fit_gain(peak_locs, pedestal=False)
    assert gain_array[0] * e_charge == gain
    assert pytest.approx(gain_array[1] * e_charge, 1e-1) == 0
