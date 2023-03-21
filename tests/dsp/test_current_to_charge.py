import pytest
import numpy as np

from sipm_studio.dsp.current_to_charge import rando_integrate_current, integrate_current


def test_rando_integrate_current():
    # If we feed it an array of ones, then it won't matter where we randomly integrate over so the test will always pass
    current_forms = np.ones((10, 1000))
    sample_time = 2e-9
    charges = rando_integrate_current(current_forms, 200, 2e-9)
    expected_charges = np.full_like(charges, 200 * 2e-9)
    assert np.array_equal(charges, expected_charges)


def test_integrate_current():
    current_forms = np.full((10, 1000), np.arange(0, 1000))
    sample_time = 2e-9
    lower_bound = 0
    upper_bound = 100
    # the sum of the first 100 integers is 100*99/2
    charges = integrate_current(current_forms, lower_bound, upper_bound, sample_time)
    expected_charges = np.full_like(charges, 50 * 99 * 2e-9)
    assert np.array_equal(charges, expected_charges)
