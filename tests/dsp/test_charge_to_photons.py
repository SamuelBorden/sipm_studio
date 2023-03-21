import pytest
import numpy as np
from uncertainties import ufloat

from sipm_studio.dsp.charge_to_photons import (
    sipm_photons,
    apd_photons,
    apd_charge_photons,
)

# Define physical constants for testing
e_charge = 1.6e-19
h = 6.64e-34
c = 3e8


def test_sipm_photons():
    gain = 1000
    gain_err = 10
    gain_u = ufloat(gain, gain_err)

    charge = 10000
    charge *= e_charge

    charge_error = 1000
    charge_error *= e_charge

    # With the gain and charge defined we would expect 10 photons
    photons = sipm_photons([charge], [charge_error], gain_u)
    assert photons.n == 10
    assert (
        pytest.approx(photons.s, 1e-2) == 1
    )  # the error should be around 1 photon, plus/minus error propagation


def test_apd_photons():
    # photosensitivity is in A/W
    photosensitivity = 100
    # lambda is in nm and convert it to watts
    lamb = 500 / 1e-9 * h * c

    # so we expect an overall gain of 5
    charge = 100

    charge_error = 1

    photons = apd_photons([charge], [charge_error], photosensitivity, lamb)

    assert photons.n == 500
    assert (
        pytest.approx(photons.s, 1) == 10
    )  # expect 10 10% error because of the uncertainty on the wavelength


def test_apd_charge_photons():
    light_charge = 100
    light_charge_error = 1

    dark_charge = 20
    dark_charge_error = 1

    # The QE = hardcoded as 0.7 and the gain is 50, so we should expect 80/35 photons
    photons = apd_charge_photons(
        [light_charge * e_charge],
        [light_charge_error * e_charge],
        [dark_charge * e_charge],
        [dark_charge_error * e_charge],
    )
    assert photons.n == 80 / 35
    assert pytest.approx(photons.s, 1e-2) == 0.33
