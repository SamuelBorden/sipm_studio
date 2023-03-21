import pytest
import numpy as np
from scipy.stats import norm

from sipm_studio.dsp.qpe_peak_finding import (
    gaussian,
    guess_peaks,
    guess_peaks_no_width,
    fit_peaks,
    fit_peak,
)


def test_gaussian():
    xs = np.arange(0, 100)
    mu = 10
    sigma = 1.5
    A = 10

    scipy_gaussian = (
        A * norm.pdf(xs, mu, sigma) * np.sqrt(2 * np.pi) * sigma
    )  # differs from our normalization by sqrt(2*pi)*sigma
    sipm_gaussian = gaussian(xs, A, mu, sigma)

    assert np.allclose(scipy_gaussian, sipm_gaussian, 1e-5)


def test_guess_peaks():
    """
    Create a spectrum with two gaussian peaks, well separated at mu1 and mu2. See if the peaks can be recovered.

    TODO: Test for a not as well resolved spectrum...
    """
    xs = np.arange(0, 1000)
    mu = 100
    sigma = 1.5
    A = 100

    A2 = 200
    mu2 = 300
    sigma2 = 10

    sipm_gaussian = gaussian(xs, A, mu, sigma)
    sipm_gaussian += gaussian(xs, A2, mu2, sigma2)
    peaks, peak_locs, amps = guess_peaks(
        sipm_gaussian, xs, min_height=50, min_dist=10, min_width=1
    )

    assert np.array_equal(peaks, [mu, mu2])
    assert np.array_equal(peak_locs, [mu, mu2])
    assert np.array_equal(amps, [A, A2])


def test_guess_peaks_no_width():
    """
    Create a spectrum with two gaussian peaks, well separated at mu1 and mu2. See if the peaks can be recovered

    TODO: Test for a not as well resolved spectrum...
    """
    xs = np.arange(0, 1000)
    mu = 100
    sigma = 1.5
    A = 100

    A2 = 200
    mu2 = 300
    sigma2 = 10

    sipm_gaussian = gaussian(xs, A, mu, sigma)
    sipm_gaussian += gaussian(xs, A2, mu2, sigma2)
    peaks, peak_locs, amps = guess_peaks_no_width(
        sipm_gaussian, xs, min_height=50, min_dist=10
    )

    assert np.array_equal(peaks, [mu, mu2])
    assert np.array_equal(peak_locs, [mu, mu2])
    assert np.array_equal(amps, [A, A2])


def test_fit_peaks():
    """
    Create a spectrum with two gaussian peaks, well separated at mu1 and mu2.
    See if we can recover the Gaussian parameters from fitting the peaks
    """
    xs = np.arange(0, 1000)
    mu = 100
    sigma = 1.5
    A = 100

    A2 = 200
    mu2 = 300
    sigma2 = 10

    sipm_gaussian = gaussian(xs, A, mu, sigma)
    sipm_gaussian += gaussian(xs, A2, mu2, sigma2)

    gauss_params, gauss_errs = fit_peaks(
        sipm_gaussian, xs, [mu, mu2], [mu, mu2], [A, A2], fit_width=30
    )

    assert pytest.approx(gauss_params[0][0], 1e-1) == A
    assert pytest.approx(gauss_params[0][1], 1e-1) == mu
    assert pytest.approx(gauss_params[0][2], 1e-1) == sigma

    assert pytest.approx(gauss_params[1][0], 1e-1) == A2
    assert pytest.approx(gauss_params[1][1], 1e-1) == mu2
    assert pytest.approx(gauss_params[1][2], 1e-1) == sigma2

    assert pytest.approx(gauss_errs[0][0], 1e-1) == 0
    assert pytest.approx(gauss_errs[0][1], 1e-1) == 0
    assert pytest.approx(gauss_errs[0][2], 1e-1) == 0

    assert pytest.approx(gauss_errs[1][0], 1e-1) == 0
    assert pytest.approx(gauss_errs[1][1], 1e-1) == 0
    assert pytest.approx(gauss_errs[1][2], 1e-1) == 0


def test_fit_peak():
    """
    Create a spectrum with two gaussian peaks, well separated at mu1 and mu2.
    See if we can recover the Gaussian parameters from fitting the peaks
    """
    xs = np.arange(0, 1000)
    mu = 100
    sigma = 1.5
    A = 100

    A2 = 200
    mu2 = 300
    sigma2 = 10

    sipm_gaussian = gaussian(xs, A, mu, sigma)
    sipm_gaussian += gaussian(xs, A2, mu2, sigma2)

    gauss_params, gauss_errs = fit_peak(
        sipm_gaussian, xs, [mu2], [mu2], [A2], fit_width=30
    )

    assert pytest.approx(gauss_params[0][0], 1e-1) == A2
    assert pytest.approx(gauss_params[0][1], 1e-1) == mu2
    assert pytest.approx(gauss_params[0][2], 1e-1) == sigma2

    assert pytest.approx(gauss_errs[0][0], 1e-1) == 0
    assert pytest.approx(gauss_errs[0][1], 1e-1) == 0
    assert pytest.approx(gauss_errs[0][2], 1e-1) == 0
