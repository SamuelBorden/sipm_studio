import pytest
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.stats import norm

from sipm_studio.light.light import calculate_light
from sipm_studio.dsp.adc_to_current import current_converter, compute_sipm_amp
from sipm_studio.dsp.current_to_charge import integrate_current

# test_file = Path(__file__).parent / "configs/gain_wfs.h5"

e_charge = 1.6e-19
percent_error_tolerance = 5
np.random.seed(42)

num_wfs = 5000
rv_sigma = 0.1


def test_calculate_gain(tmp_path):
    # make the output directory and gain file
    d = tmp_path / "light_test"
    d.mkdir()

    gain_file = d / "light_gain.h5"
    gain_file.touch()

    wf_file = d / "light_wfs.h5"
    wf_file.touch()

    output_file = d / "light_test.h5"
    output_file.touch()

    gain = 1.7e6
    gain_err = gain * 0.02
    light_photons = 4000
    dark_photons = 0

    light_charge = light_photons * gain * e_charge
    dark_charge = dark_photons * gain * e_charge

    light_window = 1000 * 2e-9  # in seconds
    dark_window = light_window

    light_amplitude = light_charge / light_window  # in Amps
    dark_amplitude = dark_charge / dark_window

    light_amplitude *= compute_sipm_amp()  # convert from Amps to V
    dark_amplitude *= compute_sipm_amp()

    light_amplitude *= (2**14) / 2  # convert from V to ADC
    dark_amplitude *= (2**14) / 2

    # Now create the waveforms and save them to a file
    wfs = []
    bls = []
    for i in range(int(num_wfs)):
        light_part = norm.rvs(light_amplitude, light_amplitude * rv_sigma, size=1000)
        dark_part = norm.rvs(dark_amplitude, rv_sigma, size=1000)

        # light_part = np.full(1000, norm.rvs(light_amplitude, light_amplitude*rv_sigma, size=1)[0])
        # dark_part = np.full(1000, norm.rvs(dark_amplitude, rv_sigma, size=1)[0])

        wf = np.zeros(1000)
        wf = np.append(wf, light_part)
        wf = np.append(wf, np.zeros(2000))
        wf = np.append(wf, dark_part)

        wfs.append(wf)
        bls.append(np.zeros_like(wf))

    f_out = h5py.File(str(wf_file), "a")
    dset = f_out.create_dataset("raw/waveforms", data=wfs)
    dset = f_out.create_dataset("raw/baselines", data=bls)
    f_out.close()

    f_out = h5py.File(str(gain_file), "a")
    gain_arr = [gain, gain_err]
    dset = f_out.create_dataset("54.5/gain", data=gain_arr)
    f_out.close()

    # just test to make sure we did the light preprocessing correctly
    light_Is = current_converter(np.array(wfs), vpp=2, n_bits=14, device="sipm")
    qs = integrate_current(light_Is, 1000, 2000, 2e-9)

    photons = np.amax(qs) / gain / e_charge

    assert photons == pytest.approx(light_photons, rel=1)

    # do the actual light calculations
    bias = 54.5
    calculate_light(
        wf_file,
        bias,
        "sipm",
        2,
        str(gain_file),
        1000,
        2000,
        4000,
        5000,
        str(output_file),
    )

    f_read = h5py.File(output_file, "r")

    calc_light_photons = f_read[f"sipm/{bias}/n_charge"][()]
    percent_error = np.abs(calc_light_photons[0] - light_photons) / light_photons * 100
    assert percent_error <= percent_error_tolerance

    calc_dark_photons = f_read[f"sipm/dark/{bias}/n_charge"][()]
    percent_error = np.abs(calc_dark_photons[0] - dark_photons) * 100
    assert percent_error <= percent_error_tolerance

    f_read.close()
