import pytest
from pathlib import Path
import numpy as np
import h5py
import matplotlib.pyplot as plt

from sipm_studio.gain.gain import calculate_gain
from sipm_studio.simulations.simulate_sipm_response import synthetic_waveforms
from sipm_studio.dsp.adc_to_current import current_converter
from sipm_studio.dsp.current_to_charge import integrate_current

test_file = Path(__file__).parent / "configs/gain_wfs.h5"

e_charge = 1.6e-19
percent_error_tolerance = 5


@pytest.mark.filterwarnings(
    "ignore::"
)  # have to ignore a linalg warning from the signal.impulse when creating the syntehtic wf
def test_calculate_gain(tmp_path):
    # Read in the model pulse, and find its gain, so we can compare against it
    synth_wf = synthetic_waveforms(
        2,
        1000,
        [[]],
        [[50], []],
        [[], []],
        [[], []],
        [[], []],
        [[], []],
        [[], []],
        [[], []],
        [[], []],
        [[1], [1]],
        [[], []],
        [[], []],
    )
    synth_wf = synth_wf.create_quick_duara_wfs()
    synth_wf = np.array(synth_wf) * (
        2**14 / 2
    )  # duara wfs are in volts, convert back to ADC so we can use our processors
    print(synth_wf)

    current_synth_wf = current_converter(synth_wf, 2, 14, "sipm")
    real_gain = integrate_current(current_synth_wf, 0, 1000, 2e-9)[0] / e_charge

    assert real_gain == pytest.approx(3930703.710583516, rel=1e-3)

    d = tmp_path / "gain_test"
    d.mkdir()

    # make the output files so that we can assure everything works as intended

    f = d / "gain_test.h5"
    f.touch()

    bias = 54.5
    calculate_gain(test_file, bias, "sipm", 2, 40, 250, str(f))

    f_read = h5py.File(f, "r")
    calc_gain = f_read[f"{bias}/gain"][:]

    percent_error = np.abs(calc_gain[0] - real_gain) / real_gain * 100

    assert percent_error <= percent_error_tolerance
