import pytest
import numpy as np

from sipm_studio.dsp.adc_to_current import pc, _compute_amplification, current_converter


def test_class_pc():
    R1 = 10
    R2 = 100

    test_voltage_divider = pc.voltage_divider(R1, R2)
    assert test_voltage_divider == 100 / 110

    test_trans_amp = pc.trans_amp(R1)
    assert test_trans_amp == -10

    test_non_invert_amp = pc.non_invert_amp(R1, R2)
    assert test_non_invert_amp == 1.1

    test_invert_amp = pc.invert_amp(R1, R2)
    assert test_invert_amp == -0.1


def test_compute_amplification():
    setting_dict = {
        "amplifier": {
            "test": {
                "functions": ["trans_amp", "voltage_divider"],
                "values": [{"R1": 1000}, {"R1": 50, "R2": 50}],
            },
        }
    }
    # The test creates a tranimpedance amplifier with gain of -1000, and then a voltage divider that divides the gain exactly in half
    amp = _compute_amplification(setting_dict, "test")
    assert amp == -500


def test_current_converter():
    test_waveforms = np.ones((10, 100))

    # We set the bit depth to 14, and the voltage range to 2vpp, so we first convert 1*2/(2^14)
    # The setting dict for sipm_1st has an amplification of -750
    # so we should expect arrays of 2/(2^14 * -750)
    currents = current_converter(test_waveforms, vpp=2, n_bits=14, device="sipm_1st")
    expected_currents = np.full_like(currents, -2 / (np.power(2, 14) * 750))

    assert np.array_equal(currents, expected_currents)
