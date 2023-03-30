"""
Processors for converting DAQ captured waveforms to current waveforms using known gains and impedances
"""
import numpy as np

# Create a hardcoded dictionary of physical hardware settings to let the user select from
setting_dict = {
    "amplifier": {
        "sipm": {
            "functions": [
                "trans_amp",
                "voltage_divider",
                "non_invert_amp",
                "invert_amp",
                "voltage_divider",
            ],
            "values": [
                {"R1": 1.5e3},
                {"R1": 49.9, "R2": 49.9},
                {"R1": 680, "R2": 75},
                {"R1": 750, "R2": 90.9},
                {"R1": 49.9, "R2": 49.9},
            ],
        },
        "sipm_1st": {
            "functions": ["trans_amp", "voltage_divider"],
            "values": [{"R1": 1.5e3}, {"R1": 49.9, "R2": 49.9}],
        },
        "sipm_1st_low_gain": {
            "functions": ["trans_amp", "voltage_divider"],
            "values": [{"R1": 150}, {"R1": 49.9, "R2": 49.9}],
        },
        "sipm_low_gain": {
            "functions": [
                "trans_amp",
                "voltage_divider",
                "non_invert_amp",
                "invert_amp",
                "voltage_divider",
            ],
            "values": [
                {"R1": 150},
                {"R1": 49.9, "R2": 49.9},
                {"R1": 680, "R2": 75},
                {"R1": 750, "R2": 90.9},
                {"R1": 49.9, "R2": 49.9},
            ],
        },
        "apd": {
            "functions": [
                "trans_amp",
                "voltage_divider",
                "non_invert_amp",
                "invert_amp",
                "voltage_divider",
            ],
            "values": [
                {"R1": 2.49e3},
                {"R1": 49.9, "R2": 49.9},
                {"R1": 499, "R2": 69.8},
                {"R1": 499, "R2": 100},
                {"R1": 49.9, "R2": 49.9},
            ],
        },
    },
    "v_range": {"sipm": 2, "apd": 0.5},
}


class pc:
    """
    Class that defines circuit components and their amplification
    """

    def voltage_divider(R1: float, R2: float) -> float:
        return R2 / (R1 + R2)

    def trans_amp(R1: float) -> float:
        return -R1

    def non_invert_amp(R1: float, R2: float) -> float:
        return 1 + (R1 / R2)

    def invert_amp(R1: float, R2: float) -> float:
        return -R1 / R2


def _compute_amplification(settings: dict, channel: str) -> float:
    """
    Compute the amplification for a specific hardware configuration.
    """
    full_amp = 1
    for i, func in enumerate(settings["amplifier"][channel]["functions"]):
        amp_func = getattr(pc, func)
        full_amp *= amp_func(**settings["amplifier"][channel]["values"][i])
    return full_amp


def compute_sipm_amp() -> float:
    return _compute_amplification(setting_dict, "sipm")


apd_amp = _compute_amplification(setting_dict, "apd")
sipm_amp = _compute_amplification(setting_dict, "sipm")
sipm_amp_1st_stage = _compute_amplification(setting_dict, "sipm_1st")
sipm_amp_1st_stage_low_gain = _compute_amplification(setting_dict, "sipm_1st_low_gain")
sipm_amp_low_gain = _compute_amplification(setting_dict, "sipm_low_gain")


def current_converter(
    waveforms: np.array, vpp: float = 2, n_bits: int = 14, device: str = "sipm"
) -> np.array:
    r"""
    Convert a DAQ ADC waveform to current using the amplification associated with the hardware configuration.

    1. Convert the ADC (least significant bit, lsb) to voltage  $ADC \cdot \frac{v_{pp}}{2^{n_bits}}$
        The CAEN DAQ splits the volage dynamic range (v_pp) into a binary number with n_bits of precision
    2. Convert the voltage to current using the amplification/impedance of the hardware for the device: $V / Amp [A/V]$
    """
    amp = _compute_amplification(setting_dict, device)
    return waveforms * (vpp / 2**n_bits) * (1 / amp)
