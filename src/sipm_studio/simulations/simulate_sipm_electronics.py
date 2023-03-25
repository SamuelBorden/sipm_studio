"""
Processors for simulating a SiPM's electronics response; namely, simulate its transfer function.
"""
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# The saved waveform is measured in volts
# the saved sipm parameters are for a fit from a Hamamatsu 3050CS biased at 54.5V

Rs = 50  # 50 ohm shunt resistance to connect to Oscope or amp
N = 3600  # number of microcells


def sipm_duara_model(
    Rq: float, Rd: float, Cd: float, Cf: float, Cq: float, Cg: float, dV: float
) -> type[signal.TransferFunction]:
    """
    Create a transfer function based on the model proposed in Duara 2020.

    Parameters
    ----------
    Rq
        Quenching resistor, in ohms
    Cd
        Junction/diode capacitance
    Cf
        Summed diode and parasitic capacitances
    Cq
        Parasitic capacitance
    Cg
        Ground capacitance
    dV
        Maximum height of the pulse
    """
    b1 = Rq * (Cd + 2 * Cq)
    b2 = Rq**2 * Cq * (Cd + Cq)
    a1 = 2 * Rq * (Cd + Cq) + Rs * (Cg + N * Cd)
    a2 = Rq * (
        N * Cd * Rs * (Cd + 2 * Cq) + Rq * (Cd + Cq) ** 2 + 2 * Cg * Rs * (Cd + Cq)
    )
    a3 = Rq**2 * Rs * (Cd + Cq) * (N * Cd * Cq + Cg * (Cd + Cq))
    alpha = N * dV * Cf * Rs

    num = [alpha * b2, alpha * b1, alpha]
    den = [Rd * Cd * a3, Rd * Cd * a2 + a3, Rd * Cd * a1 + a2, Rd * Cd + a1, 1]

    return signal.TransferFunction(num, den)


def model_fit(
    t: np.array,
    t0_idx: int,
    Rq: float,
    Rd: float,
    Cd: float,
    Cf: float,
    Cq: float,
    Cg: float,
    dV: float,
) -> np.array:
    """
    Given a set of parameters, and an initial time, create a modeled pulse and shift it to t0. The length of the waveform is the length of t.

    Parameters
    ----------
    t
        Time array
    t0_idx
        index in time array to start the pulse
    Rq
        Quenching resistor, in ohms
    Cd
        Junction/diode capacitance
    Cf
        Summed diode and parasitic capacitances
    Cq
        Parasitic capacitance
    Cg
        Ground capacitance
    dV
        Maximum height of the pulse
    """
    if np.isnan(t0_idx):
        return np.zeros(len(t))
    fit_model = sipm_duara_model(Rq, Rd, Cd, Cf, Cq, Cg, dV)
    N_samples = len(t[int(t0_idx) :])
    if (N_samples == 0) or (N_samples == 1):
        return np.zeros(len(t))
    t_model, y = signal.impulse(fit_model, N=N_samples)
    out = np.insert(y, 0, np.zeros(len(t) - len(y)))  # insert 0s at the start before t0
    return out
