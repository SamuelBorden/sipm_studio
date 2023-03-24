"""
Processors for simulating a SiPM under dark conditions
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, binom, uniform, expon
from scipy.stats import linregress
from scipy.stats import rv_continuous
from scipy.special import gamma
from scipy.stats import rv_continuous, rv_discrete
from scipy.special import gamma, gammaln


class borel_gen(rv_discrete):
    """
    Create a class with methods that compute a Borel distribution
    """

    def _pmf(self, n: int, mu: float) -> float:
        r"""
        Computes the pmf for a Borel distribution $P(n, \mu) = \frac{e^{-\mu n}(\mu n)^{n-1}}{n!}$
        """

        return ((mu * n) ** (n - 1)) / (gamma(n + 1) * np.exp(mu * n))

    def _logpmf(self, n: int, mu: float) -> float:
        r"""
        Computes the log of pmf for a Borel distribution $P(n, \mu) = -\mu n + (n-1)\log(\mu n)-\log(n!)$
        """
        return -mu * n + (n - 1) * np.log(mu * n) - gammaln(n + 1)


# A borel distribution only has support on domain [1, np.inf)
borel = borel_gen(a=1, name="borel")


class sipm:
    def __init__(
        self,
        gain: float,
        DCR: float,
        P_xt: float,
        avg_xt_num: float,
        P_ap: float,
        cell_recovery: float,
        alpha: float,
        PDE: float,
    ):
        """
        A class that contains all the parameters to describe a specific SiPM device operating at a specific reverse bias

        Parameters
        ----------
        gain
            The gain of a SiPM, or the number of electrons produced per single Geiger discharge
        DCR
            The dark count rate, in counts per sample
        P_xt
            The probability for a cross-talk event.
        avg_xt_num
            The average number of cross-talk induced Geiger discharges per primary Geiger discharge
        P_ap
            The probability for afterpulsing
        cell_recovery
            The time for a single cell to recover from a Geiger discharge, in samples
        alpha
            Controls the number of afterpulses per Geiger discharge
        PDE
            The photon detection efficiency
        """
        self.gain = gain
        self.DCR = DCR
        self.P_xt = P_xt
        self.avg_xt_num = avg_xt_num
        self.P_ap = P_ap
        self.cell_recovery = cell_recovery
        self.alpha = alpha
        self.PDE = PDE

    def get_params(self):
        """
        Return a given instance's parameters as a tuple
        """
        return (
            self.gain,
            self.DCR,
            self.P_xt,
            self.avg_xt_num,
            self.P_ap,
            self.cell_recovery,
            self.alpha,
            self.PDE,
        )


class synthetic_waveforms:
    def __init__(
        self,
        num_wfs: int,
        wf_len: int,
        wf_dts: np.array[np.array, ...],
        wf_t0s: np.array,
        wf_num_xts: np.array[np.array, ...],
        wf_xt_t0s: np.array,
        wf_xt_dts: np.array[np.array, ...],
        wf_num_aps: np.array[np.array, ...],
        wf_num_xts_per_ap: np.array[np.array, ...],
        wf_ap_t0s: np.array[np.array, ...],
        wf_ap_dts: np.array,
        dcr_amps: np.array[np.array, ...],
        xt_amps: np.array[np.array, ...],
        ap_amps: np.array[np.array, ...],
    ):
        """
        All the information needed to completely create a set of synthetic waveforms, with timing information generated by :func:`simulate_t0s`

        Parameters
        ----------
        num_wfs
            The number of waveforms to simulate
        wf_len
            The length of a waveform
        wf_dts
            The delay times of all dark count events, stored flat for easy analysis
        wf_t0s
            Array of arrays, each array contains the indices of dark counts within a waveform
        wf_num_xts
            Array of arrays, each array contains the numbers of a Geiger discharges per xt event
        wf_xt_t0s
            Array of arrays,  each array contains the indices of xt events within a waveform
        wf_xt_dts
            The delay times of all xt events, stored flat for easy analysis
        wf_num_aps
            Array of arrays, each array contains the numbers of afterpulses per primary Geiger discharge
        wf_num_xts_per_ap
            Array of arrays, each array contains the numbers of Geiger discharges prior to an afterpulse train
        wf_ap_t0s
            Array of arrays,  each array contains the indices of ap events within a waveform
        wf_ap_dts
            The delay times of all ap events, stored flat for easy analysis
        dcr_amps
            Array of arrays, each array contains the amplitude of each dark count event in a waveform
        xt_amps
            Array of arrays, each array contains the amplitude of each xt event in a waveform
        ap_amps
            Array of arrays, each array contains the amplitude of each ap event in a waveform
        """
        self.num_wfs = num_wfs
        self.wf_len = wf_len
        self.wf_dts = wf_dts
        self.wf_t0s = wf_t0s
        self.wf_num_xts = wf_num_xts
        self.wf_xt_t0s = wf_xt_t0s
        self.wf_xt_dts = wf_xt_dts
        self.wf_num_aps = wf_num_aps
        self.wf_num_xts_per_ap = wf_num_xts_per_ap
        self.wf_ap_t0s = wf_ap_t0s
        self.wf_ap_dts = wf_ap_dts
        self.dcr_amps = dcr_amps
        self.xt_amps = xt_amps
        self.ap_amps = ap_amps

    def full_amplitudes(self) -> np.array:
        """
        Create the a flat arrays that has every single pulse amplitude in it
        """
        actual_amps = np.concatenate(np.array(self.dcr_amps, dtype="object")).flatten()
        flat_xt_amps = np.concatenate(np.array(self.xt_amps, dtype="object")).flatten()
        flat_ap_amps = np.concatenate(np.array(self.ap_amps, dtype="object")).flatten()
        actual_amps = np.append(actual_amps, flat_xt_amps)
        actual_amps = np.append(actual_amps, flat_ap_amps)
        return actual_amps

    def calculate_xt(self) -> float:
        """
        Calculate the cross talk probability of a synthetic dataset by using $P_{xt} = \frac{\# XT}{\# XT + \# DCR}$
        """
        return len(self.wf_xt_dts) / (len(self.wf_dts) + len(self.wf_xt_dts))

    def calculate_ap(self) -> float:
        """
        Calculate the afterpulsing probability of a synthetic dataset by using $P_{ap} = \frac{\# AP}{\# XT + \# DCR}$
        """
        return len(self.wf_ap_dts) / (len(self.wf_dts) + len(self.wf_xt_dts))

    def create_dirac_wfs(self) -> np.array[np.array, ...]:
        """
        Create an array of arrays, where each array is a waveform with a dirac spike for each simulated event.
        Each waveform has dark counts, cross talks, and afterpulses with amplitudes that were generated during the simulation.
        """
        wfs = []
        ts = np.arange(0, self.wf_len)

        for k in range(self.num_wfs):
            # Make one synthetic waveform, just put in dirac spikes
            wf = np.zeros(len(ts))

            # Put in the dark counts
            for j, i in enumerate(self.wf_t0s[k]):
                wf[int(i)] = self.dcr_amps[k][j]

            # put in the cross talks
            for j, i in enumerate(self.wf_xt_t0s[k]):
                wf[int(i)] = self.xt_amps[k][j]

            # put in the afterpulses
            for j, i in enumerate(self.wf_ap_t0s[k]):
                wf[int(i)] = self.ap_amps[k][j]

            wfs.append(wf)

        return wfs


def ap_recovery(dts: np.array, recovery_time: float) -> np.array:
    """
    Returns the height of an afterpulse based on how far along a cell's recovery time the after-pulse occurs,
    following a simple exponential model for the recovery time.

    Parameters
    ----------
    dts
        An array containing the time since a Geiger discharge that the afterpulse occurs
    recovery_time
        The recovery time of a single cell after a Geiger discharge

    Notes
    -----
    recovery_times and dts must share the same units!
    """

    return 1 - np.exp(-1 * np.array(dts) / recovery_time)


def simulate_t0s(
    num_wfs: int,
    wf_len: int,
    sipm: type[sipm],
    sigma_detector: float,
    sampling_rate: float = 1,
) -> type[synthetic_waveforms]:
    """
    A function that simulates a SiPM under dark conditions.

    Parameters
    ----------
    num_wfs
        The number of waveforms to simulate
    wf_len
        The length of a waveform
    sipm
        A :class:`sipm` that contains the parameters necessary to simulate, such as DCR
    sigma_detector
        The energy resolution of a single p.e. peak due to electronics noise
    sampling_rate
        The sampling rate of the time axis

    Returns
    -------
    waveform_class
        An instance of :class:`synthetic_waveforms` that contains all of the waveform information

    Notes
    -----
    The function first performs a loop over the number of waveforms. For each waveform, a delay time is generated from an exponential distribution.
    A running counter adds the dt to the current position inside a waveform, and the following process repeats until we run outside of the waveform.

    1. Sample a uniform distribution to see if we exceed P_xt -- if we do, generate a number of xt Geiger discharges from a Borel distribution.
    Save the xt's t0, dt, and number of Geiger discharges it corresponds to.
    2. If we don't get a xt event, then the event is just a normal dark count. Save its t0 and dt
    3. Regardless if we get an xt or a dc, sample a uniform distribution to see if we exceed P_ap -- if we do, sample a binomial distribution to
    see how many afterpulses we get. For each afterpulse, generate a dt from the parent Geiger discharge(s) according to an exponential distribution
    with the cell_recovery time. Also, multiply each afterpulse by the number of parent Geiger discharge(s) it spwans from. The amplitude of each ap
    is also determined during this step, based on an amplitude recovery determined by an exponential cell recovery time.

    This cycle continues until we run out of waveform. The parameters are appended to an array, and then we start generating the next waveform.
    At the end, each amplitude is convolved with a Gaussian distribution corresponding to the detector's electronics response.
    """
    # Extract the parameters from the sipm class
    gain, DCR, P_xt, avg_xt_num, P_ap, cell_recovery, alpha, PDE = sipm.get_params()

    mu = 1 - (1 / avg_xt_num)  # shape parameter of the Borel distribution for XT

    wf_t0s = []  # an array of arrays, all of the positions of the t0s
    wf_dts = []  # a flat array, all of the delay times between dark count events

    wf_xt_t0s = []  # an array of arrays, of all of the positions of cross talk events
    wf_xt_dts = []  # a flat array, delay times to xt events
    wf_num_xts = (
        []
    )  # an array of arrays, containing the number of cross talk discharges

    wf_ap_t0s = []  # an array of arrays, of all positions of afterpulsing events
    wf_ap_dts = (
        []
    )  # a flat array, of all of the afterpulse delay times, can check against it later
    wf_num_aps = (
        []
    )  # an array of arrays, containing the number of afterpulses from k-number of XT events, based on a binomial
    wf_num_xts_per_ap = (
        []
    )  # an array of arrays, if an afterpulse follows a cross talk event

    dcr_amps = []  # amplitudes of dark events
    xt_amps = []
    ap_amps = []

    for i in range(num_wfs):
        t0s = []  # an array of all of the positions of the t0s
        dts = []  # an array of all of the delay times between dark count events

        xt_t0s = []  # an array of all of the positions of cross talk events
        xt_dts = []
        num_xts = []  # an array containing the number of cross talk discharges

        ap_t0s = []  # an array of all positions of afterpulsing events
        ap_dts = (
            []
        )  # an array of all of the afterpulse delay times, can check against it later
        num_aps = (
            []
        )  # an array containing the number of afterpulses from k-number of XT events, based on a binomial
        num_xts_per_ap = []

        # Keep sampling until we are outside the waveform
        position = 0
        while position < wf_len:
            dt0 = expon.rvs(0, DCR, size=1)[
                0
            ]  # the exponential distribution is defined by loc, scale like anything in scipy.stats
            # Advance the position by this dt
            position += dt0

            if position > wf_len:
                break

            if position in ap_t0s:
                pass

            else:
                # Now see if we get any XTs or APs
                num = 1  # set default to the 1 DC already created
                if P_xt >= uniform.rvs(size=1)[0]:
                    # The number of XT events is determined by a Borel distribution
                    num += borel.rvs(mu, size=1)[0]
                    num_xts.append(num)
                    xt_t0s.append(position)
                    xt_dts.append(dt0)

                else:
                    # If we're still inside the wf_len, then save it and the delay time
                    # Save the dark counts
                    dts.append(dt0)
                    t0s.append(position)

                num_ap = 0
                if P_ap >= uniform.rvs(size=1)[0]:
                    # The number of AP events from the k-number of XT events above is determined by a Binomial distribution
                    while num_ap == 0:  # make sure we don't get 0 afterpulses
                        num_ap = binom.rvs(num, alpha, size=1)[0] + 1
                    num_aps.append(num_ap)
                    for i in range(num_ap):
                        dt_ap = expon.rvs(0, AP_rate, size=1)[0]
                        if position + dt_ap <= wf_len:
                            num_xts_per_ap.append(num)
                            ap_t0s.append(position + dt_ap)
                            ap_dts.append(dt_ap)

        # Create amplitudes for the dcrs and xts
        dcr_amps.append(norm.rvs(mu_dcr, sigma_detector, size=len(t0s)))
        xt_amps.append(
            np.array(num_xts) * norm.rvs(mu_dcr, sigma_detector, size=len(xt_t0s))
        )

        # Create amplitude for the APs
        lap_amps = ap_recovery(ap_dts, AP_rate)
        lap_amps = np.array(num_xts_per_ap) * lap_amps
        ap_amps.append(lap_amps * norm.rvs(mu_dcr, sigma_detector, size=len(ap_t0s)))

        # Save all of the info, each list in the list of lists is one waveform's info
        wf_dts.extend(dts)
        wf_t0s.append(t0s)
        wf_num_xts.append(num_xts)
        wf_xt_t0s.append(xt_t0s)
        wf_xt_dts.extend(xt_dts)
        wf_num_aps.append(num_aps)
        wf_num_xts_per_ap.append(num_xts_per_ap)
        wf_ap_t0s.append(ap_t0s)
        wf_ap_dts.append(ap_dts)

    waveform_class = synthetic_waveforms(
        num_wfs,
        wf_len,
        wf_dts,
        wf_t0s,
        wf_num_xts,
        wf_xt_t0s,
        wf_xt_dts,
        wf_num_aps,
        wf_num_xts_per_ap,
        wf_ap_t0s,
        wf_ap_dts,
        dcr_amps,
        xt_amps,
        ap_amps,
    )

    return waveform_class
