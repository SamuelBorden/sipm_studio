import pytest
import numpy as np
import h5py

import os
from pathlib import Path

from sipm_studio.raw.daq_to_raw import (
    get_event_size,
    get_event,
    get_event_v2,
    _output_to_h5file,
    build_raw,
)

test_file = Path(__file__).parent / "compass_test_data.BIN"


def test_get_event_size():
    event_size, flag = get_event_size(test_file)
    assert flag == True
    assert event_size == 2025  # packet of 25 bytes plus 1000 sample length waveform


def test_get_event():
    # need to make a compass v1 file...
    event_size, flag = get_event_size(test_file)
    with open(test_file, "rb") as metadata_file:
        file_header = metadata_file.read(
            2
        )  # read in the header present in v2 Compass...
        event_data_bytes = metadata_file.read(event_size)
    event_data_bytes = event_data_bytes[:20] + event_data_bytes[21:]
    info_array, waveform = get_event(event_data_bytes)
    assert np.array_equal(
        waveform[:10], [2745, 2742, 2745, 2746, 2745, 2743, 2745, 2744, 2746, 2747]
    )
    assert np.array_equal(info_array, [97876200000, 798, 135, 16384])


def test_get_event_v2():
    event_size, flag = get_event_size(test_file)
    with open(test_file, "rb") as metadata_file:
        file_header = metadata_file.read(
            2
        )  # read in the header present in v2 Compass...
        event_data_bytes = metadata_file.read(event_size)
    info_array, waveform = get_event_v2(event_data_bytes)
    assert np.array_equal(
        waveform[:10], [2745, 2742, 2745, 2746, 2745, 2743, 2745, 2744, 2746, 2747]
    )
    assert np.array_equal(info_array, [97876200000, 798, 135, 16384])


def test_output_to_h5file(tmp_path):

    d = tmp_path / "daq_to_raw_test"
    d.mkdir()

    # make the input files so that we can assure everything works as intended

    f = d / "t1_output_test.h5"
    f.touch()

    file_name = "output_test"

    board = 0
    channel = 2
    timestamp = 130405
    energy = 10
    energy_short = 1
    flags = 111
    num_samples = 2025
    waveform = np.array([1, 2, 3, 4])
    baselines = np.array([0, 0, 0, 0])

    events = np.array([timestamp, energy, energy_short, flags])
    out_waveforms = waveform

    _output_to_h5file(file_name, d, events, out_waveforms, baselines)

    with h5py.File(f, "r") as output_file:
        times = output_file["/raw/timetag"][()]
        energies = output_file["/raw/energy"][()]
        waveforms = output_file["/raw/waveforms"][:]
        bls = output_file["/raw/baselines"][:]
        adc = output_file["adc_to_v"][()]

    assert adc == 2 / 2**14
    assert times == timestamp
    assert energies == energy
    assert np.array_equal(waveforms, waveform)
    assert np.array_equal(bls, baselines)


def test_build_raw(tmp_path):
    d = tmp_path / "daq_to_raw_test"
    d.mkdir()

    # make the input files so that we can assure everything works as intended

    f = d / "t1_compass_test_data.BIN.h5"
    f.touch()

    build_raw(test_file, d)

    with h5py.File(f, "r") as output_file:
        times = output_file["/raw/timetag"][:1]
        energies = output_file["/raw/energy"][:1]
        waveforms = output_file["/raw/waveforms"][:]
        bls = output_file["/raw/baselines"][:]
        adc = output_file["adc_to_v"][()]

    assert adc == 2 / 2**14
    assert times[0] == 97876200000
    assert energies[0] == 798
    assert np.allclose(waveforms[0][:5], np.array([2745, 2742, 2745, 2746, 2745]), 1e-6)
    assert np.allclose(
        bls[0][:5],
        np.array(
            [2736.86041856, 2737.81281246, 2738.75816431, 2739.69648967, 2740.62780407]
        ),
        1e-6,
    )
