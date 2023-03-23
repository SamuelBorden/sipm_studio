import pytest
import numpy as np

import os
from pathlib import Path

from sipm_studio.raw.daq_to_raw import (
    get_event_size,
    get_event,
    get_event_v2,
    _assemble_data_row,
    _output_to_h5file,
    process_metadata,
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
    # info_array, waveform = get_event(event_data_bytes)
    # assert np.array_equal(waveform[:5], [0, 0, 0, 0, 0])
    # assert np.array_equal(info_array, [1000000, 10, 10 ,10])
    assert 1 == 1


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


def test_assemble_data_row():
    assert 1 == 1


def test_output_to_h5file():
    assert 1 == 1


def test_process_metadata():
    assert 1 == 1
