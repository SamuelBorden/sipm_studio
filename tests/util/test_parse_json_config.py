import pytest
from pathlib import Path
import os
import h5py
import copy
import json

from sipm_studio.util.parse_json_config import (
    parse_gain_json,
    parse_light_json,
    parse_raw_json_config,
    parse_pde_json,
)


gain_config = {
    "input_path": "/tmp/",
    "output_path": "/tmp/",
    "output_file_name": "test_gain_output.h5",
    "input_files": [
        {
            "file": "t1_Data_CH0@DT5730_1463_ketek_dark_rt_03-10-2023_apd_150V_sipm_545dv.h5",
            "bias": 54.5,
            "device_name": "sipm_1st",
            "vpp": 0.5,
            "start_idx": 50,
            "end_idx": 250,
        },
        {
            "file": "t1_Data_CH0@DT5730_1463_ketek_dark_rt_03-10-2023_apd_150V_sipm_550dv.h5",
            "bias": 55,
            "device_name": "sipm_1st",
            "vpp": 0.5,
            "start_idx": 50,
            "end_idx": 250,
        },
    ],
}


def test_parse_gain_json(tmp_path):

    d = tmp_path / "gain_parse_test"
    d.mkdir()

    # make the input files so that we can assure everything works as intended

    f = d / "t1_Data_CH0@DT5730_1463_ketek_dark_rt_03-10-2023_apd_150V_sipm_545dv.h5"
    f.touch()

    f2 = d / "t1_Data_CH0@DT5730_1463_ketek_dark_rt_03-10-2023_apd_150V_sipm_550dv.h5"
    f2.touch()

    good_config = copy.copy(gain_config)
    good_config["input_path"] = str(d)
    good_config["output_path"] = str(d)

    json_to_write = json.dumps(good_config)
    with open(f"{str(d)}/gain_specs.json", "w") as outfile:
        outfile.write(json_to_write)

    out_args = parse_gain_json(rf"{d}/gain_specs.json")

    assert out_args[0] == (
        f"{d}/t1_Data_CH0@DT5730_1463_ketek_dark_rt_03-10-2023_apd_150V_sipm_545dv.h5",
        54.5,
        "sipm_1st",
        0.5,
        50,
        250,
        f"{d}/test_gain_output.h5",
    )
    assert out_args[1] == (
        f"{d}/t1_Data_CH0@DT5730_1463_ketek_dark_rt_03-10-2023_apd_150V_sipm_550dv.h5",
        55,
        "sipm_1st",
        0.5,
        50,
        250,
        f"{d}/test_gain_output.h5",
    )

    # touch the output file so that it already exists
    f3 = d / "test_gain_output.h5"
    f3.touch()
    with pytest.raises(ValueError) as exc_info:
        out_args = parse_gain_json(rf"{d}/gain_specs.json")

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "Output file already exists"


light_config = {
    "input_path": "/tmp/",
    "output_path": "/tmp/",
    "output_file_name": "test_light_output.h5",
    "input_files": [
        {
            "file": "t1_Data_CH0@DT5730_1463_ketek_dark_rt_03-10-2023_apd_150V_sipm_545dv.h5",
            "bias": 54.5,
            "device_name": "sipm_1st",
            "vpp": 2,
            "gain_file": "gain_file.h5",
            "light_window_start_idx": 50,
            "light_window_end_idx": 250,
            "dark_window_start_idx": 4000,
            "dark_window_end_idx": 4200,
        },
        {
            "file": "t1_Data_CH1@DT5730_1463_ketek_dark_rt_03-10-2023_apd_150V_sipm_545dv.h5",
            "bias": 54.5,
            "device_name": "apd",
            "vpp": 0.5,
            "gain_file": "gain_file.h5",
            "light_window_start_idx": 10,
            "light_window_end_idx": 20,
            "dark_window_start_idx": 4100,
            "dark_window_end_idx": 4400,
        },
    ],
}


def test_parse_light_json(tmp_path):
    d = tmp_path / "light_parse_test"
    d.mkdir()

    # make the input files so that we can assure everything works as intended

    f = d / "t1_Data_CH0@DT5730_1463_ketek_dark_rt_03-10-2023_apd_150V_sipm_545dv.h5"
    f.touch()

    f2 = d / "t1_Data_CH1@DT5730_1463_ketek_dark_rt_03-10-2023_apd_150V_sipm_545dv.h5"
    f2.touch()

    gain = d / "gain_file.h5"
    gain.touch()

    good_config = copy.copy(light_config)
    good_config["input_path"] = str(d)
    good_config["output_path"] = str(d)
    good_config["input_files"][0]["gain_file"] = str(gain)
    good_config["input_files"][1]["gain_file"] = str(gain)

    json_to_write = json.dumps(good_config)
    with open(f"{str(d)}/light_specs.json", "w") as outfile:
        outfile.write(json_to_write)

    out_args = parse_light_json(rf"{d}/light_specs.json")

    assert out_args[0] == (
        f"{d}/t1_Data_CH0@DT5730_1463_ketek_dark_rt_03-10-2023_apd_150V_sipm_545dv.h5",
        54.5,
        "sipm_1st",
        2,
        f"{d}/gain_file.h5",
        50,
        250,
        4000,
        4200,
        f"{d}/test_light_output.h5",
    )
    assert out_args[1] == (
        f"{d}/t1_Data_CH1@DT5730_1463_ketek_dark_rt_03-10-2023_apd_150V_sipm_545dv.h5",
        54.5,
        "apd",
        0.5,
        f"{d}/gain_file.h5",
        10,
        20,
        4100,
        4400,
        f"{d}/test_light_output.h5",
    )

    # touch the output file so that it already exists
    f3 = d / "test_light_output.h5"
    f3.touch()
    with pytest.raises(ValueError) as exc_info:
        out_args = parse_gain_json(rf"{d}/light_specs.json")

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "Output file already exists"


def test_parse_raw_json_config(tmp_path):
    d = tmp_path / "raw_parse_test"
    d.mkdir()

    # make the input files so that we can assure everything works as intended
    f = d / "t1_Data_CH0@DT5730_1463_ketek_dark_rt_03-10-2023_apd_150V_sipm_545dv.BIN"
    f.touch()

    config_dict = dict({"input_path": f"{d}", "output_path": f"{d}"})

    json_to_write = json.dumps(config_dict)
    with open(f"{str(d)}/process_raw.json", "w") as outfile:
        outfile.write(json_to_write)

    out_files = parse_raw_json_config(f"{str(d)}/process_raw.json")

    assert out_files[0] == (f"{f}", f"{d}")


def test_parse_pde_json_config(tmp_path):
    d = tmp_path / "pde_parse_test"
    d.mkdir()

    # make the input files so that we can assure everything works as intended
    f = d / "t1_Data_CH0@DT5730_1463_ketek_dark_rt_03-10-2023_apd_150V_sipm_545dv.h5"
    f.touch()

    output_name = d / "pde_test.h5"

    config_dict = dict(
        {
            "input_path": f"{d}",
            "output_path": f"{d}",
            "temperature": "LN",
            "output_file_name": "pde_test.h5",
            "led_wavelength": 560,
            "input_files": [
                {
                    "file": "t1_Data_CH0@DT5730_1463_ketek_dark_rt_03-10-2023_apd_150V_sipm_545dv.h5",
                    "bias": 54.5,
                    "device_name": "sipm",
                    "vpp": 0.5,
                    "light_window_start_idx": 10,
                    "light_window_end_idx": 100,
                    "dark_window_start_idx": 11,
                    "dark_window_end_idx": 101,
                }
            ],
        }
    )

    json_to_write = json.dumps(config_dict)
    with open(f"{str(d)}/process_raw.json", "w") as outfile:
        outfile.write(json_to_write)

    out_array = parse_pde_json(f"{str(d)}/process_raw.json")[0]

    assert out_array[0] == f"{f}"
    assert out_array[1] == "LN"
    assert out_array[2] == 0.2457870367786596
    assert out_array[3] == 54.5
    assert out_array[4] == "sipm"
    assert out_array[5] == 0.5
    assert out_array[6] == 10
    assert out_array[7] == 100
    assert out_array[8] == 11
    assert out_array[9] == 101
    assert out_array[10] == f"{output_name}"
