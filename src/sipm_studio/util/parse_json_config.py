"""
Define functions that parse a json config file for various measurement configurations.
As an input takes the json file and the measurement level specified by the user and returns parameters needed
for those specific measurements
"""
import os, json, h5py
import numpy as np
from sipm_studio.util.parse_compass_filename import parse_compass_file_name
import copy


def parse_gain_json(json_file_name: str) -> list[list, ...]:
    """
        Parse a config file defined specifically for the gain measurement script

        Parameters
        ----------
        json_file_name
            Path to a json file containing parameters used for performing gain analysis

        Returns
        -------
        out_args
            List of lists. Each list looks like ["path/to/input/input", bias, device_name, vpp, start_idx, end_idx, "path/to/output/output.h5"]


        Example Config
        --------------
        {
        "input_path": "/path/to/raw/files",
        "output_path": "/path/to/analyzed/data",
        "output_file_name": "output_name.h5",
        "input_files": [
        {"file": "t1_Data_Channel@DT5730_1463_devicename_dark/light/_rt/ln_MONTH-DAY-YEAR_apd_apdbiasinvolts_sipm_sipmbiasindecivolt_(optional LED info)
    ", "bias": 54, "device_name": "sipm_1st", "vpp": 0.5, "start_idx": 50, "end_idx": 250 },
        {"file": "t1_Data_Channel@DT5730_1463_devicename_dark/light/_rt/ln_MONTH-DAY-YEAR_apd_apdbiasinvolts_sipm_sipmbiasindecivolt_(optional LED info)
    ", "bias": 54, "device_name": "sipm_1st", "vpp": 0.5, "start_idx": 50, "end_idx": 250 }
        ],
        }
    """
    f = open(json_file_name)
    json_file = json.load(f)
    f.close()

    # make the output directory if we need to
    if not os.path.exists(json_file["output_path"]):
        os.mkdir(json_file["output_path"])

    # make sure we don't override an existing analysis
    if os.path.exists(
        os.path.join(json_file["output_path"], json_file["output_file_name"])
    ):
        raise ValueError("Output file already exists")

    # go through each file, join the paths, check that the names are valid, and then put in a list that can
    # be passed to multiprocessing.pool

    out_args = []
    dict_list = copy.copy(json_file["input_files"])
    for dictionary in dict_list:
        input_file = os.path.join(json_file["input_path"], dictionary["file"])
        output_file = os.path.join(
            json_file["output_path"], json_file["output_file_name"]
        )

        # Check that the file is valid
        if not os.path.exists(input_file):
            raise ValueError("Input file not found")

        # Check that the filename values match those in the config file
        date, channel, bias = parse_compass_file_name(dictionary["file"])

        if channel != "0":
            raise ValueError("Channel from Filename Does not Correspond to SiPM")

        if bias != dictionary["bias"]:
            raise ValueError(
                "Bias from filename does not match bias provided in config file"
            )

        out_form = (
            input_file,
            bias,
            dictionary["device_name"],
            dictionary["vpp"],
            dictionary["start_idx"],
            dictionary["end_idx"],
            output_file,
        )

        out_args.append(out_form)

    return out_args


def parse_light_json(json_file_name: str) -> list[list, ...]:
    """
        Parse a config file defined specifically for the light/photon rate measurement script

        Parameters
        ----------
        json_file_name
            Path to a json file containing parameters used for performing light analysis

        Returns
        -------
        out_args
            List of lists. Each list looks like ["path/to/input/input", bias, device_name, vpp, "path/to/gain/gain_file.h5", light_window_start_idx, light_window_end_idx, dark_window_start_idx, dark_window_end_idx, "path/to/output/output.h5"]


        Notes
        -----
        Must be performed after the `.gain.gain` analysis script has been run on dark characterization data

        Example Config
        --------------
        {
        "input_path": "/path/to/raw/files",
        "output_path": "/path/to/analyzed/data",
        "output_file_name": "light_output_name.h5",
        "input_files": [
        {"file": "t1_Data_Channel@DT5730_1463_devicename_dark/light/_rt/ln_MONTH/DAY/YEAR_apd_apdbiasinvolts_sipm_sipmbiasindecivolt_(optional LED info)
    ", "bias": 54, "device_name": "sipm_1st", "vpp": 0.5, "gain_file": "path/to/gain/file/gain_file.h5", "light_window_start_idx": 50, "light_window_end_idx": 250, "dark_window_start_idx": 4000, "dark_window_end_idx": 4200  },
        {"file": "t1_Data_Channel@DT5730_1463_devicename_dark/light/_rt/ln_MONTH/DAY/YEAR_apd_apdbiasinvolts_sipm_sipmbiasindecivolt_(optional LED info)
    ", "bias": 54.5, "device_name": "sipm_1st", "vpp": 0.5, "gain_file": "path/to/gain/file/gain_file.h5", "light_window_start_idx": 50, "light_window_end_idx": 250, "dark_window_start_idx": 4000, "dark_window_end_idx": 4200   }
        ],
        }
    """
    f = open(json_file_name)
    json_file = json.load(f)
    f.close()
    # make the output directory if we need to
    if not os.path.exists(json_file["output_path"]):
        os.mkdir(json_file["output_path"])

    # make sure we don't override an existing analysis
    if os.path.exists(
        os.path.join(json_file["output_path"], json_file["output_file_name"])
    ):
        raise ValueError("Output file already exists")

    # go through each file, join the paths, check that the names are valid, and then put in a list that can
    # be passed to multiprocessing.pool

    out_args = []
    for dictionary in json_file["input_files"]:
        input_file = os.path.join(json_file["input_path"], dictionary["file"])
        output_file = os.path.join(
            json_file["output_path"], json_file["output_file_name"]
        )

        # Check that the file is valid
        if not os.path.exists(input_file):
            raise ValueError("Input file not found")

        # Try reading in the gain file to see that it will work for our analysis
        if not os.path.exists(dictionary["gain_file"]):
            raise ValueError("Gain file not found!")
        # try:
        #     f = h5py.File(dictionary["gain_file"], "r")
        #     n_gains = f[f'{dictionary["bias"]}/gain'][()]
        #     f.close()
        # except:
        #     raise ValueError("SiPM bias not found in gain file!")

        # Check that the filename values match those in the config file
        date, channel, bias = parse_compass_file_name(dictionary["file"])

        if channel not in ["0", "1"]:
            raise ValueError("Channel from Filename Does not Correspond to SiPM or APD")

        if bias != dictionary["bias"]:
            raise ValueError(
                "Bias from filename does not match bias provided in config file"
            )

        if (channel == "1") and (
            str(dictionary["device_name"]) not in ["apd", "apd_goofy"]
        ):
            raise ValueError("Channel number does not agree that this is an APD")

        if (channel == "0") and (
            str(dictionary["device_name"])
            not in ["sipm", "sipm_1st", "sipm_1st_low_gain"]
        ):
            raise ValueError("Channel number does not agree that this is a SiPM")

        # Now create a tuple that multiprocessor.pool can take as an input
        out_form = (
            input_file,
            bias,
            dictionary["device_name"],
            dictionary["vpp"],
            dictionary["gain_file"],
            dictionary["light_window_start_idx"],
            dictionary["light_window_end_idx"],
            dictionary["dark_window_start_idx"],
            dictionary["dark_window_end_idx"],
            output_file,
        )

        out_args.append(out_form)

    return out_args
