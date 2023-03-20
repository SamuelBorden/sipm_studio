"""
Define functions that parse a json config file for various measurement configurations. 
As an input takes the json file and the measurement level specified by the user and returns parameters needed
for those specific measurements
"""
import os, json, h5py
import numpy as np 
from parse_config_filename import parse_compass_file_name

def parse_gain_json(json_file_name: str) -> list(list, ... ):
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
    {"file": "t1_Data_Channel@DT5730_1463_devicename_dark/light/_rt/ln_MONTH/DAY/YEAR_apd_apdbiasinvolts_sipm_sipmbiasindecivolt_(optional LED info)
", "bias": 54, "device_name": "sipm_1st", "vpp": 0.5, "start_idx": 50, "end_idx": 250 },
    {"file": "t1_Data_Channel@DT5730_1463_devicename_dark/light/_rt/ln_MONTH/DAY/YEAR_apd_apdbiasinvolts_sipm_sipmbiasindecivolt_(optional LED info)
", "bias": 54, "device_name": "sipm_1st", "vpp": 0.5, "start_idx": 50, "end_idx": 250 }
    ],
    }
    """
    json_file = json.loads(json_file_name)
    
    # make the output directory if we need to
    if not os.path.exists(json_file["output_path"]):
        os.mkdir(json_file["output_path"])
        
    # make sure we don't override an existing analysis 
    if os.path.exists(os.path.join(json_file["output_path"], json_file["output_file_name"])):
        raise ValueError("Output file already exists")
        
    # go through each file, join the paths, check that the names are valid, and then put in a list that can
    # be passed to multiprocessing.pool 
    
    out_args = []
    for dict in json_file["input_files"]:
        input_file = os.path.join(json_file["input_path"], dict["file"])
        output_file = os.path.join(json_file["output_path"], json_file["output_file_name"])
        
        # Check that the file is valid 
        if not os.path.exists(input_file):
            raise ValueError("Input file not found")
        
        # Check that the filename values match those in the config file 
        date, channel, bias = parse_compass_file_name(dict["file"])
        
        if channel != "0": 
            raise ValueError("Channel from Filename Does not Correspond to SiPM")
            
        if bias != dict["bias"]:
            raise ValueError("Bias from filename does not match bias provided in config file")
        
        out_form = (input_file, bias, dict["device_name"], dict["vpp"], dict["start_idx"], dict["end_idx"], output_file)
        
        out_args.append(out_form)
        
    return out_args
            
        
            
        
    
    
    
