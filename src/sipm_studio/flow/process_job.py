"""
This handles the distribution of jobs to processors.
This python script takes a json file path as a -i input, as well as a -s input to specify which processor
this json config file is going to run with.
"""
# import os module
import os

# turn off file locking
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# import python modules
import argparse
import numpy as np
import h5py
import multiprocessing as mp
import json

from sipm_studio.util import parse_json_config
from sipm_studio.light import pde_pulse_method
from sipm_studio.raw import daq_to_raw
from sipm_studio.dark import dark_processors


# Setup the argument parser
__pars__ = argparse.ArgumentParser()

# include the arguments
__pars__.add_argument("-i", type=str, default="", help="path to the input json file")
__pars__.add_argument(
    "-s", type=str, default="pde_pulse", help="stage/processor to run"
)
__pars__.add_argument("-c", type=int, default=1, help="set the number of cores")

# parse the arguments
__args__ = __pars__.parse_args()

# interpret the arguments
proc_count = __args__.c if __args__.c < mp.cpu_count() else mp.cpu_count()
proc_input = __args__.i
proc_stage = __args__.s


if __name__ == "__main__":

    print("processing counts", proc_count, proc_input, proc_stage)

    # Get the correct arguments to run, and then run the mp
    if proc_stage == "pde_pulse":
        args = parse_json_config.parse_pde_json(proc_input)
        print(args)

        m = mp.Manager()
        l = m.Lock()

        args = [[*arg, l] for arg in args]  # pass the lock?

        # launch the parallel processes
        with mp.Pool(proc_count) as p:
            p.starmap(pde_pulse_method.calculate_pulse_pde, args, chunksize=1)
        print("exited gracefully.")

    if proc_stage == "build_raw":
        args = parse_json_config.parse_raw_json_config(proc_input)
        print(args)

        # launch the parallel processes
        with mp.Pool(proc_count) as p:
            p.starmap(daq_to_raw.build_raw, args, chunksize=1)
        print("exited gracefully.")

    if proc_stage == "dark":
        args = parse_json_config.parse_dark_json_config(proc_input)
        print(args)

        m = mp.Manager()
        l = m.Lock()

        args = [[*arg, l] for arg in args]  # pass the lock?

        # launch the parallel processes
        with mp.Pool(proc_count) as p:
            p.starmap(dark_processors.run_dark, args, chunksize=1)
        print("exited gracefully.")
