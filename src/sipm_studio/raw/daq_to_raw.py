"""
Convert a CoMPASS Binary (.bin or .BIN) into a `raw` tier format h5py file.
"""
import os
import time
import sys

sys.path.append("/opt/anaconda3/lib/python3.8/site-packages")
sys.path.append("/opt/anaconda3/bin/python")
sys.path.append("/opt/anaconda3/lib/python3.8/site-packages/h5py")
import h5py
import peakutils
import numpy as np
import glob
import argparse
import multiprocessing as mp


# # Create some parser arguments on how many cores to multiprocess over

# parser = argparse.ArgumentParser(description="Convert CoMPASS files to h5 files")
# parser.add_argument(
#     "-c",
#     "--core_num",
#     help="number of cores to distribute files over",
#     type=int,
#     default=1,
# )
# args = parser.parse_args()

# # set the number of processors from the user's specification
# num_processors = int(args.core_num)

# # just need it to glob together all files in a folder, pass the folder and then take the last part for the file name
# # so that you don't accidentally pass it a path instead of a file name

# files = glob.glob("/data/eliza1/LEGEND/data/LNsipm/processing" + "/*.bin")
# files_2 = glob.glob("/data/eliza1/LEGEND/data/LNsipm/processing" + "/*.BIN")

# if len(files) == 0:
#     files = files_2


def get_event_size(t0_file: str):
    """
    Get the length of the waveforms in the CoMPASS file, and also tell if the binary files is a CoMPASS2.0 file or not.

    Parameters
    ----------
    t0_file
        Path to binary CoMPASS file

    Returns
    -------
    event size, flag that's true if the file is CoMPASS2.0
    """
    with open(t0_file, "rb") as file:
        first_header = file.read(2)
        file.close()

    # all CoMPASS 2.0 files have a flag in the second byte that is equal to 202 in decimal
    # This is for getting the waveform length in CoMPASS 2.0

    if first_header[1] == 202:
        with open(t0_file, "rb") as file:
            first_event = file.read(27)
            first_header_check = first_event[1]
            [num_samples] = np.frombuffer(first_event[23:27], dtype=np.uint32)
        return (
            25 + 2 * num_samples,
            True,
        )  # 25 bytes in the header per event and then 2*number of sample bytes for the waveform

    # This is for getting the waveform length in CoMPASS 1.0

    else:
        with open(t0_file, "rb") as file:
            first_event = file.read(24)
            [num_samples] = np.frombuffer(first_event[20:24], dtype=np.uint32)
        return 24 + 2 * num_samples, False  # number of bytes / 2


def get_event(event_data_bytes: bytes):
    """
    Decodes an event packet. Get all info associated with a waveform in CoMPASS 1.0, return it in an array

    Parameters
    ----------
    event_data_bytes
        The packet for a single event read from a binary file as bytes

    Returns
    -------
    board number, channel number, timestamp in ps, energy, energy short, flags, number of samples, and waveform
    """

    board = np.frombuffer(event_data_bytes[0:2], dtype=np.uint16)[0]
    channel = np.frombuffer(event_data_bytes[2:4], dtype=np.uint16)[0]
    timestamp = np.frombuffer(event_data_bytes[4:12], dtype=np.uint64)[0]
    energy = np.frombuffer(event_data_bytes[12:14], dtype=np.uint16)[0]
    energy_short = np.frombuffer(event_data_bytes[14:16], dtype=np.uint16)[0]
    flags = np.frombuffer(event_data_bytes[16:20], np.uint32)[0]
    num_samples = np.frombuffer(event_data_bytes[20:24], dtype=np.uint32)[0]
    waveform = np.frombuffer(event_data_bytes[24:], dtype=np.uint16)

    return _assemble_data_row(
        board, channel, timestamp, energy, energy_short, flags, num_samples, waveform
    )


def get_event_v2(event_data_bytes: bytes):
    """
    Decodes an event packet. Get all info associated with a waveform in CoMPASS 2.0, return it in an array.

    Parameters
    ----------
    event_data_bytes
        The packet for a single event read from a binary file as bytes

    Returns
    -------
    board number, channel number, timestamp in ps, energy, energy short, flags, number of samples, and waveform
    """
    board = np.frombuffer(event_data_bytes[0:2], dtype=np.uint16)[0]
    channel = np.frombuffer(event_data_bytes[2:4], dtype=np.uint16)[0]
    timestamp = np.frombuffer(event_data_bytes[4:12], dtype=np.uint64)[0]
    energy = np.frombuffer(event_data_bytes[12:14], dtype=np.uint16)[0]
    energy_short = np.frombuffer(event_data_bytes[14:16], dtype=np.uint16)[0]
    flags = np.frombuffer(event_data_bytes[16:20], np.uint32)[0]

    code = np.frombuffer(event_data_bytes[20:21], np.uint8)[0]

    num_samples = np.frombuffer(event_data_bytes[21:25], dtype=np.uint32)[0]
    waveform = np.frombuffer(event_data_bytes[25:], dtype=np.uint16)

    return _assemble_data_row(
        board, channel, timestamp, energy, energy_short, flags, num_samples, waveform
    )


def _assemble_data_row(
    board: int,
    channel: int,
    timestamp: float,
    energy: float,
    energy_short: float,
    flags: float,
    num_samples: int,
    waveform: np.array,
):
    """
    Create an array from the event data.

    TODO: moonlight this function
    """
    timestamp = timestamp
    energy = energy
    energy_short = energy_short
    flags = flags
    waveform = waveform
    return np.array([timestamp, energy, energy_short, flags]), waveform


# Write the massive output array to one h5 file


def _output_to_h5file(
    data_file: str,
    output_name: str,
    output_path: str,
    events: list,
    waveforms: np.array,
    baselines: np.array,
) -> None:
    """
    Given an output file name and output path, write the given data to an h5py file.

    Parameters
    ----------
    data_file
        TODO: deprecate
    output_name
        The name of the h5 file to write to
    output_path
        The path for the h5 output file
    events
        A list of information from the decoded events in a CoMPASS binary file,  timestamp, energy, energy_short, flags]
    waveforms
        A numpy array of equal sized numpy arrays containing waveform data from CoMPASS
    baselines
        A numpy array of equal sized numpy arrays containing calculated baselines from waveform data from CoMPASS
    """
    destination = os.path.join(output_path, "t1_" + output_name + ".h5")
    with h5py.File(destination, "w") as output_file:
        output_file.create_dataset("/raw/timetag", data=events.T[0])
        output_file.create_dataset("/raw/energy", data=events.T[1])
        output_file.create_dataset("/raw/waveforms", data=waveforms)
        output_file.create_dataset("/raw/baselines", data=baselines)
        output_file.create_dataset("adc_to_v", data=2 / (2**14))


# Process all the files, for each file read in waveform by waveform and append data to one massive array that is written to file once


def process_metadata(files: str, output_path: str) -> None:
    r"""
    Process all the files, for each file read in waveform by waveform and append data to one massive array that is written to file once.

    Parameters
    ----------
    files
        A CoMPASS binary files that follow the regex "/data/eliza1/LEGEND/data/LNsipm/processing" + "/\*.BIN"
    output_path
        The path to a directory to which to write all output


    Notes
    -----
    This function requires files to be placed in the "processing" directory under "/data/eliza1/LEGEND/data/LNsipm/".
    It works by reading in the first 2 bytes of a binary file to determine if it is a CoMPASS v2 file or not using :func:`get_event_size`.
    Then, it iterates over the bytes in file and decodes them with :func:`get_event_v2` or :func:`get_event. Finally, it writes the decoded
    values to disk using :func:`_output_to_h5file`.

    TODO: rename the files parameter `file` as this only takes one file at a time
    TODO: rename this function, this processes actual data and not metadata.

    """

    file_name = files

    #         print("processing file:")
    #         print(file_name)
    event_rows = []
    waveform_rows = []
    baseline_rows = []

    # get the waveform size, and also check if it is a CoMPASS v2 file or not
    event_size, flag = get_event_size(file_name)

    without_extra_slash = os.path.normpath(file_name)
    last_part = os.path.basename(without_extra_slash)

    # flag determines if it is a CoMPASS v2 file or not.
    if flag:

        with open(file_name, "rb") as metadata_file:
            file_header = metadata_file.read(
                2
            )  # read in the header present in v2 Compass...
            event_data_bytes = metadata_file.read(event_size)
            # Decode
            while event_data_bytes != b"":
                event, waveform = get_event_v2(event_data_bytes)
                baseline = peakutils.baseline(waveform)
                event_rows.append(event)
                waveform_rows.append(waveform)
                baseline_rows.append(baseline)
                event_data_bytes = metadata_file.read(event_size)
        # Write to file
        _output_to_h5file(
            file_name,
            last_part,
            output_path,
            np.array(event_rows),
            np.array(waveform_rows),
            np.array(baseline_rows),
        )

    if not flag:

        with open(file_name, "rb") as metadata_file:
            event_data_bytes = metadata_file.read(event_size)
            # Decode
            while event_data_bytes != b"":
                event, waveform = get_event(event_data_bytes)
                baseline = peakutils.baseline(waveform)
                event_rows.append(event)
                waveform_rows.append(waveform)
                baseline_rows.append(baseline)
                event_data_bytes = metadata_file.read(event_size)
        # Output to h5 file
        _output_to_h5file(
            file_name,
            last_part,
            output_path,
            np.array(event_rows),
            np.array(waveform_rows),
            np.array(baseline_rows),
        )


# # Now run the program
# if __name__ == "__main__":
#     """
#     Parameters
#     ----------
#     files is a list of globbed binary files, determined from a global variable set in this file to look at a specific directory
#     """

#     hardcoded_path = "/data/eliza1/LEGEND/data/LNsipm/processed"
#     hardcoded_path_array = np.full(len(files), hardcoded_path)

#     with mp.Pool(num_processors) as p:
#         p.starmap(process_metadata, zip(files, hardcoded_path_array))

#     # End the program
