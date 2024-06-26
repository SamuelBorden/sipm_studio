"""
Returns the date, channel number, and SiPM bias from a CoMPASS file name.
All file names must have the format
t1_Data_Channel@DT5730_1463_devicename_dark/light/_rt/ln_MONTH/DAY/YEAR_apd_apdbiasinvolts_sipm_sipmbiasindecivolt_(optional LED info)
"""
from datetime import datetime


def parse_compass_file_name(file_name: str):
    """
    Reads in a CoMPASS file name and returns the date, channel number, and sipm bias

    Parameters
    ----------
    file_name
        Reads in a file name with the following format:
        `t1_Data_Channel@DT5730_1463_devicename_dark/light/_rt/ln_MONTH-DAY-YEAR_apd_apdbiasinvolts_sipm_sipmbiasindecivolt_(optional LED info)`


    Returns
    -------
    date
        Date that the CoMPASS file was recorded on
    channel_num
        Channel number for the file, 0 is SiPM, 1 is reference diode
    sipm_bias
        The SiPM bias that the data was recorded at
    """
    if ("CH" not in file_name) or (len(file_name.split("CH")) < 2):
        raise ValueError("File not named with 'CH0@' format.")
    channel_num = file_name.split("CH")[1].split("@")[0]

    if "dv" not in file_name:
        raise ValueError("File not named with SiPM bias in decivolts (dv).")

    if "dv" in file_name:
        sipm_bias = int(file_name.split("dv")[0][-3:])
        sipm_bias /= 10  # convert decivolts to volts
    elif "mv" in file_name:
        sipm_bias = int(file_name.split("mv")[0][-4:])
        sipm_bias /= 100  # convert decivolts to volts
    else:
        raise ValueError("Cannot find correct SiPM bias.")

    splits = file_name.split("_")
    date_idx = 0
    print(splits)
    for i, split in enumerate(splits):
        try:
            datetime.strptime(str(split), "%m-%d-%Y")
            date_idx = i
        except:
            pass
    if date_idx == 0:
        print("Date time could not be processed from filename.")
        date = "00-00-0000"  # default nan date
    else:
        date = splits[date_idx]

    print(date, channel_num, sipm_bias)

    return date, channel_num, sipm_bias
