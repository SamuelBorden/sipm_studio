"""
Returns the date, channel number, and SiPM bias from a CoMPASS file name.
All file names must have the format:
t1_Data_Channel@DT5730_1463_devicename_dark/light/_rt/ln_MONTH/DAY/YEAR_apd_apdbiasinvolts_sipm_sipmbiasindecivolt_(optional LED info)
"""
import pytest


def test_parse_compass_file_name():
    assert 1 == 1
