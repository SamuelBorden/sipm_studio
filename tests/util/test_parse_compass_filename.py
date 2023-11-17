"""
Returns the date, channel number, and SiPM bias from a CoMPASS file name.
All file names must have the format:
t1_Data_CH0@DT5730_1463_devicename_dark/light/rt/ln_MONTH/DAY/YEAR_apd_apdbiasinvolts_sipm_sipmbiasindecivolt_(optional LED info, led in volts)
"""
import pytest
from sipm_studio.util.parse_compass_filename import parse_compass_file_name


def test_parse_compass_file_name():
    test_file_name = "t1_Data_CH0@DT5730_1463_sipm_dark_rt_03-10-2023_apd_150V_sipm_545dv_led_1kHz_500ns_210dv.BIN"
    date, channel_num, bias = parse_compass_file_name(test_file_name)

    assert date == "03-10-2023"
    assert int(channel_num) == 0
    assert bias == 54.5

    # Make sure we get exceptions in wrong date case
    bad_test_name = "t1_Data_CH1@DT5730_1463_sipm_dark_rt_03-100-2023_apd_150V_sipm_545dv_led_1kHz_500ns_210v.BIN"
    date, channel_num, bias = parse_compass_file_name(bad_test_name)
    assert date == "00-00-0000"

    # Make sure we get exceptions in wrong channel name case
    bad_test_name = "t1_Data_Channel1@DT5730_1463_sipm_dark_rt_03-10-2023_apd_150V_sipm_545dv_led_1kHz_500ns_210v.BIN"
    with pytest.raises(ValueError) as exc_info:
        date, channel_num, bias = parse_compass_file_name(bad_test_name)

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "File not named with 'CH0@' format."

    # Make sure we get exceptions in wrong sipm bias case
    bad_test_name = "t1_Data_CH1@DT5730_1463_sipm_dark_rt_03-10-2023_apd_150V_sipm_55.0v_led_1kHz_500ns_210v.BIN"
    with pytest.raises(ValueError) as exc_info:
        date, channel_num, bias = parse_compass_file_name(bad_test_name)

    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "File not named with SiPM bias in decivolts (dv)."
