"""
Returns the date, channel number, and SiPM bias from a CoMPASS file name. 
All file names must have the format: 
t1_Data_Channel@DT5730_1463_devicename_dark/light/_rt/ln_MONTH/DAY/YEAR_apd_apdbiasinvolts_sipm_sipmbiasindecivolt_(optional LED info)
"""

def parse_compass_file_name(file_name :str) -> tuple(str, int, float): 
    """ 
    Reads in a CoMPASS file name and returns the date, channel number, and sipm bias

    Parameters 
    ----------
    file_name
        Reads in a file name with the following format: 
        `t1_Data_Channel@DT5730_1463_devicename_dark/light/_rt/ln_MONTH/DAY/YEAR_apd_apdbiasinvolts_sipm_sipmbiasindecivolt_(optional LED info)` 
    
    Returns 
    -------
    date 
        Date that the CoMPASS file was recorded on
    channel_num
        Channel number for the file, 0 is SiPM, 1 is APD 
    sipm_bias 
        The SiPM bias that the data was recorded at
    """
    channel_num = file_name.split("CH")[1].split("@")[0]
    
    sipm_bias = int(file_name.split("dv")[0][-3:])
    sipm_bias /= 10 # convert decivolts to volts 
    

    splits = file_name.split("_")
    date_idx = 0
    for i, split in enumerate(splits):
        try: 
            datetime.strptime(str(split), '%m/%d/%Y')
            date_idx = i
        except:
            pass
    if date_idx == 0 :
        raise ValueError("Date time could not be processed from filename")
    else:
        date = splits[date_idx]
        
    return date, channel_num, sipm_bias
    
    
    
    

