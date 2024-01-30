Processing PDE Data
=========================

The photon detection efficiency can be calculated from data taken while triggering on an LED flash. The function :func:`pde_pulse_method.calculate_pulse_pde` helps to compute the PDE.
It does this by saving the mean number of photons detected to a file. The mean number of photons detected is computed by

- Using amplifier values to convert waveforms in ADC to current
- Finding and ubstracting the average baseline of all the waveforms
- Integrating waveforms in a user-specified time window and constructing a charge histogram
- Finding the tallest point in the charge spectrum and then using the inflection point to find the distance to the next peak
- Fitting the peaks in a charge spectrum with Gaussian functions
- Finally, computing the mean number of photons detected by taking the negative log ratio of the counts in the pedestal peak over the total number of waveforms

The processing of PDE analysis can be done in parallel using the `examples.submit_pde_processing.sh` file. This job submission takes a json file as its input to specify the analysis to perform.
Here's an example of the configuration json file, in addition to the one that can be found at `examples.process_pde_json`:

.. code-block:: json

   {
    "input_path": "/data/eliza1/LEGEND/data/LNsipm/processed",
    "output_path": "/home/sjborden/automatic_pde_calculator_results",
    "led_wavelength": 560,
    "output_file_name": "ketek_rt_lid_pde_10-13-2023_1.h5",
    "input_files": [
        {"file": "t1_Data_CH0@DT5730_1463_ketek_rt_pde_lid_10-13-2023_248dv_broadcom_3440mv_led_3kHz_16ns_330v.BIN.h5" , "bias": 24.8, "device_name": "sipm", "vpp": 0.5, "light_window_start_idx": 248, "light_window_end_idx": 598, "dark_window_start_idx": 1650, "dark_window_end_idx": 2000}
    ]
   }

- The `input_path` key has the full path to where all of the PDE data to be processed are located.
- The `output_path` key has the full path of where the output analysis file will be located. Monitoring plots will also appear here.
- The `led_wavelength` key has an integer that specifies which LED wavelength data are being analyzed, so that the reference diode's PDE can be used correctly.
- The `output_file_name` key has the name of the output analysis file.
- The `input_files` key has a list of dictionaries containing the following: the  `file` key which has the name of the file to analyze, the `bias` key which has the bias the SiPM was set at, the `device_name` key which is either `sipm` or `reference` depending on which device's data are being analyzed, the `vpp` key which holds the voltage division CoMPASS was recording data at, and then the integration window values specified in samples in which to integrate the waveforms.
