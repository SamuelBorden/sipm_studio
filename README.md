# sipm_studio
Software for the automatic characterization of silicon photomultipliers (SiPMs).

*sipm_studio* provides the tools for:

- Conversion of CoMPASS binary DAQ files to the HDF5 file format via the *raw* submodule
- Automatic calculation of the gain via the *gain* submodule
- Computation of the PDE using either the integration method or the pedestal method via the *light* submodule
- Parallelization of PDE processing as shown in the *examples* submodule

See documentation here: https://sipm-studio.readthedocs.io/en/latest/
