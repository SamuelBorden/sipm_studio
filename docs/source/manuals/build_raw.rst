Creating Raw Files
=========================

The processing of binary CoMPASS DAQ files into HDF5 files is done by the :func:`daq_to_raw.process_metadata` function.
This function can process multiple files in parallel on a computing cluster.

The file :func:`daq_to_raw_eliza1.sh` is written specifically for the CENPA cluster and should be submitted as a job by users wanting to process files.
It scans for binary files in the :func:`/data/eliza1/LEGEND/data/LNsipm/waiting` directory.
It then processes them to the raw stage and stores them in :func:`/data/eliza1/LEGEND/data/LNsipm/processed`.
