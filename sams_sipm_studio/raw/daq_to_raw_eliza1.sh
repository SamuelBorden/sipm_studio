#!/bin/bash

#$ -S /bin/bash                      # use bash
#$ -m n                              # don't send mail when job starts or stops.
#$ -w e                              # verify syntax and give error if so
#$ -V                                # inherit environment variables
#$ -N compass_daq_to_raw             # job name
#$ -e /data/eliza1/LEGEND/jobs/logs  # error output of script
#$ -o /data/eliza1/LEGEND/jobs/logs  # standard output of script
#$ -l h_rt=10:00:00                  # hard time limit, your job is killed if it uses this much cpu.
#$ -l s_rt=9:50:00                   # soft time limit, your job gets signaled when you use this much time. Maybe you can gracefully shut down?
#$ -cwd                              # execute from the current working directory
#$ -t 1                              # give me N identical jobs, labelled by variable SGE_TASK_ID
#$ -l scratch=10G                    # Give me 10 gigs of data in scratch
#$ -pe smp 5                        # Give me fifteen processors on this node

#execute the $SGE_TASK_ID'th sub-job
set -x # I think this makes everything beyond this call get saved to the log


echo ${NSLOTS}
echo $TMPDIR

cd /data/eliza1/LEGEND/data/LNsipm

mkdir processing

mv waiting/* processing

singularity exec --bind /data/eliza1/LEGEND/data/LNsipm:/data/eliza1/LEGEND/data/LNsipm /data/eliza1/LEGEND/sw/containers/legend-base.sif /home/sjborden/sams_sipm_studio/raw/submit_daq_to_raw.sh


mv processing/* processed/processed_daq
rmdir processing


date
set +x

exit
