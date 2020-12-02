#!/bin/bash
#SBATCH -J actions          # job name
#SBATCH -o actions.o%j             # output file name (%j expands to jobID)
#SBATCH -e actions.e%j             # error file name (%j expands to jobID)
#SBATCH -N 4
#SBATCH -t 24:00:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH -p cca

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/dr2-actions/scripts

module load gcc openmpi2

date

srun python compute-freqs.py -v -f derp --mpi

date
