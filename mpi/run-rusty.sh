#!/bin/bash
#SBATCH -J actions
#SBATCH -o logs/actions.o%j
#SBATCH -e logs/actions.e%j
#SBATCH -N 8
#SBATCH -t 24:00:00
#SBATCH -p cca
#SBATCH --constraint=skylake

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/gaia-actions/scripts

init_conda

date

mpirun python compute_actions.py -v -f ~/data/GaiaEDR3/edr3-rv-good-plx-result.fits.gz --mpi -o

date
