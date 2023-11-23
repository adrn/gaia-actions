#!/bin/bash
#SBATCH -J actions
#SBATCH -o logs/actions.o%j
#SBATCH -e logs/actions.e%j
#SBATCH -N 12
#SBATCH -t 36:00:00
#SBATCH -p cca
# --constraint=rome

source ~/.bash_profile

cd /mnt/ceph/users/apricewhelan/projects/gaia-actions/scripts

# init_conda
init_env

date

# mpirun python compute_actions.py -v -f ~/data/GaiaEDR3/edr3-rv-good-plx-result.fits.gz --mpi

# 2022-06-09
# Generating EDR3 file for Jason Hunt
# mpirun python compute_actions_staeckel.py -f ~/data/GaiaEDR3/edr3-rv-good-plx-result.fits --mpi --potential=../potentials/MilkyWayPotential2022.yml

# 2022-08-19
# DR3
# NOTE: updated to integrate longer before computing ecc, peri, apo
# mpirun python compute_actions_staeckel.py -f ~/data/GaiaDR3/dr3-rv-good-plx.fits --mpi --potential=../potentials/MilkyWayPotential2022.yml --staeckel

# 2022-12-12
# 2023-02-27
# 2023-03-30
# mpirun python compute_actions.py -v --mpi --staeckel \
# -f ~/data/Gaia/DR3/dr3-rv-good-plx.fits \
# --id-col=source_id -p ../potentials/MilkyWayPotential2022.yml

# Vedant: DR3 Bailer-Jones distances
mpirun python compute_actions.py -v --mpi --staeckel \
    -f ../data/a23_tab2_bjcoords.h5 \
    --id-col=source_id \
    -p ../potentials/MilkyWayPotential2022.yml \
    --dist-col=r_med_geo_kpc

date
