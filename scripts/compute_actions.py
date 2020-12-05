# Standard library
import os
os.environ["OMP_NUM_THREADS"] = "1"
import logging
import pathlib
import sys

# Third-party
from astropy.utils import iers
iers.conf.auto_download = False
import astropy.table as at
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import h5py
from pyia import GaiaData

import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from schwimmbad.utils import batch_tasks

logger = logging.getLogger(__name__)
logging.basicConfig()

integrate_time = 25. * u.Gyr  # HACK: hardcoded! ~100 orbital periods
Nmax = 8  # for find_actions


def worker(task):
    (i, j), idx, galcen, meta, pot, cache_file, id_colname, ids = task
    ids = ids[idx]
    galcen = galcen[idx]

    w0 = gd.PhaseSpacePosition(galcen.cartesian)
    H = gp.Hamiltonian(pot)

    logger.debug(f"Worker {i}-{j}: running {j-i} tasks now")

    # Set up data containers:
    all_data = {}
    for k, info in meta.items():
        if k == id_colname:
            all_data[k] = ids
        else:
            shape = (len(ids), ) + info['shape'][1:]
            all_data[k] = np.full(shape, np.nan)

    for n in range(len(galcen)):
        all_data[id_colname][n] = ids[n]
        all_data['xyz'][n] = galcen.data.xyz[:, n].to_value(
            meta['xyz']['unit'])
        all_data['vxyz'][n] = galcen.velocity.d_xyz[:, n].to_value(
            meta['vxyz']['unit'])

        try:
            orbit = H.integrate_orbit(w0[n], dt=0.5*u.Myr, t1=0*u.Myr,
                                      t2=integrate_time,
                                      Integrator=gi.DOPRI853Integrator)
        except Exception as e:
            logger.error(f'Failed to integrate orbit {i}\n{str(e)}')
            continue

        # Compute actions / frequencies / angles
        try:
            res = gd.find_actions(orbit, N_max=Nmax)
            all_data['actions'][n] = res['actions'].to_value(
                meta['actions']['unit'])
            all_data['angles'][n] = res['angles'].to_value(
                meta['angles']['unit'])
            all_data['freqs'][n] = res['freqs'].to_value(
                meta['freqs']['unit'])
        except Exception as e:
            logger.error(f'Failed to run find actions for orbit {i}\n{str(e)}')

        # Other various things:
        try:
            rper = orbit.pericenter(approximate=True).to_value(
                meta['r_per']['unit'])
            rapo = orbit.apocenter(approximate=True).to_value(
                meta['r_apo']['unit'])

            all_data['z_max'][n] = orbit.zmax(approximate=True).to_value(
                meta['z_max']['unit'])
            all_data['r_per'][n] = rper
            all_data['r_apo'][n] = rapo
            all_data['ecc'][n] = (rapo - rper) / (rapo + rper)
        except Exception as e:
            logger.error(f'Failed to compute zmax peri apo for orbit {i}\n{e}')

        # Lz and E
        try:
            all_data['L'][n] = np.mean(
                orbit.angular_momentum().to_value(meta['L']['unit']), axis=1)
            all_data['E'][n] = np.mean(
                orbit.energy().to_value(meta['E']['unit']))
        except Exception as e:
            logger.error(f'Failed to compute E Lz for orbit {i}\n{e}')

    return idx, cache_file, all_data


def callback(res):
    idx, cache_file, all_data = res

    logger.debug(f'Writing block {idx[0]}-{idx[-1]} to cache file')
    with h5py.File(cache_file, 'r+') as f:
        for k in all_data:
            f[k][idx] = all_data[k]


def main(pool, source_file, overwrite=False,
         id_colname=None, dist_colname=None, rv_colname=None):

    logger.debug(f'Starting file {source_file}...')

    cache_path = pathlib.Path(__file__).parent / '../cache'
    cache_path = cache_path.resolve()
    cache_path.mkdir(exist_ok=True)

    source_file = pathlib.Path(source_file).resolve()
    cache_file = cache_path / f"{source_file.name.split('.')[0]}.hdf5"
    logger.debug(f'Writing to cache file {cache_file}'.format(cache_file))

    # Global parameters
    with coord.galactocentric_frame_defaults.set('v4.0'):
        gc_frame = coord.Galactocentric()
    mw = gp.MilkyWayPotential()

    # Load the source data table:
    g = GaiaData(at.QTable.read(source_file))
    g = g[:5]

    mask = np.ones(len(g), dtype=bool)
    if id_colname is None:  # assumes gaia
        id_colname = 'source_id'
        ids = g.source_id
    else:
        ids = g.data[id_colname]

    if dist_colname is None:  # assumes gaia
        dist = coord.Distance(parallax=g.parallax, allow_negative=True)
        mask &= np.isfinite(dist)
    else:
        dist = g.data[dist_colname]

    if rv_colname is None:  # assumes gaia
        if hasattr(g, 'radial_velocity'):
            rv = g.radial_velocity
        elif hasattr(g, 'dr2_radial_velocity'):
            rv = g.dr2_radial_velocity
        else:
            raise ValueError('...')
        mask &= np.isfinite(rv)
    else:
        rv = g.data[rv_colname]

    # Get coordinates, and only keep good values:
    if ~np.all(mask):
        logger.warn(f"Filtering {mask.sum()} bad distance or RV values")

    c = g[mask].get_skycoord(distance=dist[mask],
                             radial_velocity=rv[mask])
    ids = ids[mask]

    galcen = c.transform_to(gc_frame)
    logger.debug('Data loaded...')

    Nstars = len(c)

    # Column metadata: map names to shapes
    meta = {
        id_colname: {'shape': (Nstars, ),
                     'dtype': g.data[id_colname].dtype,
                     'fillvalue': None},
        'xyz': {'shape': (Nstars, 3), 'unit': u.kpc},
        'vxyz': {'shape': (Nstars, 3), 'unit': u.km/u.s},
        # Frequencies, actions, and angles computed with Sanders & Binney
        'freqs': {'shape': (Nstars, 3), 'unit': 1/u.Gyr},
        'actions': {'shape': (Nstars, 3), 'unit': u.kpc * u.km/u.s},
        'angles': {'shape': (Nstars, 3), 'unit': u.rad},
        # Orbit parameters:
        'z_max': {'shape': (Nstars, ), 'unit': u.kpc},
        'r_per': {'shape': (Nstars, ), 'unit': u.kpc},
        'r_apo': {'shape': (Nstars, ), 'unit': u.kpc},
        'ecc': {'shape': (Nstars, ), 'unit': u.one},
        'L': {'shape': (Nstars, 3), 'unit': u.kpc * u.km/u.s},
        'E': {'shape': (Nstars, ), 'unit': (u.km/u.s)**2},
    }

    # Make sure output file exists
    if not cache_file.exists() or overwrite:
        with h5py.File(cache_file, 'w') as f:
            for name, info in meta.items():
                d = f.create_dataset(name, shape=info['shape'],
                                     dtype=info.get('dtype', 'f8'),
                                     fillvalue=info.get('fillvalue', np.nan))
                if 'unit' in info:
                    d.attrs['unit'] = str(info['unit'])

    # If path exists, see what indices are not already done
    with h5py.File(cache_file, 'r') as f:
        i1 = np.all(np.isnan(f['freqs'][:]), axis=1)
        i2 = np.isnan(f['ecc'][:])
        todo_idx = np.where(i1 & i2)[0]

    logger.info(f"{len(todo_idx)} left to process")

    n_batches = min(16 * max(1, pool.size - 1), len(todo_idx))
    tasks = batch_tasks(n_batches=n_batches, arr=todo_idx,
                        args=(galcen, meta, mw, cache_file, id_colname, ids))
    for r in pool.map(worker, tasks, callback=callback):
        pass

    pool.close()
    sys.exit(0)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    from argparse import ArgumentParser
    parser = ArgumentParser(description="")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--nprocs", dest="n_procs", default=1,
                       type=int, help="Number of processes (uses "
                                      "multiprocessing).")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    parser.add_argument("-f", "--file", dest="source_file", required=True)
    parser.add_argument("-o", "--overwrite", dest="overwrite",
                        action="store_true", default=False)
    parser.add_argument("-v", dest="verbose", default=False,
                        action="store_true", help="Verbose mode.")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # deal with multiproc:
    if args.mpi:
        from schwimmbad.mpi import MPIPool
        Pool = MPIPool
        kw = dict()
    elif args.n_procs > 1:
        from schwimmbad import MultiPool
        Pool = MultiPool
        kw = dict(processes=args.n_procs)
    else:
        from schwimmbad import SerialPool
        Pool = SerialPool
        kw = dict()
    Pool = Pool
    Pool_kwargs = kw

    with threadpool_limits(limits=1, user_api='blas'):
        with Pool(**Pool_kwargs) as pool:
            main(pool, args.source_file, overwrite=args.overwrite)
