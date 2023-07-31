# Standard library
import os

os.environ["OMP_NUM_THREADS"] = "1"
import logging
import pathlib
import pickle
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

import agama

agama.setUnits(mass=u.Msun, length=u.kpc, time=u.Myr)
logger = logging.getLogger(__name__)
logging.basicConfig()


def worker_agama(task):
    (
        (i, j),
        idx,
        galcen,
        meta,
        gala_potential,
        frame,
        cache_file,
        id_colname,
        ids,
    ) = task
    ids = ids[idx]
    galcen = galcen[idx]

    # Convert to Agama potential:
    agama_components = []
    for p in gala_potential["disk"].get_three_potentials().values():
        agama_components.append(
            dict(
                type="miyamotonagai",
                mass=p.parameters["m"].value,
                scaleradius=p.parameters["a"].value,
                scaleheight=p.parameters["b"].value,
            )
        )

    for k in ["bulge", "nucleus"]:
        p = gala_potential[k]
        agama_components.append(
            dict(
                type="dehnen",
                mass=p.parameters["m"].value,
                scaleradius=p.parameters["c"].value,
                gamma=1.0,
            )
        )

    p = gala_potential["halo"]
    agama_components.append(
        dict(
            type="nfw",
            mass=p.parameters["m"].value,
            scaleradius=p.parameters["r_s"].value,
        )
    )
    agama_pot = agama.Potential(*agama_components)

    w0 = gd.PhaseSpacePosition(galcen.cartesian)
    H = gp.Hamiltonian(gala_potential, frame)
    static_frame = gp.StaticFrame(H.units)

    logger.debug(f"Worker {i}-{j}: running {j-i} tasks now")

    # Set up data containers:
    all_data = {}
    for k, info in meta.items():
        if k == id_colname:
            all_data[k] = ids
        else:
            shape = (len(ids),) + info["shape"][1:]
            all_data[k] = np.full(shape, np.nan)

    act_finder = agama.ActionFinder(agama_pot)
    for n in range(len(galcen)):
        all_data[id_colname][n] = ids[n]
        all_data["xyz"][n] = galcen.data.xyz[:, n].to_value(meta["xyz"]["unit"])
        all_data["vxyz"][n] = galcen.velocity.d_xyz[:, n].to_value(meta["vxyz"]["unit"])

        xv = np.squeeze(w0[n].w(gala_potential.units))
        try:
            act, ang, freq = act_finder(xv, angles=True)
        except Exception as e:
            logger.error(f"Failed to compute actions {i}\n{str(e)}")
            continue

        act = act * u.kpc**2 / u.Myr
        freq = freq * u.Myr
        ang = ang * u.rad

        T = 4 * np.abs(2 * np.pi / freq.min()).to(u.Gyr)
        try:
            orbit = H.integrate_orbit(
                w0[n],
                dt=0.5 * u.Myr,
                t1=0 * u.Myr,
                t2=T,
                Integrator=gi.DOPRI853Integrator,
            )
            orbit = orbit.to_frame(static_frame)
        except Exception as e:
            logger.error(f"Failed to integrate orbit {i+n}\n{str(e)}")
            continue

        # Compute actions / frequencies / angles
        idx = [0, 2, 1]
        all_data["actions"][n] = act[idx].to_value(meta["actions"]["unit"])
        all_data["angles"][n] = ang[idx].to_value(meta["angles"]["unit"])
        all_data["freqs"][n] = freq[idx].to_value(
            meta["freqs"]["unit"], u.dimensionless_angles()
        )

        # Other various things:
        try:
            rper = orbit.pericenter(approximate=True).to_value(meta["r_per"]["unit"])
            rapo = orbit.apocenter(approximate=True).to_value(meta["r_apo"]["unit"])

            all_data["z_max"][n] = orbit.zmax(approximate=True).to_value(
                meta["z_max"]["unit"]
            )
            all_data["r_per"][n] = rper
            all_data["r_apo"][n] = rapo
            all_data["ecc"][n] = (rapo - rper) / (rapo + rper)
        except Exception as e:
            logger.error(f"Failed to compute zmax peri apo for orbit {i+n}\n{e}")

        # Lz and E
        try:
            all_data["L"][n] = np.mean(
                orbit.angular_momentum().to_value(meta["L"]["unit"]), axis=1
            )
            all_data["E"][n] = np.mean(orbit.energy().to_value(meta["E"]["unit"]))
        except Exception as e:
            logger.error(f"Failed to compute E Lz for orbit {i+n}\n{e}")

    return idx, cache_file, all_data


def callback(res):
    idx, cache_file, all_data = res

    logger.debug(f"Writing block {idx[0]}-{idx[-1]} to cache file")
    with h5py.File(cache_file, "r+") as f:
        for k in all_data:
            f[k][idx] = all_data[k]


def main(
    pool,
    source_file,
    overwrite=False,
    id_colname=None,
    dist_colname=None,
    rv_colname=None,
    galcen_filename=None,
):
    logger.debug(f"Starting file {source_file}...")

    cache_path = pathlib.Path(__file__).parent / "../cache"
    cache_path = cache_path.resolve()
    cache_path.mkdir(exist_ok=True)

    source_file = pathlib.Path(source_file).resolve()

    # Global parameters

    gala_pot = gp.MilkyWayPotential2022()
    potential_name = "MilkyWayPotential2022"
    act_name = "agama"

    H = gp.Hamiltonian(gala_pot)

    if galcen_filename is None:
        gc_frame = coord.Galactocentric(
            galcen_distance=8.275 * u.kpc, galcen_v_sun=[8.4, 251.8, 8.4] * u.km / u.s
        )
    else:
        with open(galcen_filename, "rb") as f:
            gc_frame = pickle.load(f)

    source_name = source_file.name.split(".")[0]
    cache_file = cache_path / f"{source_name}-{potential_name}-{act_name}.hdf5"
    logger.debug(f"Writing to cache file {cache_file}".format(cache_file))

    galcen_cache_file = (
        cache_path / f"{source_name}-{potential_name}-{act_name}.galcen.pkl"
    )
    with open(galcen_cache_file, "wb") as f:
        pickle.dump(gc_frame, f)

    # Load the source data table:
    g = GaiaData(at.QTable.read(source_file))

    mask = np.ones(len(g), dtype=bool)
    if id_colname is None:  # assumes gaia
        id_colname = "source_id"
        ids = g.source_id.astype("i8")
    else:
        ids = g.data[id_colname]

    if dist_colname is None:  # assumes gaia
        dist = coord.Distance(parallax=g.parallax, allow_negative=True)
        mask &= np.isfinite(dist)
    else:
        dist = g.data[dist_colname]
    mask &= dist > 0

    if rv_colname is None:  # assumes gaia
        if hasattr(g, "radial_velocity"):
            rv = g.radial_velocity
        elif hasattr(g, "dr2_radial_velocity"):
            rv = g.dr2_radial_velocity
        else:
            raise ValueError("Invalid radial velocity column or dataset")
        mask &= np.isfinite(rv)
    else:
        rv = g.data[rv_colname]

    if not hasattr(dist, "unit") or dist.unit == u.one or dist.unit is None:
        logger.warning("No distance unit specified in table - assuming kpc")
        dist = dist * u.kpc

    if not hasattr(rv, "unit") or rv.unit == u.one or rv.unit is None:
        logger.warning("No RV unit specified in table - assuming km/s")
        rv = rv * u.km / u.s

    # Get coordinates, and only keep good values:
    if ~np.all(mask):
        logger.warning(f"Filtering {mask.sum()} bad distance or RV values")

    c = g[mask].get_skycoord(
        distance=u.Quantity(dist[mask]), radial_velocity=u.Quantity(rv[mask])
    )
    ids = ids[mask]

    galcen = c.transform_to(gc_frame)
    logger.debug("Data loaded...")

    Nstars = len(c)

    # Column metadata: map names to shapes
    meta = {
        id_colname: {
            "shape": (Nstars,),
            "dtype": g.data[id_colname].dtype,
            "fillvalue": None,
        },
        "xyz": {"shape": (Nstars, 3), "unit": u.kpc},
        "vxyz": {"shape": (Nstars, 3), "unit": u.km / u.s},
        # Frequencies, actions, and angles computed with Sanders & Binney
        "freqs": {"shape": (Nstars, 3), "unit": u.rad / u.Gyr},
        "actions": {"shape": (Nstars, 3), "unit": u.kpc * u.km / u.s},
        "angles": {"shape": (Nstars, 3), "unit": u.rad},
        # Orbit parameters:
        "R_guide": {"shape": (Nstars,), "unit": u.kpc},
        "z_max": {"shape": (Nstars,), "unit": u.kpc},
        "r_per": {"shape": (Nstars,), "unit": u.kpc},
        "r_apo": {"shape": (Nstars,), "unit": u.kpc},
        "ecc": {"shape": (Nstars,), "unit": u.one},
        "L": {"shape": (Nstars, 3), "unit": u.kpc * u.km / u.s},
        "E": {"shape": (Nstars,), "unit": (u.km / u.s) ** 2},
    }

    # Make sure output file exists
    if not cache_file.exists() or overwrite:
        with h5py.File(cache_file, "w") as f:
            for name, info in meta.items():
                d = f.create_dataset(
                    name,
                    shape=info["shape"],
                    dtype=info.get("dtype", "f8"),
                    fillvalue=info.get("fillvalue", np.nan),
                )
                if "unit" in info:
                    d.attrs["unit"] = str(info["unit"])

    # If path exists, see what indices are not already done
    with h5py.File(cache_file, "r") as f:
        i1 = np.all(np.isnan(f["freqs"][:]), axis=1)
        i2 = np.isnan(f["ecc"][:])
        todo_idx = np.where(i1 & i2)[0]

    logger.info(f"{len(todo_idx)} left to process")

    n_batches = min(16 * max(1, pool.size - 1), len(todo_idx))
    tasks = batch_tasks(
        n_batches=n_batches,
        arr=todo_idx,
        args=(galcen, meta, H.potential, H.frame, cache_file, id_colname, ids),
    )
    for r in pool.map(worker_agama, tasks, callback=callback):
        pass

    pool.close()
    sys.exit(0)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    from argparse import ArgumentParser

    parser = ArgumentParser(description="")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--nprocs",
        dest="n_procs",
        default=1,
        type=int,
        help="Number of processes (uses " "multiprocessing).",
    )
    group.add_argument(
        "--mpi",
        dest="mpi",
        default=False,
        action="store_true",
        help="Run with MPI.",
    )

    parser.add_argument("-f", "--file", dest="source_file", required=True)
    parser.add_argument(
        "-o",
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-v",
        dest="verbose",
        default=False,
        action="store_true",
        help="Verbose mode.",
    )

    parser.add_argument("--id-col", dest="id_colname", default=None)
    parser.add_argument("--dist-col", dest="dist_colname", default=None)
    parser.add_argument("--rv-col", dest="rv_colname", default=None)

    parser.add_argument("-g", "--galcen", dest="galcen_filename", default=None)

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

    with threadpool_limits(limits=1, user_api="blas"):
        with Pool(**Pool_kwargs) as pool:
            main(
                pool,
                args.source_file,
                overwrite=args.overwrite,
                id_colname=args.id_colname,
                dist_colname=args.dist_colname,
                rv_colname=args.rv_colname,
                galcen_filename=args.galcen_filename,
            )