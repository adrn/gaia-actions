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
import agama
import astropy.coordinates as coord
import astropy.table as at
import astropy.units as u
import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
import h5py
import numpy as np
from astropy.io import fits
from astropy.io.fits.column import FITS2NUMPY
from pyia import GaiaData
from schwimmbad.utils import batch_tasks

agama.setUnits(mass=u.Msun, length=u.kpc, time=u.Myr)
logger = logging.getLogger(__name__)
logging.basicConfig()

act_name = "agama"


def get_c_error_samples(g, N_error_samples, colnames, rng):
    """
    Only works for one star at a time
    """
    C, units = g.get_cov()
    C = C[0]  # only one star at a time

    y = np.full((6, 1), np.nan)
    y[0] = g.ra.to_value(units["ra"])
    y[1] = g.dec.to_value(units["dec"])
    y[3] = g.pmra.to_value(units["pmra"])
    y[4] = g.pmdec.to_value(units["pmdec"])

    if colnames["dist"] == "parallax":
        y[2] = g.parallax.to_value(units["parallax"])
    else:
        dist = getattr(g, colnames["dist"])
        y[2] = dist.value if hasattr(dist, "unit") else dist

        dist_err = getattr(g, colnames["dist_err"])
        C[:, 2] = C[2] = 0.0
        C[2, 2] = dist_err.value**2 if hasattr(dist_err, "unit") else dist_err ** 2

    if colnames["rv"] == "radial_velocity":
        rv = g.radial_velocity
        y[5] = rv.to_value(units["radial_velocity"])
    else:
        rv = getattr(g, colnames["rv"])
        y[5] = rv.value if hasattr(rv, "unit") else rv

        rv_err = getattr(g, colnames["rv_err"])
        C[:, 5] = C[5] = 0.0
        C[5, 5] = rv_err.value**2 if hasattr(rv_err, "unit") else rv_err ** 2

    y = y[:, 0]
    if N_error_samples > 0:
        try:
            samples = rng.multivariate_normal(y, C, size=N_error_samples)
        except np.linalg.LinAlgError:
            logger.error(
                f"Failed to sample from error distribution for source {g.source_id}"
            )
            samples = np.full((N_error_samples, 6), np.nan)
        samples = np.concatenate(([y], samples)).T
    else:
        samples = y[:, None]

    data = dict(
        ra=samples[0] * units["ra"],
        dec=samples[1] * units["dec"],
        pm_ra_cosdec=samples[3] * units["pmra"],
        pm_dec=samples[4] * units["pmdec"],
    )

    if colnames["dist"] == "parallax":
        data["distance"] = coord.Distance(
            parallax=samples[2] * units["parallax"], allow_negative=True
        )
    else:
        if not hasattr(dist, "unit") or dist.unit == u.one or dist.unit is None:
            logger.warning("No distance unit specified in table - assuming kpc")
            dist_unit = u.kpc
        else:
            dist_unit = dist.unit
        data["distance"] = samples[2] * dist_unit
        data["distance"][data["distance"] < 0] = np.nan * dist_unit

    if not hasattr(rv, "unit") or rv.unit == u.one or rv.unit is None:
        logger.warning("No RV unit specified in table - assuming km/s")
        rv_unit = u.km / u.s
    else:
        rv_unit = rv.unit
    data["radial_velocity"] = samples[5] * rv_unit

    for k, v in data.items():
        if hasattr(v, "filled"):
            data[k] = v.filled(np.nan)

    return coord.SkyCoord(**data, frame="icrs")


def worker_agama(task):
    (
        (i, j),
        idx,
        meta,
        source_file,
        gala_potential,
        galcen_frame,
        cache_file,
        colnames,
        rng,
    ) = task

    if len(meta["xyz"]["shape"]) > 2:
        N_error_samples = meta["xyz"]["shape"][1] - 1
    else:
        N_error_samples = 0

    # Convert potential from Gala to Agama
    agama_pot = gala_potential.as_interop("agama")

    # Load source data file:
    g = GaiaData(at.QTable.read(source_file))[idx]

    H = gp.Hamiltonian(gala_potential)

    logger.debug(f"Worker {i}-{j}: running {j-i} tasks now")

    # Set up data containers for this worker:
    all_data = {}
    for k, info in meta.items():
        shape = (len(idx),) + info["shape"][1:]
        all_data[k] = np.full(shape, -1).astype(info.get("dtype", "f8"))

    act_finder = agama.ActionFinder(agama_pot)
    for n, i in enumerate(np.sort(idx)):
        all_data[colnames["id"]][n] = getattr(g, colnames["id"])[n]

        # Always comes back with ndim > 0
        c_n = get_c_error_samples(
            g[n], N_error_samples=N_error_samples, colnames=colnames, rng=rng
        )

        # Returned objects have shape (1, 1 + N_error_samples) - the 0th element values
        # along axis 1 are the catalog values and all other values are error samples
        galcen = c_n.transform_to(galcen_frame)
        w0 = gd.PhaseSpacePosition(galcen.cartesian)

        xv = w0.w(gala_potential.units).T  # shape (N, 6)
        bad_mean = np.any(~np.isfinite(xv[0]))
        good_xv_mask = np.all(np.isfinite(xv), axis=1)
        if bad_mean:
            logger.error(
                f"Failed to compute xyz, vxyz for catalog value for source {i}"
            )
            all_data["flags"][n] += 2**0
            continue
        if not np.any(good_xv_mask):
            logger.error(f"All xyz, vxyz for error samples are bad values {i}")
            all_data["flags"][n] += 2**0
            continue

        all_data["xyz"][n] = galcen.data.xyz.to_value(meta["xyz"]["unit"]).T
        all_data["vxyz"][n] = galcen.velocity.d_xyz.to_value(meta["vxyz"]["unit"]).T
        all_data["flags"][n] = 0

        try:
            act, ang, freq = act_finder(xv, angles=True)
        except Exception as e:
            logger.error(f"Failed to compute actions {i}\n{str(e)}")
            all_data["flags"][n] += 2**1
            continue

        act = act * u.kpc**2 / u.Myr
        freq = freq / u.Myr
        ang = ang * u.rad

        # Note: 4 is a magic number, so is 32 Gyr
        T = min(
            4 * np.max(np.abs(2 * np.pi / freq[good_xv_mask]).to(u.Gyr)), 32 * u.Gyr
        )
        try:
            orbit = H.integrate_orbit(
                w0[good_xv_mask],
                dt=0.5 * u.Myr,
                t1=0 * u.Myr,
                t2=T,
                Integrator=gi.DOPRI853Integrator,
            )
            if len(orbit.shape) < 2:
                orbit = orbit[:, None]
            # orbit = orbit.to_frame(static_frame)
        except Exception as e:
            logger.error(f"Failed to integrate orbit {i+n}\n{str(e)}")
            all_data["flags"][n] += 2**2
            continue

        # Compute actions / frequencies / angles
        reorder = [0, 2, 1]
        all_data["actions"][n] = act[..., reorder].to_value(meta["actions"]["unit"])
        all_data["angles"][n] = ang[..., reorder].to_value(meta["angles"]["unit"])
        all_data["freqs"][n] = freq[..., reorder].to_value(
            meta["freqs"]["unit"], u.dimensionless_angles()
        )

        # L and E
        L = np.full((3, len(c_n)), np.nan)
        E = np.full((len(c_n),), np.nan)
        try:
            L[:, good_xv_mask] = np.mean(
                orbit.angular_momentum().to_value(meta["L"]["unit"]), axis=1
            )
            E[good_xv_mask] = np.mean(orbit.energy().to_value(meta["E"]["unit"]))
        except Exception as e:
            logger.error(f"Failed to compute E Lz for orbit {i+n}\n{e}")
            all_data["flags"][n] += 2**4
        all_data["L"][n] = np.squeeze(L.T)
        all_data["E"][n] = np.squeeze(E)

        # Other various things:
        zmax = np.full((len(c_n),), np.nan)
        rper = np.full((len(c_n),), np.nan)
        rapo = np.full((len(c_n),), np.nan)
        ecc = np.full((len(c_n),), np.nan)
        try:
            all_data["R_guide"][n] = w0.guiding_radius(gala_potential).to_value(
                meta["R_guide"]["unit"]
            )

            zmax[good_xv_mask] = orbit.zmax(approximate=True).to_value(
                meta["z_max"]["unit"]
            )
            rper[good_xv_mask] = orbit.pericenter(approximate=True).to_value(
                meta["r_per"]["unit"]
            )
            rapo[good_xv_mask] = orbit.apocenter(approximate=True).to_value(
                meta["r_apo"]["unit"]
            )
            ecc = (rapo - rper) / (rapo + rper)
        except Exception as e:
            logger.error(f"Failed to compute zmax peri apo for orbit {i+n}\n{e}")
            all_data["flags"][n] += 2**3

        all_data["z_max"][n] = np.squeeze(zmax)
        all_data["r_per"][n] = np.squeeze(rper)
        all_data["r_apo"][n] = np.squeeze(rapo)
        all_data["ecc"][n] = np.squeeze(ecc)

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
    dist_err_colname=None,
    rv_colname=None,
    rv_err_colname=None,
    galcen_filename=None,
    N_error_samples=0,
    seed=None,
):
    logger.debug(f"Starting file {source_file}...")

    cache_path = pathlib.Path(__file__).parent / "../cache"
    cache_path = cache_path.resolve()
    cache_path.mkdir(exist_ok=True)

    source_file = pathlib.Path(source_file).resolve()
    if not source_file.exists():
        raise IOError(f"Source data file {source_file} does not exist")

    # Global parameters
    gala_potential = gp.MilkyWayPotential2022()
    potential_name = "MilkyWayPotential2022"

    if galcen_filename is None:
        # See: Hunt, Price-Whelan et al. 2022
        galcen_frame = coord.Galactocentric(
            galcen_distance=8.275 * u.kpc, galcen_v_sun=[8.4, 251.8, 8.4] * u.km / u.s
        )
    else:
        with open(galcen_filename, "rb") as f:
            galcen_frame = pickle.load(f)

    source_name, source_ext, *_ = source_file.name.split(".")
    cache_file = cache_path / f"{source_name}-{potential_name}-{act_name}.hdf5"
    logger.debug(f"Writing to cache file {cache_file}".format(cache_file))

    galcen_cache_file = (
        cache_path / f"{source_name}-{potential_name}-{act_name}.galcen.pkl"
    )
    with open(galcen_cache_file, "wb") as f:
        pickle.dump(galcen_frame, f)

    if id_colname is None:  # assumes gaia
        id_colname = "source_id"

    if dist_colname is None:  # assumes gaia
        dist_colname = "parallax"

    if rv_colname is None:  # assumes gaia
        rv_colname = "radial_velocity"
        rv_err_colname = "radial_velocity_error"

    # Column names for things that may have been provided by other surveys:
    colnames = {
        "id": id_colname,
        "rv": rv_colname,
        "rv_err": rv_err_colname,
        "dist": dist_colname,
        "dist_err": dist_err_colname,
    }

    # Count the number of sources in the source data table:
    if source_ext.lower() == "fits":
        # We don't actually have to load the data - we can use the header
        hdr = fits.getheader(source_file, ext=1)
        Nstars = hdr["NAXIS2"]
        for k, v in hdr.items():
            if k.startswith("TTYPE"):
                if v.strip() == id_colname:
                    tform = hdr[f"TFORM{k[5:]}"].strip()
        id_dtype = FITS2NUMPY[tform]
    else:
        tbl = at.Table.read(source_file)
        Nstars = len(tbl)
        id_dtype = tbl[id_colname].dtype
        del tbl

    # Column metadata: map names to shapes
    if N_error_samples > 0:
        base_shape = (Nstars, 1 + N_error_samples)
    else:
        base_shape = (Nstars,)

    meta = {
        id_colname: {
            "shape": (Nstars,),
            "dtype": id_dtype,
            "fillvalue": None,
        },
        "xyz": {"shape": base_shape + (3,), "unit": u.kpc},
        "vxyz": {"shape": base_shape + (3,), "unit": u.km / u.s},
        # Frequencies, actions, and angles:
        "freqs": {"shape": base_shape + (3,), "unit": u.rad / u.Gyr},
        "actions": {"shape": base_shape + (3,), "unit": u.kpc * u.km / u.s},
        "angles": {"shape": base_shape + (3,), "unit": u.rad},
        # Orbit parameters:
        "R_guide": {"shape": base_shape, "unit": u.kpc},
        "z_max": {"shape": base_shape, "unit": u.kpc},
        "r_per": {"shape": base_shape, "unit": u.kpc},
        "r_apo": {"shape": base_shape, "unit": u.kpc},
        "ecc": {"shape": base_shape, "unit": u.one},
        "L": {"shape": base_shape + (3,), "unit": u.kpc * u.km / u.s},
        "E": {"shape": base_shape, "unit": (u.km / u.s) ** 2},
        "flags": {"shape": base_shape, "dtype": "i4", "fillvalue": -1},
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
        flags = f["flags"][:]
        if flags.ndim > 1:
            todo_idx = np.where(np.any(flags == -1, axis=1))[0]
        else:
            todo_idx = np.where(flags == -1)[0]

    logger.info(f"{len(todo_idx)} sources left to process")

    # Random number generator:
    parent_rng = np.random.default_rng(seed)

    n_batches = min(8 * max(1, pool.size - 1), len(todo_idx))
    tasks = batch_tasks(
        n_batches=n_batches,
        arr=todo_idx,
        args=(meta, source_file, gala_potential, galcen_frame, cache_file, colnames),
    )
    rngs = parent_rng.spawn(len(tasks))
    tasks = [tuple(t) + (rng,) for t, rng in zip(tasks, rngs)]

    for r in pool.map(worker_agama, tasks, callback=callback):
        pass

    pool.close()
    sys.exit(0)


if __name__ == "__main__":
    from argparse import ArgumentParser

    from threadpoolctl import threadpool_limits

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
    parser.add_argument("--dist-err-col", dest="dist_err_colname", default=None)
    parser.add_argument("--rv-col", dest="rv_colname", default=None)
    parser.add_argument("--rv-err-col", dest="rv_err_colname", default=None)

    parser.add_argument(
        "--n-error-samples", dest="N_error_samples", default=0, type=int
    )

    parser.add_argument("--seed", dest="seed", default=None, type=int)

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
                dist_err_colname=args.dist_err_colname,
                rv_colname=args.rv_colname,
                rv_err_colname=args.rv_err_colname,
                galcen_filename=args.galcen_filename,
                N_error_samples=args.N_error_samples,
                seed=args.seed,
            )
