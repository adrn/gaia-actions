import json
import logging
import pathlib

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

agama.setUnits(mass=1.0, length=1.0, time=1.0)  # Msun, kpc, Myr
logger = logging.getLogger(__name__)
logging.basicConfig()


def batch_worker(
    source_data_idx,
    data_model,
    source_file,
    gala_potential,
    galcen_frame,
    catalog_cache_file,
    samples_cache_file,
    colnames,
    rng,
):
    if len(data_model["xyz"]["shape"]) > 2:
        N_error_samples = data_model["xyz"]["shape"][1] - 1
    else:
        N_error_samples = 0

    source_data_idx = np.sort(source_data_idx)

    # Convert potential from Gala to Agama
    agama_pot = gala_potential.as_interop("agama")

    # Load source data file:
    gaiadata_kw = {
        "distance_colname": colnames["dist"],
        "distance_error_colname": colnames["dist_err"],
        "radial_velocity_colname": colnames["rv"],
        "radial_velocity_error_colname": colnames["rv_err"],
    }
    g = GaiaData(at.QTable.read(source_file), **gaiadata_kw)[source_data_idx]

    H = gp.Hamiltonian(gala_potential)

    logger.debug(f"Worker running {len(source_data_idx)} tasks now")

    # Set up data containers for this worker:
    batch_data = {}
    for k, info in data_model.items():
        shape = (len(source_data_idx),) + info["shape"][1:]
        fill_val = info.get("fillvalue", np.nan)
        batch_data[k] = np.full(shape, fill_val if fill_val is not None else -1).astype(
            info.get("dtype", "f8")
        )

    act_finder = agama.ActionFinder(agama_pot)
    for n, i in enumerate(source_data_idx):
        batch_data[colnames["id"]][n] = getattr(g, colnames["id"])[n]

        if N_error_samples > 0:
            g_samples = g[n].get_error_samples(size=N_error_samples, rng=rng)

            # TODO: workaround because distance errors can be big and lead to negative
            # distance values that aren't being handled, for some reason, in
            # get_distance()
            _dist = getattr(g_samples, g_samples.distance_colname)
            _dist[_dist < 0] = np.nan
            setattr(g_samples, g_samples.distance_colname, _dist)

            c_mean = g[n].get_skycoord()
            c_n = g_samples.get_skycoord()
            c_n = coord.concatenate((c_mean, c_n[0]))
        else:
            c_n = g[n].get_skycoord()

        # Returned objects have shape (1, 1 + N_error_samples) - the 0th element values
        # along axis 1 are the catalog values and all other values are error samples
        galcen = c_n.transform_to(galcen_frame)
        w0 = gd.PhaseSpacePosition(galcen.cartesian)

        xv = w0.w(gala_potential.units).T  # shape (N, 6)
        bad_mean = np.any(~np.isfinite(xv[0]))
        good_xv_mask = np.all(np.isfinite(xv), axis=1)
        if bad_mean:
            logger.debug(
                f"Failed to compute xyz, vxyz for catalog value for source {i}"
            )
            batch_data["flags"][n] = 2**0
            continue
        if not np.any(good_xv_mask):
            logger.debug(f"All xyz, vxyz for error samples are bad values {i}")
            batch_data["flags"][n] = 2**0
            continue

        batch_data["xyz"][n] = galcen.data.xyz.to_value(data_model["xyz"]["unit"]).T
        batch_data["vxyz"][n] = galcen.velocity.d_xyz.to_value(
            data_model["vxyz"]["unit"]
        ).T
        batch_data["flags"][n] = 0

        try:
            act, ang, freq = act_finder(xv, angles=True)
        except Exception as e:
            logger.debug(f"Failed to compute actions {i}\n{str(e)}")
            batch_data["flags"][n] = 2**1
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
                Integrator=gi.LeapfrogIntegrator,
            )
            if len(orbit.shape) < 2:
                orbit = orbit[:, None]
            # orbit = orbit.to_frame(static_frame)
        except Exception as e:
            logger.debug(f"Failed to integrate orbit {i + n}\n{str(e)}")
            batch_data["flags"][n] = 2**2
            continue

        # Compute actions / frequencies / angles
        reorder = [0, 2, 1]
        batch_data["actions"][n] = act[..., reorder].to_value(
            data_model["actions"]["unit"]
        )
        batch_data["angles"][n] = ang[..., reorder].to_value(
            data_model["angles"]["unit"]
        )
        batch_data["freqs"][n] = freq[..., reorder].to_value(
            data_model["freqs"]["unit"], u.dimensionless_angles()
        )

        # L and E
        L = np.full((3, len(c_n)), np.nan)
        E = np.full((len(c_n),), np.nan)
        try:
            L[:, good_xv_mask] = np.mean(
                orbit.angular_momentum().to_value(data_model["L"]["unit"]), axis=1
            )
            E[good_xv_mask] = np.mean(
                orbit.energy().to_value(data_model["E"]["unit"]), axis=0
            )
        except Exception as e:
            logger.debug(f"Failed to compute E Lz for orbit {i + n}\n{e}")
            batch_data["flags"][n] = 2**4
        batch_data["L"][n] = np.squeeze(L.T)
        batch_data["E"][n] = np.squeeze(E)

        # Other various things:
        zmax = np.full((len(c_n),), np.nan)
        rper = np.full((len(c_n),), np.nan)
        rapo = np.full((len(c_n),), np.nan)
        ecc = np.full((len(c_n),), np.nan)
        try:
            batch_data["R_guide"][n] = np.squeeze(
                w0.guiding_radius(gala_potential).to_value(
                    data_model["R_guide"]["unit"]
                )
            )

            zmax[good_xv_mask] = orbit.zmax(approximate=True).to_value(
                data_model["z_max"]["unit"]
            )
            rper[good_xv_mask] = orbit.pericenter(approximate=True).to_value(
                data_model["r_per"]["unit"]
            )
            rapo[good_xv_mask] = orbit.apocenter(approximate=True).to_value(
                data_model["r_apo"]["unit"]
            )
            ecc = (rapo - rper) / (rapo + rper)
        except Exception as e:
            logger.debug(f"Failed to compute zmax peri apo for orbit {i + n}\n{e}")
            batch_data["flags"][n] += 2**3

        batch_data["z_max"][n] = np.squeeze(zmax)
        batch_data["r_per"][n] = np.squeeze(rper)
        batch_data["r_apo"][n] = np.squeeze(rapo)
        batch_data["ecc"][n] = np.squeeze(ecc)

    return {
        "data": batch_data,
        "catalog_cache_file": catalog_cache_file,
        "samples_cache_file": samples_cache_file,
        "idx": source_data_idx,
    }


def batch_callback(future):
    if hasattr(future, "result"):
        result = future.result()
    else:
        result = future

    idx = result["idx"]
    data = result["data"]
    col_order = list(data.keys())
    has_samples = result["samples_cache_file"] is not None

    logger.debug(f"Writing block {idx[0]}-{idx[-1]} to cache file(s)")

    # Write catalog values
    with h5py.File(result["catalog_cache_file"], "r+") as f:
        f.attrs["col_order"] = col_order
        for k in data:
            if has_samples and data[k].ndim >= 2:
                # Take catalog value (index 0 along sample axis)
                f[k][idx] = data[k][:, 0]
            else:
                f[k][idx] = data[k]

    # Write error samples
    if has_samples:
        with h5py.File(result["samples_cache_file"], "r+") as f:
            f.attrs["col_order"] = col_order
            for k in data:
                if data[k].ndim >= 2:
                    # Take error samples (indices 1: along sample axis)
                    f[k][idx] = data[k][:, 1:]
                else:
                    # Scalar columns (e.g. id) — same in both files
                    f[k][idx] = data[k]


def prepare_data_and_cache(
    source_file,
    overwrite=False,
    id_colname=None,
    dist_colname=None,
    dist_err_colname=None,
    rv_colname=None,
    rv_err_colname=None,
    N_error_samples=0,
):
    logger.debug(f"Starting file {source_file!s}...")

    cache_path = pathlib.Path(__file__).parent / "../cache"
    cache_path = cache_path.resolve()
    cache_path.mkdir(exist_ok=True)

    source_file = pathlib.Path(source_file).resolve()
    if not source_file.exists():
        raise IOError(f"Source data file {source_file!s} does not exist")

    source_name, source_ext, *_ = source_file.name.split(".")

    if id_colname is None:  # NOTE: assumes gaia
        id_colname = "source_id"

    dist_colname = "parallax" if dist_colname is None else dist_colname
    dist_err_colname = (
        "parallax_error" if dist_err_colname is None else dist_err_colname
    )

    rv_colname = "radial_velocity" if rv_colname is None else rv_colname
    rv_err_colname = (
        "radial_velocity_error" if rv_err_colname is None else rv_err_colname
    )

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
    # The internal data model used by batch_worker always includes the sample
    # dimension when N_error_samples > 0 (catalog value + samples together).
    # Two separate output files are created: one for catalog values (scalar per
    # star) and one for error samples (array-valued per star).
    if N_error_samples > 0:
        internal_base_shape = (Nstars, 1 + N_error_samples)
        catalog_base_shape = (Nstars,)
        samples_base_shape = (Nstars, N_error_samples)
    else:
        internal_base_shape = (Nstars,)
        catalog_base_shape = (Nstars,)
        samples_base_shape = None

    def _make_data_model(base_shape):
        return {
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

    # Internal model used by batch_worker (combined catalog + samples)
    data_model = _make_data_model(internal_base_shape)
    logger.debug(f"Data model: {data_model!s}")

    # Catalog output file
    catalog_cache_file = cache_path / f"{source_name}-actions.hdf5"
    catalog_data_model = _make_data_model(catalog_base_shape)

    # Samples output file (only when error samples are requested)
    if N_error_samples > 0:
        samples_cache_file = cache_path / f"{source_name}-actions-err.hdf5"
        samples_data_model = _make_data_model(samples_base_shape)
    else:
        samples_cache_file = None
        samples_data_model = None

    def _init_cache_file(path, model):
        with h5py.File(path, "w") as f:
            for name, info in model.items():
                d = f.create_dataset(
                    name,
                    shape=info["shape"],
                    dtype=info.get("dtype", "f8"),
                    fillvalue=info.get("fillvalue", np.nan),
                )
                if "unit" in info:
                    d.attrs["unit"] = str(info["unit"])

    # Make sure output files exist
    if not catalog_cache_file.exists() or overwrite:
        _init_cache_file(catalog_cache_file, catalog_data_model)
    logger.debug(f"Catalog cache file at {catalog_cache_file!s}")

    if samples_cache_file is not None:
        if not samples_cache_file.exists() or overwrite:
            _init_cache_file(samples_cache_file, samples_data_model)
        logger.debug(f"Samples cache file at {samples_cache_file!s}")

    # If path exists, see what indices are not already done
    with h5py.File(catalog_cache_file, "r") as f:
        flags = f["flags"][:]
        todo_idx = np.where(flags == -1)[0]

    logger.info(f"{len(todo_idx)} sources left to process")

    return {
        "source_file": source_file,
        "catalog_cache_file": catalog_cache_file,
        "samples_cache_file": samples_cache_file,
        "todo_idx": todo_idx,
        "data_model": data_model,
        "colnames": colnames,
    }


def run_batches(
    source_file,
    catalog_cache_file,
    samples_cache_file,
    todo_idx,
    data_model,
    colnames,
    seed=None,
    pool=None,
):
    # TODO: allow user to customize these?
    gala_potential = gp.MilkyWayPotential(version="v2")
    galcen_frame = coord.Galactocentric(
        galcen_distance=8.275 * u.kpc, galcen_v_sun=[8.4, 251.8, 8.4] * u.km / u.s
    )

    # Write metadata to both output files
    meta_attrs = {
        "source_file": str(source_file),
        "action_method": "agama.ActionFinder",
        "gala_potential": json.dumps(gp.io.to_dict(gala_potential)),
        "galcen_distance_kpc": galcen_frame.galcen_distance.to_value(u.kpc),
        "galcen_v_sun_kms": galcen_frame.galcen_v_sun.d_xyz.to_value(u.km / u.s),
    }
    for cache_file in [catalog_cache_file, samples_cache_file]:
        if cache_file is None:
            continue
        with h5py.File(cache_file, "r+") as f:
            for attr_k, attr_v in meta_attrs.items():
                f.attrs[attr_k] = attr_v

    parent_rng = np.random.default_rng(seed)

    pool_size = 1 if pool is None else pool.num_workers

    batch_size = max(len(todo_idx) // pool_size, 1)
    n_batches = int(np.ceil(len(todo_idx) / batch_size))
    rngs = parent_rng.spawn(n_batches)

    for i in range(n_batches):
        logger.debug(
            f"Submitting batch {i} of {n_batches - 1} with {batch_size} sources"
        )
        source_data_idx = todo_idx[i * batch_size : (i + 1) * batch_size]
        if len(source_data_idx) == 0:
            break

        worker_kw = dict(
            data_model=data_model,
            source_file=source_file,
            gala_potential=gala_potential,
            galcen_frame=galcen_frame,
            catalog_cache_file=catalog_cache_file,
            samples_cache_file=samples_cache_file,
            colnames=colnames,
            rng=rngs[i],
        )

        if pool is not None:
            result = pool.submit(
                batch_worker,
                source_data_idx,
                **worker_kw,
            )
            result.add_done_callback(batch_callback)
        else:
            result = batch_worker(
                source_data_idx,
                **worker_kw,
            )
            batch_callback(result)


if __name__ == "__main__":
    from argparse import ArgumentParser

    from threadpoolctl import threadpool_limits

    parser = ArgumentParser(description="")

    parser.add_argument(
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

    # Data column names:
    parser.add_argument("--id-col", dest="id_colname", default=None)
    parser.add_argument("--dist-col", dest="dist_colname", default=None)
    parser.add_argument("--dist-err-col", dest="dist_err_colname", default=None)
    parser.add_argument("--rv-col", dest="rv_colname", default=None)
    parser.add_argument("--rv-err-col", dest="rv_err_colname", default=None)

    # Controls whether to generate error samples:
    parser.add_argument(
        "--n-error-samples", dest="N_error_samples", default=0, type=int
    )

    # Random number seed:
    parser.add_argument("--seed", dest="seed", default=None, type=int)

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if args.mpi:
        from mpi4py import MPI
        from mpi4py.futures import MPIPoolExecutor

        logger.info(f"Running with MPI: with {MPI.COMM_WORLD.Get_size()} workers ")

        config = prepare_data_and_cache(
            args.source_file,
            overwrite=args.overwrite,
            id_colname=args.id_colname,
            dist_colname=args.dist_colname,
            dist_err_colname=args.dist_err_colname,
            rv_colname=args.rv_colname,
            rv_err_colname=args.rv_err_colname,
            N_error_samples=args.N_error_samples,
        )

        with threadpool_limits(limits=1, user_api="blas"):
            with MPIPoolExecutor() as pool:
                run_batches(**config, seed=args.seed, pool=pool)

    else:
        logger.info("Running in serial mode")
        config = prepare_data_and_cache(
            args.source_file,
            overwrite=args.overwrite,
            id_colname=args.id_colname,
            dist_colname=args.dist_colname,
            dist_err_colname=args.dist_err_colname,
            rv_colname=args.rv_colname,
            rv_err_colname=args.rv_err_colname,
            N_error_samples=args.N_error_samples,
        )

        run_batches(**config, seed=args.seed)
