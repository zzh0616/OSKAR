"""Offline end-to-end CPU astrometry checks. No production data or installation.

Reference calculations share ERFA, but form visibility directly from ITRS
baselines, not OSKAR UVW or its Jones/beam routines. This tests implementation
consistency, not an independent absolute astrometric accuracy bound.
"""

import argparse
import ctypes as ct
import hashlib
import json
import os
import subprocess
from pathlib import Path

import erfa
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.utils import iers


class Reader:
    def __init__(self, library):
        self.lib = ct.CDLL(str(library))
        p, i, d, s = ct.c_void_p, ct.c_int, ct.c_double, ct.POINTER(ct.c_int)
        self.bind("oskar_binary_create", p, [ct.c_char_p, ct.c_char, s])
        self.bind("oskar_binary_free", None, [p])
        self.bind("oskar_vis_header_read", p, [p, s])
        self.bind("oskar_vis_header_free", None, [p, s])
        for name in (
            "num_blocks",
            "num_stations",
            "num_channels_total",
            "num_times_total",
            "astrometry_frame",
        ):
            self.bind("oskar_vis_header_" + name, i, [p])
        self.bind("oskar_vis_header_astrometry_eop", ct.c_char_p, [p])
        self.bind("oskar_vis_header_station_offset_ecef_metres", p, [p, i])
        self.bind("oskar_vis_block_create_from_header", p, [i, p, s])
        self.bind("oskar_vis_block_read", None, [p, p, p, i, s])
        self.bind("oskar_vis_block_free", None, [p, s])
        self.bind("oskar_vis_block_station_to_baseline_coords", None, [p, p, s])
        for name in (
            "num_times",
            "num_channels",
            "start_time_index",
            "start_channel_index",
        ):
            self.bind("oskar_vis_block_" + name, i, [p])
        self.bind("oskar_vis_block_baseline_uvw_metres", p, [p, i])
        self.bind("oskar_vis_block_cross_correlations", p, [p])
        self.bind("oskar_mem_double", ct.POINTER(d), [p, s])
        self.bind("oskar_mem_float", ct.POINTER(ct.c_float), [p, s])
        self.bind("oskar_mem_void", p, [p])
        self.bind("oskar_mem_length", ct.c_size_t, [p])
        self.bind("oskar_mem_type", i, [p])

    def bind(self, name, result, args):
        f = getattr(self.lib, name)
        f.restype, f.argtypes = result, args

    def array(self, mem, count, complex_data=False):
        kind = self.lib.oskar_mem_type(mem)
        dtype = np.float64 if kind & 0x08 else np.float32
        size = np.dtype(dtype).itemsize
        raw = (ct.c_char * (size * count)).from_address(self.lib.oskar_mem_void(mem))
        a = np.frombuffer(raw, dtype=dtype).copy()
        if complex_data:
            a = a.view(np.complex128 if dtype == np.float64 else np.complex64)
        return a

    def read(self, filename, full):
        lib, status = self.lib, ct.c_int(0)
        stream = lib.oskar_binary_create(os.fsencode(filename), b"r", ct.byref(status))
        header = lib.oskar_vis_header_read(stream, ct.byref(status))
        block = None
        try:
            if status.value:
                raise RuntimeError(f"header status {status.value}")
            ns = lib.oskar_vis_header_num_stations(header)
            nt = lib.oskar_vis_header_num_times_total(header)
            nc = lib.oskar_vis_header_num_channels_total(header)
            nb = ns * (ns - 1) // 2
            npol = 4 if full else 1
            xyz = np.column_stack(
                [self.array(lib.oskar_vis_header_station_offset_ecef_metres(header, k), ns) for k in range(3)]
            )
            data = np.empty((nt, nc, nb, npol), dtype=complex)
            uvw = np.empty((nt, nb, 3))
            block = lib.oskar_vis_block_create_from_header(0, header, ct.byref(status))
            for index in range(lib.oskar_vis_header_num_blocks(header)):
                lib.oskar_vis_block_read(block, header, stream, index, ct.byref(status))
                lib.oskar_vis_block_station_to_baseline_coords(block, header, ct.byref(status))
                t0 = lib.oskar_vis_block_start_time_index(block)
                f0 = lib.oskar_vis_block_start_channel_index(block)
                ntime = lib.oskar_vis_block_num_times(block)
                nfreq = lib.oskar_vis_block_num_channels(block)
                if status.value:
                    raise RuntimeError(f"block status {status.value}")
                raw = self.array(
                    lib.oskar_vis_block_cross_correlations(block),
                    ntime * nfreq * nb * npol * 2,
                    True,
                )
                data[t0 : t0 + ntime, f0 : f0 + nfreq] = raw.reshape(ntime, nfreq, nb, npol)
                for k in range(3):
                    uvw[t0 : t0 + ntime, :, k] = self.array(
                        lib.oskar_vis_block_baseline_uvw_metres(block, k), ntime * nb
                    ).reshape(ntime, nb)
            frame = lib.oskar_vis_header_astrometry_frame(header)
            snapshot = lib.oskar_vis_header_astrometry_eop(header)
            return data, uvw, xyz, frame, snapshot
        finally:
            cleanup = ct.c_int(0)
            if block:
                lib.oskar_vis_block_free(block, ct.byref(cleanup))
            if header:
                lib.oskar_vis_header_free(header, ct.byref(cleanup))
            lib.oskar_binary_free(stream)


def directions(time, coordinates, frame, eop):
    """Equinox-apparent route independent of native CIRS implementation."""
    dut1 = np.interp(time.utc.mjd, eop[:, 0], eop[:, 1])
    xp = np.interp(time.utc.mjd, eop[:, 0], eop[:, 2]) * np.pi / (180 * 3600)
    yp = np.interp(time.utc.mjd, eop[:, 0], eop[:, 3]) * np.pi / (180 * 3600)
    ut1 = erfa.utcut1(time.utc.jd1, time.utc.jd2, dut1)
    tt = (time.tt.jd1, time.tt.jd2)
    gast = erfa.gst06a(*ut1, *tt)
    ra, dec = np.asarray(coordinates).T
    if frame != "apparent":
        if frame == "fk5_j2000":
            ra, dec = erfa.fk5hz(ra, dec, 2451545.0, 0.0)
        ri, dec, eo = erfa.atci13(ra, dec, 0.0, 0.0, 0.0, 0.0, *tt)
        ra = ri - eo
    v = np.column_stack((np.cos(dec) * np.cos(ra - gast), np.cos(dec) * np.sin(ra - gast), np.sin(dec)))
    pm = erfa.pom00(xp, yp, erfa.sp00(*tt))
    return v @ pm.T


def bases(time, coordinates, frame, eop):
    coordinates = np.asarray(coordinates)
    p = directions(time, coordinates, frame, eop)
    # Independent two-point stencil at a different step from the C implementation.
    step = 2e-6
    plus, minus = coordinates.copy(), coordinates.copy()
    plus[:, 1] += step
    minus[:, 1] -= step
    north = directions(time, plus, frame, eop) - directions(time, minus, frame, eop)
    north -= np.sum(north * p, axis=1)[:, None] * p
    north /= np.linalg.norm(north, axis=1)[:, None]
    east = np.cross(north, p)
    return p, north, east


def enu_matrix(lon, lat):
    return np.array(
        [
            [-np.sin(lon), np.cos(lon), 0.0],
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        ]
    )


def gaussian_factors(time, coordinates, frame, eop):
    """Independent two-point shape Jacobian, returned in ITRS (not UVW)."""
    factors = []
    widths = np.deg2rad(np.array([[2.0, 1.0, 20.0], [4.0, 1.0, 70.0], [0.0, 0.0, 0.0]]))
    widths[:, :2] /= 3600
    step = 2e-6
    for (ra, dec), (major, minor, pa) in zip(coordinates, widths, strict=True):
        centre = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
        north = np.array([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)])
        east = np.array([-np.sin(ra), np.cos(ra), 0.0])
        axes = []
        for width, tangent in (
            (major, np.sin(pa) * east + np.cos(pa) * north),
            (minor, np.cos(pa) * east - np.sin(pa) * north),
        ):
            p = np.cos(step) * centre + np.sin(step) * tangent
            m = np.cos(step) * centre - np.sin(step) * tangent
            angles = [[np.arctan2(v[1], v[0]), np.arctan2(v[2], np.hypot(v[0], v[1]))] for v in (p, m)]
            mapped = directions(time, angles, frame, eop)
            axes.append((mapped[0] - mapped[1]) / (2 * step) * width * np.pi / (2 * np.sqrt(np.log(2))))
        factors.append(axes)
    return np.array(factors)


def expected(case, xyz, eop, sky, phase, station_geo, elements, beam_evaluator=None):
    full = case.get("full", False)
    result = np.zeros((3, 2, 3, 4 if full else 1), complex)
    uvw = np.empty((3, 3, 3))
    pairs = [(0, 1), (0, 2), (1, 2)]
    nt = case.get("nt", 1) if case.get("dt", 0) else 1
    nf = case.get("nf", 1) if case.get("df", 0) else 1
    frame = case.get("frame", "icrs")
    xt, wt = np.polynomial.legendre.leggauss(nt)
    xf, wf = np.polynomial.legendre.leggauss(nf)
    start = Time(59000.25, format="mjd", scale="utc")
    for ti in range(3):
        centre_time = start + (ti + 0.5) * 1800 * u.s
        p0, n0, e0 = bases(centre_time, [phase], frame, eop)
        b = np.array([e0[0], n0[0], p0[0]])
        for bi, (a, c) in enumerate(pairs):
            uvw[ti, bi] = ((xyz[a] - xyz[c]) if case.get("casa", True) else (xyz[c] - xyz[a])) @ b.T
        for it in range(nt):
            time = centre_time + case.get("dt", 0) * xt[it] / 2 * u.s
            p, north, east = bases(time, sky[:, :2], frame, eop)
            phase_dir = directions(time, [phase], frame, eop)[0]
            shape = gaussian_factors(time, sky[:, :2], frame, eop) if case.get("gaussian") else None
            rotations, local_dirs, local_phase = [], [], []
            for lon, lat, _ in station_geo:
                enu = enu_matrix(lon, lat)
                dirs = p @ enu.T
                local_dirs.append(dirs)
                local_phase.append(enu @ phase_dir)
                q = np.arctan2(east @ enu[2], north @ enu[2])
                r = np.zeros((len(p), 2, 2))
                r[:, 0, 0] = r[:, 1, 1] = np.cos(q)
                r[:, 0, 1], r[:, 1, 0] = -np.sin(q), np.sin(q)
                rotations.append(r)
            for fi, f0 in enumerate((100e6, 120e6)):
                for jf in range(nf):
                    freq = f0 + case.get("df", 0) * xf[jf] / 2
                    weight = wt[it] * wf[jf] / 4
                    k = 2 * np.pi * freq / 299792458.0
                    gains = np.ones((3, len(p)), dtype=complex)
                    station_jones = (
                        beam_evaluator(time, freq, p, phase_dir, station_geo) if beam_evaluator is not None else None
                    )
                    if case.get("aa") and station_jones is None:
                        for st in range(3):
                            gains[st] = np.mean(
                                np.exp(1j * k * ((local_dirs[st] - local_phase[st]) @ elements.T)),
                                axis=1,
                            )
                            gains[st, local_dirs[st][:, 2] <= 0] = 0
                    for bi, (left, right) in enumerate(pairs):
                        if not case.get("casa", True):
                            left, right = right, left
                        ph = np.exp(1j * k * ((p - phase_dir) @ (xyz[left] - xyz[right])))
                        amp = gains[left] * gains[right].conj()
                        if shape is not None:
                            projections = shape @ ((xyz[left] - xyz[right]) * freq / 299792458.0)
                            amp *= np.exp(-np.sum(projections**2, axis=1))
                        for si, source in enumerate(sky):
                            if case.get("horizon", True) and all(local_dirs[st][si, 2] <= 0 for st in range(3)):
                                continue
                            flux = source[2:6] * (freq / 100e6) ** source[6]
                            if full:
                                ii, q, u_, vv = flux
                                brightness = np.array([[ii + q, u_ + 1j * vv], [u_ - 1j * vv, ii - q]])
                                value = rotations[left][si] @ brightness @ rotations[right][si].T
                                if station_jones is not None:
                                    value = station_jones[left, si] @ value @ station_jones[right, si].conj().T
                                result[ti, fi, bi] += (ph[si] * amp[si] * value).ravel() * weight
                            else:
                                result[ti, fi, bi, 0] += ph[si] * amp[si] * flux[0] * weight
    return result, uvw


def run(args):
    iers.conf.auto_download = False
    table = iers.IERS_B.open(iers.IERS_B_FILE)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    days = Time(np.arange(58999.0, 59003.0), format="mjd", scale="utc")
    dut1, st = table.ut1_utc(days, return_status=True)
    xp, yp, sp = table.pm_xy(days, return_status=True)
    assert np.all(st >= 0) and np.all(sp >= 0)
    eop = np.column_stack(
        [
            days.mjd,
            dut1.to_value(u.s),
            xp.to_value(u.arcsec),
            yp.to_value(u.arcsec),
            np.full(4, 37.0),
        ]
    )
    np.savetxt(
        root / "eop.txt",
        eop,
        fmt="%.17g",
        header=(
            "IERS-B sha256 "
            + hashlib.sha256(Path(iers.IERS_B_FILE).read_bytes()).hexdigest()
            + "\nMJD_UTC DUT1_s xp_arcsec yp_arcsec TAI_UTC_s"
        ),
    )
    lon, lat = np.deg2rad([116.67, -26.7])
    ref = Time(59000.25, format="mjd", scale="utc")
    dut = eop[1, 1]
    ra0 = erfa.gst06a(*erfa.utcut1(ref.jd1, ref.jd2, dut), ref.tt.jd1, ref.tt.jd2) + lon
    phase = np.array([ra0 % (2 * np.pi), lat])
    sky = np.array(
        [
            [phase[0], phase[1], 2.0, 0.2, 0.1, 0.03, -0.7],
            [phase[0] + 0.01, phase[1] + 0.015, 1.2, -0.1, 0.02, -0.01, -0.9],
            [phase[0] - 0.025, phase[1] - 0.01, 0.3, 0.0, 0.0, 0.0, -0.5],
        ]
    )
    elements = np.array([[-8.0, -2.0, 0.0], [3.0, 5.0, 1.0], [9.0, -4.0, 0.0]])
    layout = np.array([[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [12000.0, 50000.0, 15.0]])
    telescope = root / "telescope.tm"
    telescope.mkdir(exist_ok=True)
    np.savetxt(telescope / "position.txt", [[np.rad2deg(lon), np.rad2deg(lat), 300.0]])
    np.savetxt(telescope / "layout.txt", layout)
    for st in range(3):
        station = telescope / f"station{st:03d}"
        station.mkdir(exist_ok=True)
        np.savetxt(station / "layout.txt", elements)
    osm = np.column_stack([np.rad2deg(sky[:, :2]), sky[:, 2:6], np.full(3, 100e6), sky[:, 6]])
    np.savetxt(root / "sky.osm", osm, fmt="%.17g")
    reader = Reader(args.build / "oskar/liboskar.so")
    cases = [
        {"name": "icrs_scalar"},
        {"name": "apparent_scalar", "frame": "apparent"},
        {"name": "fk5_scalar", "frame": "fk5_j2000"},
        {"name": "icrs_full", "full": True},
        {"name": "icrs_other_sign", "casa": False},
        {"name": "icrs_aa", "aa": True},
        {"name": "icrs_aa_full", "aa": True, "full": True},
        {
            "name": "apparent_horizon",
            "frame": "apparent",
            "aa": True,
            "full": True,
            "grazing": True,
        },
        {
            "name": "icrs_integrated",
            "aa": True,
            "full": True,
            "dt": 10.0,
            "df": 5e4,
            "nt": 8,
            "nf": 8,
        },
        {
            "name": "icrs_integrated16",
            "aa": True,
            "full": True,
            "dt": 10.0,
            "df": 5e4,
            "nt": 16,
            "nf": 16,
        },
        {
            "name": "icrs_chunks_workers",
            "aa": True,
            "full": True,
            "chunk": 1,
            "workers": 2,
        },
    ]
    cases.extend(
        [
            {"name": "gaussian_apparent", "frame": "apparent", "gaussian": True},
            {"name": "gaussian_icrs", "gaussian": True, "full": True},
            {"name": "gaussian_fk5", "frame": "fk5_j2000", "gaussian": True, "full": True},
            {"name": "gaussian_aa", "gaussian": True, "full": True, "aa": True},
            {"name": "gaussian_other_sign", "gaussian": True, "full": True, "casa": False},
            {
                "name": "gaussian_integrated",
                "gaussian": True,
                "aa": True,
                "full": True,
                "dt": 10.0,
                "df": 5e4,
                "nt": 8,
                "nf": 8,
            },
            {"name": "gaussian_chunks", "gaussian": True, "full": True, "aa": True, "workers": 2, "chunk": 1},
        ]
    )
    records, products = [], {}
    for case in cases:
        run_dir = root / case["name"]
        run_dir.mkdir(exist_ok=True)
        case_sky = sky
        sky_path = root / "sky.osm"
        if case.get("gaussian"):
            case_sky = sky.copy()
            # A large offset exposes any omitted w projection in the envelope.
            case_sky[1, :2] += [0.3, 0.1]
            sky_path = run_dir / "sky.osm"
            np.savetxt(
                sky_path,
                np.column_stack(
                    (
                        np.rad2deg(case_sky[:, :2]),
                        case_sky[:, 2:6],
                        np.full(3, 100e6),
                        case_sky[:, 6],
                        np.zeros(3),
                        [[2.0, 1.0, 20.0], [4.0, 1.0, 70.0], [0.0, 0.0, 0.0]],
                    )
                ),
                fmt="%.17g",
            )
        if case.get("grazing"):
            time = ref + 900 * u.s
            gast = erfa.gst06a(
                *erfa.utcut1(time.jd1, time.jd2, np.interp(time.mjd, eop[:, 0], eop[:, 1])),
                time.tt.jd1,
                time.tt.jd2,
            )
            # One near the rising horizon, one below all station horizons.
            case_sky = np.vstack(
                (
                    sky,
                    [gast + lon + np.pi / 2 + 3e-5, 0.0, 10.0, 0.1, 0.2, 0.03, -0.8],
                    [gast + lon + np.pi, 0.0, 20.0, 0.0, 0.0, 0.0, -0.7],
                )
            )
            sky_path = run_dir / "sky.osm"
            np.savetxt(
                sky_path,
                np.column_stack(
                    (
                        np.rad2deg(case_sky[:, :2]),
                        case_sky[:, 2:6],
                        np.full(len(case_sky), 100e6),
                        case_sky[:, 6],
                    )
                ),
                fmt="%.17g",
            )
        opts = {
            "simulator": {
                "use_gpus": "false",
                "double_precision": "true",
                "num_devices": case.get("workers", 1),
                "max_sources_per_chunk": case.get("chunk", 64),
            },
            "sky": {
                "oskar_sky_model/file": sky_path,
                "advanced/apply_horizon_clip": "true",
            },
            "observation": {
                "start_time_utc": "31-05-2020 06:00:00.000",
                "length": 5400,
                "num_time_steps": 3,
                "start_frequency_hz": 100e6,
                "frequency_inc_hz": 20e6,
                "num_channels": 2,
                "phase_centre_ra_deg": np.rad2deg(phase[0]),
                "phase_centre_dec_deg": np.rad2deg(phase[1]),
            },
            "telescope": {
                "input_directory": telescope,
                "pol_mode": "Full" if case.get("full") else "Scalar",
                "station_type": "Aperture array" if case.get("aa") else "Isotropic beam",
                "allow_station_beam_duplication": "false",
                "normalise_beams_at_phase_centre": "false",
                "aperture_array/element_pattern/functional_type": ("Isotropic (unpolarised)"),
                "aperture_array/element_pattern/enable_numerical": "false",
                "aperture_array/array_pattern/normalise": "true",
            },
            "interferometer": {
                "astrometry/mode": "precision",
                "astrometry/frame": case.get("frame", "icrs"),
                "astrometry/eop_file": root / "eop.txt",
                "astrometry/time_samples": case.get("nt", 8),
                "astrometry/frequency_samples": case.get("nf", 8),
                "time_average_sec": case.get("dt", 0),
                "channel_bandwidth_hz": case.get("df", 0),
                "use_casa_phase_convention": str(case.get("casa", True)).lower(),
                "oskar_vis_filename": run_dir / "result.vis",
                "max_time_samples_per_block": 2,
                "max_channels_per_block": 1,
            },
        }
        if args.ms:
            opts["interferometer"]["ms_filename"] = run_dir / "result.ms"
        text = "\n".join(
            f"[{section}]\n" + "\n".join(f"{k}={v}" for k, v in values.items()) for section, values in opts.items()
        )
        config = run_dir / "settings.ini"
        config.write_text(text + "\n", encoding="ascii")
        env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
        env.pop("LD_LIBRARY_PATH", None)
        with (run_dir / "run.log").open("w") as log:
            subprocess.run(
                [str(args.build / "apps/oskar_sim_interferometer"), str(config)],
                cwd=run_dir,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        actual, uvw, xyz, frame, snapshot = reader.read(run_dir / "result.vis", case.get("full", False))
        assert frame == {"apparent": 1, "icrs": 2, "fk5_j2000": 3}[case.get("frame", "icrs")]
        assert snapshot == (root / "eop.txt").read_bytes()
        # Station ENU frames are defined by geodetic coordinates of each station.
        centre = erfa.gd2gc(1, lon, lat, 300.0)
        station_geo = np.array([erfa.gc2gd(1, centre + p) for p in xyz])
        predicted, uvw_ref = expected(case, xyz, eop, case_sky, phase, station_geo, elements)
        error = np.max(np.abs(actual - predicted))
        uvw_error = np.max(np.abs(uvw - uvw_ref))
        record = {
            "case": case["name"],
            "visibility_max_abs_jy": float(error),
            "uvw_max_abs_m": float(uvw_error),
        }
        records.append(record)
        print(record, flush=True)
        np.savez_compressed(
            run_dir / "comparison.npz",
            actual=actual,
            reference=predicted,
            uvw=uvw,
            uvw_reference=uvw_ref,
        )
        products[case["name"]] = actual
        # Finite-difference reference basis limits the UVW comparison at ~um.
        np.testing.assert_allclose(actual, predicted, rtol=0, atol=2e-8)
        np.testing.assert_allclose(uvw, uvw_ref, rtol=0, atol=3e-5)
        if args.ms:
            from casacore.tables import table as ms_table

            with ms_table(str(run_dir / "result.ms"), ack=False) as ms:
                assert ms.getkeyword("OSKAR_ASTROMETRY_FRAME") == {1: "APP", 2: "ICRS", 3: "J2000"}[frame]
                assert ms.getkeyword("OSKAR_ASTROMETRY_EOP") == snapshot.decode("ascii")
                ms_data = ms.getcol("DATA")
                ms_data = ms_data.reshape(3, 3, 2, ms_data.shape[-1]).transpose(0, 2, 1, 3)
                ms_uvw = ms.getcol("UVW").reshape(3, 3, 3)
                np.testing.assert_allclose(ms_uvw, uvw, rtol=0, atol=1e-9)
                # MSv2 DATA remains complex64, unlike the double binary output.
                compare = ms_data if case.get("full") else ms_data[..., :1]
                np.testing.assert_allclose(compare, actual, rtol=1e-7, atol=3e-7)
            with ms_table(str(run_dir / "result.ms/FIELD"), ack=False) as field:
                assert field.getcolkeyword("PHASE_DIR", "MEASINFO")["Ref"] == {1: "APP", 2: "ICRS", 3: "J2000"}[frame]
            with ms_table(str(run_dir / "result.ms/SPECTRAL_WINDOW"), ack=False) as spw:
                np.testing.assert_array_equal(spw.getcell("CHAN_FREQ", 0), [100e6, 120e6])
                if case.get("df", 0) > 0:
                    for column in ("CHAN_WIDTH", "EFFECTIVE_BW", "RESOLUTION"):
                        np.testing.assert_array_equal(spw.getcell(column, 0), case["df"])
                    assert spw.getcell("TOTAL_BANDWIDTH", 0) == 20e6 + case["df"]
    np.testing.assert_allclose(products["icrs_aa_full"], products["icrs_chunks_workers"], rtol=0, atol=2e-12)
    report = {
        "records": records,
        "quadrature_8_vs_16_max_jy": float(np.max(np.abs(products["icrs_integrated"] - products["icrs_integrated16"]))),
        "scope": __doc__,
    }
    assert report["quadrature_8_vs_16_max_jy"] < 2e-8
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ms", action="store_true", help="Also verify MS DATA, UVW and frame metadata")
    args = parser.parse_args()
    args.build = args.build.resolve(strict=True)
    run(args)
