"""Reproduce legacy UTC/GAST UVW approximations on unmodified OSKAR 2.13.0.

Only existing public C APIs are used; no precision patch or GPU is needed.
All RA/Dec values below are equinox-based apparent coordinates. This does not
compare identical numeric coordinates in different celestial reference frames.
NumPy, Astropy (with bundled historical IERS-B) and PyERFA are test dependencies.
"""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import astropy
import erfa
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.utils import iers


def rotation(ra, dec, gast):
    h = gast - ra
    return np.array([
        [np.sin(h), np.cos(h), 0.0],
        [-np.sin(dec) * np.cos(h), np.sin(dec) * np.sin(h), np.cos(dec)],
        [np.cos(dec) * np.cos(h), -np.cos(dec) * np.sin(h), np.sin(dec)],
    ])


def project(lib, xyz, ra, dec, gast):
    buffers, status = [], ct.c_int(0)
    try:
        for axis in range(6):
            mem = lib.oskar_mem_create(8, 0, len(xyz), ct.byref(status))
            if mem:
                buffers.append(mem)
            if status.value or not mem:
                raise RuntimeError(f"allocation status {status.value}")
            if axis < 3:
                values = lib.oskar_mem_double(mem, ct.byref(status))
                np.ctypeslib.as_array(values, shape=(len(xyz),))[:] = xyz[:, axis]
        lib.oskar_convert_ecef_to_station_uvw(
            len(xyz), *buffers[:3], ra, dec, gast, 0, 0, *buffers[3:], ct.byref(status),
        )
        if status.value:
            raise RuntimeError(f"projection status {status.value}")
        return np.column_stack([
            np.ctypeslib.as_array(lib.oskar_mem_double(mem, ct.byref(status)), shape=(len(xyz),)).copy()
            for mem in buffers[3:]
        ])
    finally:
        cleanup = ct.c_int(0)
        for mem in buffers:
            lib.oskar_mem_free(mem, ct.byref(cleanup))


def run(library):
    lib = ct.CDLL(str(library))
    p, i, d, s = ct.c_void_p, ct.c_int, ct.c_double, ct.POINTER(ct.c_int)
    for name, result, args in (
        ("oskar_mem_create", p, [i, i, ct.c_size_t, s]),
        ("oskar_mem_double", ct.POINTER(d), [p, s]),
        ("oskar_mem_free", None, [p, s]),
        ("oskar_convert_mjd_to_gast_fast", d, [d]),
        ("oskar_convert_ecef_to_station_uvw", None, [i, p, p, p, d, d, d, i, i, p, p, p, s]),
    ):
        func = getattr(lib, name)
        func.restype, func.argtypes = result, args
    iers.conf.auto_download = False
    table_path = Path(iers.IERS_B_FILE)
    table = iers.IERS_B.open(str(table_path))
    xyz = np.array([[0., 0., 0.], [1000., 2000., 3000.], [-10000., 40000., 2000.], [62000., -25000., 1000.]])
    rows = []
    with iers.earth_orientation_table.set(table):
        for mjd in (59000.25, 60000.5, 60544.583333333336):
            time = Time(mjd, format="mjd", scale="utc")
            dut1, status = table.ut1_utc(time, return_status=True)
            if status < 0:
                raise ValueError("Date outside bundled IERS-B; supply a suitable local table, not predictions")
            fast_utc = lib.oskar_convert_mjd_to_gast_fast(time.utc.mjd)
            fast_ut1 = lib.oskar_convert_mjd_to_gast_fast(time.ut1.mjd)
            precise = erfa.gst06a(time.ut1.jd1, time.ut1.jd2, time.tt.jd1, time.tt.jd2)
            for ra, dec in ((0., -0.5), (1.2, 0.3), (2., -1.0)):
                actual = project(lib, xyz, ra, dec, fast_utc)
                same = xyz @ rotation(ra, dec, fast_utc).T
                corrected_time = xyz @ rotation(ra, dec, fast_ut1).T
                reference = xyz @ rotation(ra, dec, precise).T
                np.testing.assert_allclose(actual, same, rtol=0, atol=1e-9)
                rows.append({
                    "mjd_utc": mjd, "ra_apparent_rad": ra, "dec_apparent_rad": dec,
                    "dut1_s": float(dut1.to_value(u.s)),
                    "same_formula_max_m": float(np.max(np.abs(actual - same))),
                    "utc_as_ut1_max_m": float(np.max(np.abs(corrected_time - actual))),
                    "fast_gast_after_ut1_max_m": float(np.max(np.abs(reference - corrected_time))),
                    "combined_gast_change_max_m": float(np.max(np.abs(reference - actual))),
                })
    return {
        "scope": __doc__, "records": rows,
        "max_offset_m": float(np.linalg.norm(xyz, axis=1).max()),
        "astropy": astropy.__version__, "pyerfa": erfa.__version__,
        "iers_b_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "limits": "No polar-motion contribution, sky conversion, beam or visibility comparison; not an EoR error bound",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(run(args.library.resolve(strict=True)), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report, end="")
