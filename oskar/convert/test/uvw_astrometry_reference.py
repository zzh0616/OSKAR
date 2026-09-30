"""Offline UVW preparation gate; optional NumPy, Astropy and PyERFA required.

This tests a geometric GCRS-to-ITRS rotation, NOT an ICRS apparent-place
solution or a complete interferometer. Astropy and the direct route share
ERFA: agreement checks conventions/implementation, not independent physics.
"""

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path

import astropy
import erfa
import numpy as np
from astropy import units as units
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation
from astropy.time import Time
from astropy.utils import iers


def tangent_basis(ra, dec):
    return np.array(
        [
            [-np.sin(ra), np.cos(ra), 0.0],
            [-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)],
            [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)],
        ]
    )


def apparent_rotation(ra, dec, gast):
    h = gast - ra
    return np.array(
        [
            [np.sin(h), np.cos(h), 0.0],
            [-np.sin(dec) * np.cos(h), np.sin(dec) * np.sin(h), np.cos(dec)],
            [np.cos(dec) * np.cos(h), -np.cos(dec) * np.sin(h), np.sin(dec)],
        ]
    )


class NativeProjection:
    def __init__(self, filename):
        self.lib = lib = ct.CDLL(str(filename))
        pointer, integer, real = ct.c_void_p, ct.c_int, ct.c_double
        self.dp, self.ip = ct.POINTER(real), ct.POINTER(integer)
        lib.oskar_mem_create.argtypes = [integer, integer, ct.c_size_t, self.ip]
        lib.oskar_mem_create.restype = pointer
        lib.oskar_mem_double.argtypes = [pointer, self.ip]
        lib.oskar_mem_double.restype = self.dp
        lib.oskar_mem_free.argtypes = [pointer, self.ip]
        lib.oskar_mem_free.restype = None
        lib.oskar_convert_ecef_to_station_uvw_matrix.argtypes = [
            integer,
            pointer,
            pointer,
            pointer,
            self.dp,
            integer,
            integer,
            pointer,
            pointer,
            pointer,
            self.ip,
        ]
        lib.oskar_convert_ecef_to_station_uvw_matrix.restype = None
        lib.oskar_convert_ecef_to_station_uvw.argtypes = [
            integer,
            pointer,
            pointer,
            pointer,
            real,
            real,
            real,
            integer,
            integer,
            pointer,
            pointer,
            pointer,
            self.ip,
        ]
        lib.oskar_convert_ecef_to_station_uvw.restype = None
        lib.oskar_convert_mjd_to_gast_fast.argtypes = [real]
        lib.oskar_convert_mjd_to_gast_fast.restype = real

    def project(self, xyz, matrix=None, apparent=None):
        n = len(xyz)
        status = ct.c_int(0)
        buffers = []
        try:
            for i in range(6):
                mem = self.lib.oskar_mem_create(0x08, 0, n, ct.byref(status))
                if mem:
                    buffers.append(mem)
                if status.value or not mem:
                    raise RuntimeError(f"allocation failed: {status.value}")
                if i < 3:
                    data = self.lib.oskar_mem_double(mem, ct.byref(status))
                    np.ctypeslib.as_array(data, shape=(n,))[:] = xyz[:, i]
            if matrix is not None:
                rotation = np.ascontiguousarray(matrix, dtype=np.float64)
                self.lib.oskar_convert_ecef_to_station_uvw_matrix(
                    n,
                    *buffers[:3],
                    rotation.ctypes.data_as(self.dp),
                    0,
                    0,
                    *buffers[3:],
                    ct.byref(status),
                )
            else:
                self.lib.oskar_convert_ecef_to_station_uvw(
                    n,
                    *buffers[:3],
                    *apparent,
                    0,
                    0,
                    *buffers[3:],
                    ct.byref(status),
                )
            if status.value:
                raise RuntimeError(f"projection failed: {status.value}")
            result = np.column_stack(
                [
                    np.ctypeslib.as_array(
                        self.lib.oskar_mem_double(mem, ct.byref(status)),
                        shape=(n,),
                    ).copy()
                    for mem in buffers[3:]
                ]
            )
            if status.value:
                raise RuntimeError(f"read failed: {status.value}")
            return result
        finally:
            cleanup = ct.c_int(0)
            for mem in buffers:
                self.lib.oskar_mem_free(mem, ct.byref(cleanup))


def run(library):
    iers.conf.auto_download = False
    table_path = Path(iers.IERS_B_FILE)
    table = iers.IERS_B.open(str(table_path))
    native = NativeProjection(library)
    xyz = np.array(
        [[0.0, 0.0, 0.0], [1000.0, 2000.0, 3000.0], [-10000.0, 40000.0, 2000.0], [62000.0, -25000.0, 1000.0]]
    )
    records = []
    with iers.earth_orientation_table.set(table):
        for mjd in (59000.25, 60000.5, 60544.583333333336):
            time = Time(mjd, format="mjd", scale="utc")
            dut1, ut1_status = table.ut1_utc(time, return_status=True)
            xp, yp, pm_status = table.pm_xy(time, return_status=True)
            if ut1_status < 0 or pm_status < 0:
                raise ValueError("reference date outside bundled IERS-B coverage")
            tt, ut1 = time.tt, time.ut1
            rotation = erfa.c2t06a(tt.jd1, tt.jd2, ut1.jd1, ut1.jd2, xp.to_value(units.rad), yp.to_value(units.rad))
            # Rotate geocentric basis vectors, not ICRS catalogue sources.
            axes = GCRS(CartesianRepresentation(np.eye(3) * units.m), obstime=time)
            astropy_rotation = axes.transform_to(ITRS(obstime=time)).cartesian.xyz.to_value(units.m)
            np.testing.assert_allclose(rotation, astropy_rotation, rtol=0, atol=2e-14)
            fast_gast = native.lib.oskar_convert_mjd_to_gast_fast(time.utc.mjd)
            fast_gast_ut1 = native.lib.oskar_convert_mjd_to_gast_fast(time.ut1.mjd)
            precise_gast = erfa.gst06a(ut1.jd1, ut1.jd2, tt.jd1, tt.jd2)
            for ra, dec in ((0.0, -0.5), (1.2, 0.3), (2.0, -1.0)):
                basis = tangent_basis(ra, dec)
                projected = native.project(xyz, matrix=basis @ rotation.T)
                expected = (basis @ astropy_rotation.T @ xyz.T).T
                np.testing.assert_allclose(projected, expected, rtol=0, atol=2e-9)
                legacy = native.project(xyz, apparent=(ra, dec, fast_gast))
                matrix_fast = native.project(xyz, matrix=apparent_rotation(ra, dec, fast_gast))
                np.testing.assert_allclose(legacy, matrix_fast, rtol=0, atol=1e-9)
                precise_apparent = native.project(xyz, apparent=(ra, dec, precise_gast))
                fast_ut1_apparent = native.project(xyz, apparent=(ra, dec, fast_gast_ut1))
                records.append(
                    {
                        "mjd_utc": mjd,
                        "ra_rad": ra,
                        "dec_rad": dec,
                        "dut1_s": float(dut1.to_value(units.s)),
                        "xp_arcsec": float(xp.to_value(units.arcsec)),
                        "yp_arcsec": float(yp.to_value(units.arcsec)),
                        "matrix_vs_astropy_max_m": float(np.max(np.abs(projected - expected))),
                        "matrix_vs_legacy_same_formula_max_m": float(np.max(np.abs(matrix_fast - legacy))),
                        "gast_only_change_max_m": float(np.max(np.abs(precise_apparent - legacy))),
                        "utc_as_ut1_effect_max_m": float(np.max(np.abs(fast_ut1_apparent - legacy))),
                        "gast_approximation_after_ut1_max_m": float(
                            np.max(np.abs(precise_apparent - fast_ut1_apparent))
                        ),
                        "different_frame_contrast_max_m": float(np.max(np.abs(projected - legacy))),
                    }
                )
    return {
        "scope": "Coordinate preparation, not visibility or sky-frame end-to-end validation",
        "frame_warning": "Same RA/Dec numbers in apparent and GCRS are NOT the same physical direction",
        "reference_warning": "Astropy and direct route share ERFA; no independent absolute-accuracy bound",
        "astropy": astropy.__version__,
        "pyerfa": erfa.__version__,
        "iers_b_sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "max_station_offset_m": float(np.max(np.linalg.norm(xyz, axis=1))),
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(run(args.library.resolve(strict=True)), indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)
