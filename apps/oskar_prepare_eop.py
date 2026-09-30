#!/usr/bin/env python3
"""Export an offline IERS-B snapshot for OSKAR precision astrometry.

Requires Astropy, NumPy and PyERFA only when exporting. OSKAR itself links
ERFA and never imports Python or accesses the network. No predicted EOP or
extrapolation is silently substituted for historical measurements.
"""

import argparse
import hashlib
from pathlib import Path

import erfa
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.utils import iers


def export_eop(start, end, output, input_table=None):
    iers.conf.auto_download = False
    source = Path(input_table or iers.IERS_B_FILE).resolve(strict=True)
    first, last = Time(start, scale="utc"), Time(end, scale="utc")
    if not first.isscalar or not last.isscalar or last < first:
        raise ValueError("Start and end must be ordered scalar UTC dates")
    # Include complete daily brackets, even for an instantaneous observation.
    days = Time(
        np.arange(np.floor(first.mjd), np.floor(last.mjd) + 2),
        format="mjd",
        scale="utc",
    )
    table = iers.IERS_B.open(str(source))
    dut1, status_t = table.ut1_utc(days, return_status=True)
    xp, yp, status_p = table.pm_xy(days, return_status=True)
    if np.any(status_t < 0) or np.any(status_p < 0):
        raise ValueError("The local IERS-B table does not cover all daily brackets")
    dates = days.utc.ymdhms
    dat = erfa.dat(dates.year, dates.month, dates.day, 0.0)
    values = np.column_stack(
        (
            days.mjd,
            dut1.to_value(u.s),
            xp.to_value(u.arcsec),
            yp.to_value(u.arcsec),
            dat,
        )
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite EOP values")
    header = (
        f"IERS-B source: {source.name}\n"
        f"sha256: {hashlib.sha256(source.read_bytes()).hexdigest()}\n"
        "MJD_UTC DUT1_seconds xp_arcsec yp_arcsec TAI_minus_UTC_seconds"
    )
    # Never silently replace an input snapshot used by an existing run.
    with Path(output).open("x", encoding="ascii") as stream:
        np.savetxt(stream, values, fmt="%.17g", header=header)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, help="UTC ISO date/time")
    parser.add_argument("--end", required=True, help="UTC ISO date/time")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--iers-b", type=Path, help="Local IERS-B file (default: bundled)"
    )
    args = parser.parse_args()
    export_eop(args.start, args.end, args.output, args.iers_b)
