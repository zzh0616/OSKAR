"""Compare a compact Gaussian with explicit spherical point-source quadrature.

Uses the native simulator twice, but only the Gaussian run uses the new shape
Jacobian/correlator. This isolates its small-source approximation, including
centroid-only Jones evaluation, from the exact spherical point-source route.
Not a full radiative-transfer or independent astrometric reference.
"""

import argparse
import configparser
import json
import os
import subprocess
from pathlib import Path

import numpy as np
from uvw_simulation_reference import Reader


def point_cloud(ra, dec, major_arcsec, minor_arcsec, pa_deg, nodes):
    x, weight = np.polynomial.hermite.hermgauss(nodes)
    sigma = np.deg2rad([major_arcsec, minor_arcsec]) / (3600 * np.sqrt(8 * np.log(2)))
    aa, bb = np.meshgrid(np.sqrt(2) * sigma[0] * x, np.sqrt(2) * sigma[1] * x, indexing="ij")
    pa = np.deg2rad(pa_deg)
    centre = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    north = np.array([-np.sin(dec) * np.cos(ra), -np.sin(dec) * np.sin(ra), np.cos(dec)])
    east = np.array([-np.sin(ra), np.cos(ra), 0.0])
    tangent = aa.ravel()[:, None] * (np.sin(pa) * east + np.cos(pa) * north)
    tangent += bb.ravel()[:, None] * (np.cos(pa) * east - np.sin(pa) * north)
    angle = np.linalg.norm(tangent, axis=1)
    xyz = np.cos(angle)[:, None] * centre + np.sinc(angle / np.pi)[:, None] * tangent
    # Gaussian flux distribution in source-centred tangent coordinates, mapped
    # by the sphere exponential map, with total integrated I = 1 Jy.
    flux = np.outer(weight, weight).ravel() / np.pi
    return np.column_stack(
        (
            np.rad2deg(np.arctan2(xyz[:, 1], xyz[:, 0])),
            np.rad2deg(np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))),
            flux,
            np.zeros((len(flux), 3)),
            np.full(len(flux), 100e6),
            np.zeros(len(flux)),
        )
    )


def run(args):
    reader = Reader(args.build / "oskar/liboskar.so")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    template = configparser.ConfigParser(interpolation=None)
    template.read(args.fixtures / "gaussian_icrs/settings.ini")
    phase = np.deg2rad(
        [float(template["observation"]["phase_centre_ra_deg"]), float(template["observation"]["phase_centre_dec_deg"])]
    )
    ra, dec = phase + [0.3, 0.1]
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    env.pop("LD_LIBRARY_PATH", None)

    def simulate(label, sky, aa):
        directory = root / label
        directory.mkdir(exist_ok=True)
        np.savetxt(directory / "sky.osm", sky, fmt="%.17g")
        config = configparser.ConfigParser(interpolation=None)
        config.read_dict(template)
        config["sky"]["oskar_sky_model/file"] = str(directory / "sky.osm")
        config["simulator"]["max_sources_per_chunk"] = "1024"
        config["telescope"]["station_type"] = "Aperture array" if aa else "Isotropic beam"
        config["interferometer"]["oskar_vis_filename"] = str(directory / "result.vis")
        config["interferometer"].pop("ms_filename", None)
        with (directory / "settings.ini").open("w") as f:
            config.write(f, space_around_delimiters=False)
        with (directory / "run.log").open("w") as log:
            subprocess.run(
                [str(args.build / "apps/oskar_sim_interferometer"), str(directory / "settings.ini")],
                cwd=directory,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
        return reader.read(directory / "result.vis", True)[0]

    records = []
    for aa in (False, True):
        previous = None
        for size in (8.0, 4.0, 2.0):
            label = f"aa{int(aa)}_fwhm{size:g}"
            gaussian = [[*np.rad2deg([ra, dec]), 1.0, 0.0, 0.0, 0.0, 100e6, 0.0, 0.0, size, size / 2, 37.0]]
            native = simulate(label + "_gaussian", gaussian, aa)
            points = []
            for nodes in (24, 40):
                points.append(
                    simulate(label + f"_points{nodes}", point_cloud(ra, dec, size, size / 2, 37.0, nodes), aa)
                )
            quadrature = float(np.max(np.abs(points[0] - points[1])))
            error = float(np.max(np.abs(native - points[-1])))
            assert quadrature < 2e-9, quadrature
            # Relative to integrated flux, not almost-resolved visibility zeros.
            assert error < 1e-4, error
            if previous is not None:
                assert error < previous, (error, previous)
            previous = error
            record = {
                "aa": aa,
                "major_arcsec": size,
                "minor_arcsec": size / 2,
                "point_quadrature_max_abs_jy": quadrature,
                "gaussian_approximation_max_abs_jy_per_1jy": error,
            }
            records.append(record)
            print(record, flush=True)
    (root / "report.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build", "fixtures", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.build = args.build.resolve(strict=True)
    run(args)
