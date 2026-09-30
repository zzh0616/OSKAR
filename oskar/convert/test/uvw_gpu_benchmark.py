"""Bounded CPU/CUDA timing and accuracy check, not a production sky simulation."""

import argparse
import configparser
import json
import re
import shutil
from pathlib import Path

import numpy as np
from uvw_gpu_reference import compare, run_case


def run(args):
    args.build = args.build.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = args.fixture.resolve(strict=True)
    config = configparser.ConfigParser(interpolation=None)
    config.read(fixture / "icrs_aa_full/settings.ini")
    rng = np.random.default_rng(9322)
    ns, ne, nsrc = 64, 128, 4096
    telescope = args.output / "telescope.tm"
    telescope.mkdir(exist_ok=True)
    shutil.copyfile(fixture / "telescope.tm/position.txt", telescope / "position.txt")
    layout = rng.uniform(-10000, 10000, (ns, 3))
    layout[:, 2] = 0.0
    layout[0] = 0.0
    np.savetxt(telescope / "layout.txt", layout)
    element = rng.uniform(-16, 16, (ne, 3))
    element[:, 2] *= 0.01
    station = telescope / "station000"
    station.mkdir(exist_ok=True)
    np.savetxt(station / "layout.txt", element)
    ra = float(config["observation"]["phase_centre_ra_deg"])
    dec = float(config["observation"]["phase_centre_dec_deg"])
    sky = np.zeros((nsrc, 12))
    sky[:, :2] = rng.uniform(-3, 3, (nsrc, 2)) + [ra, dec]
    sky[:, 2] = rng.uniform(0.01, 1, nsrc)
    sky[:, 3:6] = sky[:, 2:3] * rng.uniform(-0.05, 0.05, (nsrc, 3))
    sky[:, 6] = 1e8
    sky[:, 7] = -0.7
    sky[::4, 9:11] = [4.0, 2.0]
    sky[::4, 11] = 30.0
    np.savetxt(args.output / "sky.osm", sky, fmt="%.17g")
    config["sky"]["oskar_sky_model/file"] = str(args.output / "sky.osm")
    config["telescope"]["input_directory"] = str(telescope)
    config["observation"].update(length="120", num_time_steps="4", num_channels="4", frequency_inc_hz="1000000")
    config["simulator"]["max_sources_per_chunk"] = "1024"
    config["interferometer"]["max_channels_per_block"] = "4"
    config["interferometer"]["max_time_samples_per_block"] = "4"
    with (args.output / "settings.ini").open("w") as stream:
        config.write(stream, space_around_delimiters=False)
    args.multi_chunk = 1024
    report = dict(
        status="running",
        stations=ns,
        elements_per_station=ne,
        sources=nsrc,
        times=4,
        channels=4,
        cpu_workers=args.cpu_workers,
        cpu_omp_threads_per_worker=1,
        gaussian_fraction=0.25,
        cases=[],
    )
    path = args.output / "report.json"
    path.write_text(json.dumps(report) + "\n")
    try:
        reference = None
        for repeat in range(args.repeats):
            for label, devices, cpu in (
                ("cpu", args.gpu, True),
                ("cuda", args.gpu, False),
                ("cuda2", args.multi_gpu, False),
            ):
                if not devices:
                    continue
                name = f"{label}_{repeat}"
                result, elapsed = run_case(args, args.output, name, devices, cpu=cpu)
                if reference is None:
                    reference = result
                record = compare(name, result, reference, elapsed)
                log = (args.output / name / "run.log").read_text()
                for key, pattern in (
                    ("simulation_sec", r"Total wall time\s*:\s*([\d.]+) s"),
                    ("host_geometry_upload_sec_sum", r"Host astrometry incl\. upload:\s*([\d.]+) s"),
                    ("jones_e_percent", r"Jones E\s*:\s*([\d.]+)%"),
                    ("correlate_percent", r"Jones correlate\s*:\s*([\d.]+)%"),
                ):
                    match = re.search(pattern, log)
                    assert match, (key, name)
                    record[key] = float(match[1])
                report["cases"].append(record)
                path.write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--multi-gpu")
    parser.add_argument("--omp-threads", type=int, default=8)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=2)
    run(parser.parse_args())
