"""Offline CPU/CUDA precision-astrometry equivalence using validated fixtures.

Run the CPU analytic and beam reference suites first. This checks the exact same
inputs on CUDA, and does not replace the independent ITRS reference tests.
"""

import argparse
import configparser
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from uvw_simulation_reference import Reader


def run_case(args, source, label, devices, *, cpu=False, single=False, no_clip=False, all_below=False):
    config = configparser.ConfigParser(interpolation=None)
    config.read(source / "settings.ini")
    target = args.output / label
    target.mkdir(parents=True, exist_ok=True)
    sim = config["simulator"]
    sim["use_gpus"] = str(not cpu).lower()
    sim["cuda_device_ids"] = devices
    sim["num_devices"] = str(len(devices.split(","))) if not cpu else str(getattr(args, "cpu_workers", 1))
    sim["double_precision"] = str(not single).lower()
    config["interferometer"]["oskar_vis_filename"] = str(target / "result.vis")
    config["interferometer"].pop("ms_filename", None)
    if no_clip:
        config["sky"]["advanced/apply_horizon_clip"] = "false"
    if all_below:
        rows = np.atleast_2d(np.loadtxt(config["sky"]["oskar_sky_model/file"]))
        rows[:, 0] = (rows[:, 0] + 180.0) % 360.0
        rows[:, 1] *= -1.0
        np.savetxt(target / "below_horizon.osm", rows, fmt="%.17g")
        config["sky"]["oskar_sky_model/file"] = str(target / "below_horizon.osm")
    if "," in devices:
        sim["max_sources_per_chunk"] = str(getattr(args, "multi_chunk", 1))
    with (target / "settings.ini").open("w") as stream:
        config.write(stream, space_around_delimiters=False)
    env = dict(os.environ, OMP_NUM_THREADS=str(getattr(args, "omp_threads", 2)), OPENBLAS_NUM_THREADS="1")
    env.pop("LD_LIBRARY_PATH", None)
    start = time.monotonic()
    with (target / "run.log").open("w") as log:
        subprocess.run(
            [str(args.build / "apps/oskar_sim_interferometer"), str(target / "settings.ini")],
            cwd=target,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=300,
        )
    elapsed = time.monotonic() - start
    log = (target / "run.log").read_text()
    if "WARNING: Ignoring '" in log:
        raise AssertionError(f"ignored setting in {target}")
    full = config["telescope"]["pol_mode"].lower() == "full"
    return Reader(args.build / "oskar/liboskar.so").read(target / "result.vis", full), elapsed


def compare(label, output, reference, elapsed, single=False):
    actual, uvw, xyz, frame, eop = output
    expected, uvw_ref, xyz_ref, ref_frame, ref_eop = reference
    assert frame == ref_frame and eop == ref_eop
    assert np.isfinite(actual).all() and np.isfinite(uvw).all()
    np.testing.assert_array_equal(xyz, xyz_ref)
    np.testing.assert_array_equal(uvw, uvw_ref)
    scale = max(float(np.max(np.abs(expected))), 1e-30)
    error = float(np.max(np.abs(actual - expected)))
    # Single precision retains float phase arithmetic on long baselines. It is
    # not an EoR-accuracy claim. Double acceptance is referenced to peak flux,
    # not unstable relative errors at individual visibility nulls.
    tolerance = (2e-3 if single else 1e-10) * scale + (1e-5 if single else 2e-10)
    assert error <= tolerance, (label, error, tolerance)
    record = dict(case=label, max_abs_jy=error, normalised_max_error=error / scale, wall_sec=elapsed)
    print(record, flush=True)
    return record


def run(args):
    args.build = args.build.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report = dict(status="running", scope=__doc__, cases=[])
    report_path.write_text(json.dumps(report) + "\n")
    reader = Reader(args.build / "oskar/liboskar.so")
    try:
        for root in args.fixtures:
            root = root.resolve(strict=True)
            for settings in sorted(root.glob("*/settings.ini")):
                source = settings.parent
                config = configparser.ConfigParser(interpolation=None)
                config.read(settings)
                if config.get("interferometer", "astrometry/mode", fallback="legacy") != "precision":
                    continue
                label = root.name + "_" + source.name
                full = config["telescope"]["pol_mode"].lower() == "full"
                reference = reader.read(source / "result.vis", full)
                result, elapsed = run_case(args, source, label, args.gpu)
                report["cases"].append(compare(label, result, reference, elapsed))
                if source.name in ("icrs_aa_full", "gaussian_aa", "feko_offset"):
                    name = label + "_all_below"
                    cpu, _ = run_case(args, source, name + "_cpu", args.gpu, cpu=True, all_below=True)
                    gpu, elapsed = run_case(args, source, name, args.gpu, all_below=True)
                    assert np.count_nonzero(cpu[0]) == 0
                    assert np.count_nonzero(gpu[0]) == 0
                    report["cases"].append(compare(name, gpu, cpu, elapsed))
                    if args.multi_gpu:
                        name = label + "_multi"
                        result, elapsed = run_case(args, source, name, args.multi_gpu)
                        report["cases"].append(compare(name, result, reference, elapsed))
                    for no_clip in (False, True):
                        name = label + ("_noclip" if no_clip else "") + "_single"
                        cpu, _ = run_case(args, source, name + "_cpu", args.gpu, cpu=True, single=True, no_clip=no_clip)
                        gpu, elapsed = run_case(args, source, name, args.gpu, single=True, no_clip=no_clip)
                        report["cases"].append(compare(name, gpu, cpu, elapsed, single=True))
                    name = label + "_noclip"
                    result, elapsed = run_case(args, source, name, args.gpu, no_clip=True)
                    report["cases"].append(compare(name, result, reference, elapsed))
        assert report["cases"], "no precision fixtures found"
        report["status"] = "passed"
        report["library_sha256"] = hashlib.sha256((args.build / "oskar/liboskar.so").read_bytes()).hexdigest()
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0", help="CUDA device ID for single-card runs")
    parser.add_argument("--multi-gpu", help="Optional comma-separated IDs, e.g. 0,1")
    run(parser.parse_args())
