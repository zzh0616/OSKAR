"""Check precision preflight failures and legacy DATA/UVW against pristine OSKAR.

Run uvw_simulation_reference.py first to generate the synthetic fixture.
All subprocesses and products stay below the supplied output directory.
"""

import argparse
import configparser
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from uvw_simulation_reference import Reader


def execute(binary, config, directory, expect_ok):
    directory.mkdir(parents=True, exist_ok=True)
    config["interferometer"]["oskar_vis_filename"] = str(directory / "result.vis")
    config["interferometer"].pop("ms_filename", None)
    path = directory / "settings.ini"
    with path.open("w") as stream:
        config.write(stream, space_around_delimiters=False)
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    env.pop("LD_LIBRARY_PATH", None)
    with (directory / "run.log").open("w") as log:
        result = subprocess.run(
            [str(binary), str(path)],
            cwd=directory,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    assert (result.returncode == 0) == expect_ok, directory
    if expect_ok:
        assert "WARNING: Ignoring '" not in (directory / "run.log").read_text()
    if not expect_ok:
        assert result.returncode > 0, "A signal/crash is not a preflight rejection"
        assert not (directory / "result.vis").exists(), directory


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    fixture = configparser.ConfigParser(interpolation=None)
    fixture.read(args.fixtures / "icrs_scalar/settings.ini")
    executable = args.build / "apps/oskar_sim_interferometer"
    extended = np.loadtxt(args.fixtures / "sky.osm")
    extended = np.column_stack((extended, np.zeros((len(extended), 4))))
    extended[:, 9:11] = 60.0
    extended[:, 11] = np.nan
    np.savetxt(root / "extended.osm", extended)
    bad_telescope = root / "undefined_element.tm"
    shutil.copytree(args.fixtures / "telescope.tm", bad_telescope, dirs_exist_ok=True)
    np.savetxt(bad_telescope / "station000/element_types.txt", [0, 1, 2], fmt="%d")
    failures = {
        "undefined_element": ("telescope", "input_directory", str(bad_telescope)),
        "no_eop": ("interferometer", "astrometry/eop_file", str(root / "missing.txt")),
        "out_of_range": ("observation", "start_time_utc", "31-05-2019 06:00:00.000"),
        "legacy_icrs": ("interferometer", "astrometry/mode", "legacy"),
        "beam_duplication": ("telescope", "allow_station_beam_duplication", "true"),
        "ignore_w": ("interferometer", "ignore_w_components", "true"),
        "ionosphere": ("telescope", "ionosphere_screen_type", "External"),
        "nonfinite_gaussian_angle": ("sky", "oskar_sky_model/file", str(root / "extended.osm")),
        "zero_time_samples": ("interferometer", "astrometry/time_samples", "0"),
        "huge_sample_count": ("interferometer", "astrometry/time_samples", "1025"),
        "bad_frequency": ("observation", "start_frequency_hz", "0"),
        "nan_bandwidth": ("interferometer", "channel_bandwidth_hz", "nan"),
    }
    for name, (section, key, value) in failures.items():
        config = copy.deepcopy(fixture)
        config[section][key] = value
        execute(executable, config, root / name, False)
    for axis, width in (
        ("time", "time_average_sec"),
        ("frequency", "channel_bandwidth_hz"),
    ):
        config = copy.deepcopy(fixture)
        config["interferometer"][f"astrometry/{axis}_samples"] = "1"
        config["interferometer"][width] = "10"
        execute(executable, config, root / (axis + "_undersampled"), False)
    leap_table = root / "leap_eop.txt"
    leap_table.write_text("57753 -0.4 0 0 36\n57754 0.6 0 0 37\n57755 0.6 0 0 37\n")
    config = copy.deepcopy(fixture)
    config["interferometer"]["astrometry/eop_file"] = str(leap_table)
    config["observation"]["start_time_utc"] = "31-12-2016 23:59:59.000"
    config["observation"]["length"] = "6"
    execute(executable, config, root / "crosses_leap", False)
    records = []
    reader = Reader(args.build / "oskar/liboskar.so")
    for name in (
        "apparent_scalar",
        "icrs_aa_full",
        "icrs_integrated",
        "icrs_chunks_workers",
        "gaussian_icrs",
    ):
        config = configparser.ConfigParser(interpolation=None)
        config.read(args.fixtures / name / "settings.ini")
        for key in list(config["interferometer"]):
            if key.startswith("astrometry/"):
                del config["interferometer"][key]
        products = []
        for label, build in (
            ("patched", args.build),
            ("pristine", args.reference_build),
        ):
            directory = root / ("legacy_" + name + "_" + label)
            execute(
                build / "apps/oskar_sim_interferometer",
                copy.deepcopy(config),
                directory,
                True,
            )
            data, uvw, _, frame, _ = reader.read(directory / "result.vis", name != "apparent_scalar")
            assert frame == 0
            products.append((data, uvw))
        difference = float(np.max(np.abs(products[0][0] - products[1][0])))
        if name == "icrs_chunks_workers":
            # Multiworker chunk reduction has a nondeterministic addition order.
            np.testing.assert_allclose(products[0][0], products[1][0], rtol=0, atol=2e-15)
        else:
            np.testing.assert_array_equal(products[0][0], products[1][0])
        np.testing.assert_array_equal(products[0][1], products[1][1])
        records.append({"case": name, "data_max_abs_jy": difference, "uvw_bitwise_equal": True})
    report = {"preflight_rejections": len(failures) + 3, "legacy": records}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("build", "fixtures", "output", "reference-build"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.build = args.build.resolve(strict=True)
    args.reference_build = args.reference_build.resolve(strict=True)
    run(args)
