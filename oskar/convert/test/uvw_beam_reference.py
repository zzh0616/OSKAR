"""Offline FEKO/HDF5 and station-pointing acceptance for precision astrometry.

The reference computes directions with PyERFA, feeds explicit ENU coordinates
and AZ/EL steering to the legacy station-beam API, and forms visibility directly
from ITRS baselines. It deliberately shares the FEKO electromagnetic kernel:
this tests geometry integration, not independent FEKO model fidelity.
Synthetic coefficients are regression fixtures, not a physical SKA beam model.
"""

import argparse
import configparser
import ctypes as ct
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import erfa
import h5py
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.utils import iers
from uvw_simulation_reference import Reader, directions, enu_matrix, expected
from uvw_simulation_reference import run as make_fixtures

DOUBLE, MATRIX, CPU, ENU, RADEC, AZEL = 8, 104, 0, 1, 2, 4


def horizon_angles(p):
    return np.arctan2(p[0], p[1]), np.arctan2(p[2], np.hypot(p[0], p[1]))


class Beam(Reader):
    """Small explicit bindings: no installed OSKAR Python package is needed."""

    def __init__(self, library, telescope, case):
        super().__init__(library)
        p, i, d, s = ct.c_void_p, ct.c_int, ct.c_double, ct.POINTER(ct.c_int)
        pp = ct.POINTER(p)
        self.status = ct.c_int(0)
        self.tel, self.work, self.astrometry = None, None, None
        self.memory = []
        signatures = {
            "oskar_telescope_create": (p, [i, i, i, s]),
            "oskar_telescope_free": (None, [p, s]),
            "oskar_telescope_load": (None, [p, ct.c_char_p, p, s]),
            "oskar_telescope_set_enable_numerical_patterns": (None, [p, i]),
            "oskar_telescope_set_station_type": (None, [p, ct.c_char_p, s]),
            "oskar_telescope_set_pol_mode": (None, [p, ct.c_char_p, s]),
            "oskar_telescope_analyse": (None, [p, s]),
            "oskar_telescope_station": (p, [p, i]),
            "oskar_telescope_num_station_models": (i, [p]),
            "oskar_station_work_create": (p, [i, i, s]),
            "oskar_station_work_free": (None, [p, s]),
            "oskar_station_work_set_astrometry": (None, [p, p]),
            "oskar_station_set_phase_centre": (None, [p, i, d, d]),
            "oskar_station_set_normalise_array_pattern": (None, [p, i]),
            "oskar_station_set_normalise_element_pattern": (None, [p, i]),
            "oskar_station_set_normalise_final_beam": (None, [p, i]),
            "oskar_station_set_enable_array_pattern": (None, [p, i]),
            "oskar_station_set_swap_xy": (None, [p, i]),
            "oskar_station_set_seed_time_variable_errors": (None, [p, ct.c_uint]),
            "oskar_station_has_child": (i, [p]),
            "oskar_station_num_elements": (i, [p]),
            "oskar_station_child": (p, [p, i]),
            "oskar_station_beam": (None, [p, p, i, i, pp, d, d, i, d, d, i, d, d, d, i, p, s]),
            "oskar_mem_create": (p, [i, i, ct.c_size_t, s]),
            "oskar_mem_free": (None, [p, s]),
            "oskar_mem_ensure": (None, [p, ct.c_size_t, s]),
            "oskar_astrometry_create": (p, [i, ct.c_char_p, s]),
            "oskar_astrometry_free": (None, [p]),
            "oskar_astrometry_state_create": (p, [p, d, d, d, d, s]),
            "oskar_astrometry_state_free": (None, [p]),
        }
        for name, (result, args) in signatures.items():
            self.bind(name, result, args)
        lib = self.lib
        try:
            self.tel = lib.oskar_telescope_create(DOUBLE, CPU, 0, self.err)
            self.check()
            lib.oskar_telescope_set_enable_numerical_patterns(self.tel, 1)
            lib.oskar_telescope_load(self.tel, os.fsencode(telescope), None, self.err)
            self.check()
            if lib.oskar_telescope_num_station_models(self.tel) != 3:
                raise ValueError("The beam acceptance fixture requires three station models")
            lib.oskar_telescope_set_station_type(self.tel, b"Aperture array", self.err)
            lib.oskar_telescope_set_pol_mode(self.tel, b"Full", self.err)
            self.stations = [lib.oskar_telescope_station(self.tel, j) for j in range(3)]
            for station in self.stations:
                self.configure(station, case)
            lib.oskar_telescope_analyse(self.tel, self.err)
            self.work = lib.oskar_station_work_create(DOUBLE, CPU, self.err)
            self.inputs = [self.alloc(DOUBLE, 3) for _ in range(3)]
            self.output = self.alloc(MATRIX, 3)
            self.check()
        except Exception:
            self.close()
            raise

    @property
    def err(self):
        return ct.byref(self.status)

    def check(self):
        if self.status.value:
            raise RuntimeError(f"Native beam status {self.status.value}")

    def alloc(self, kind, n):
        result = self.lib.oskar_mem_create(kind, CPU, n, self.err)
        self.memory.append(result)
        self.check()
        return result

    def configure(self, station, case):
        lib = self.lib
        lib.oskar_station_set_normalise_array_pattern(station, 1)
        lib.oskar_station_set_enable_array_pattern(station, 1)
        lib.oskar_station_set_normalise_element_pattern(station, int(case.get("element_norm", False)))
        lib.oskar_station_set_normalise_final_beam(station, int(case.get("normalise", False)))
        lib.oskar_station_set_swap_xy(station, int(case.get("swap", False)))
        lib.oskar_station_set_seed_time_variable_errors(station, 1)
        if lib.oskar_station_has_child(station):
            for j in range(lib.oskar_station_num_elements(station)):
                self.configure(lib.oskar_station_child(station, j), case)

    def set_pointing(self, station, kind, angles, recursive=True):
        self.lib.oskar_station_set_phase_centre(station, kind, *angles)
        if recursive and self.lib.oskar_station_has_child(station):
            for j in range(self.lib.oskar_station_num_elements(station)):
                self.set_pointing(self.lib.oskar_station_child(station, j), kind, angles)

    def evaluate(self, station, coords, time, freq, norm_angles, kind=ENU, norm_kind=AZEL):
        for k in range(3):
            self.lib.oskar_mem_ensure(self.inputs[k], len(coords), self.err)
            self.check()
            a = np.ctypeslib.as_array(self.lib.oskar_mem_double(self.inputs[k], self.err), shape=(len(coords),))
            a[:] = coords[:, k]
        self.lib.oskar_mem_ensure(self.output, len(coords), self.err)
        self.lib.oskar_station_beam(
            station,
            self.work,
            kind,
            len(coords),
            (ct.c_void_p * 3)(*self.inputs),
            0.0,
            0.0,
            norm_kind,
            *norm_angles,
            0,
            59000.25,
            float(time.utc.mjd),
            freq,
            0,
            self.output,
            self.err,
        )
        self.check()
        return self.array(self.output, len(coords) * 8, True).reshape(-1, 2, 2)

    def close(self):
        clean = ct.c_int(0)
        if self.work:
            self.lib.oskar_station_work_set_astrometry(self.work, None)
            self.lib.oskar_station_work_free(self.work, ct.byref(clean))
        for memory in self.memory:
            self.lib.oskar_mem_free(memory, ct.byref(clean))
        if self.tel:
            self.lib.oskar_telescope_free(self.tel, ct.byref(clean))
        if self.astrometry:
            self.lib.oskar_astrometry_free(self.astrometry)
        self.work, self.tel, self.astrometry, self.memory = None, None, None, []


def write_patterns(path, real_file=None):
    provenance = []
    with h5py.File(path, "w") as target:
        if real_file:
            with h5py.File(real_file, "r") as source:
                # Only three embedded elements; no full-station production run.
                for antenna in range(1, 4):
                    for pol in "XY":
                        key = f"{pol}{antenna}_100000000"
                        data = source[key][...]
                        target[key] = data
                        provenance.append(
                            {
                                "dataset": key,
                                "shape": list(data.shape),
                                "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
                            }
                        )
        else:
            # l_max=2: eight (l,m) modes, interleaved TE/TM amplitudes and phases.
            for antenna in range(1, 4):
                for pi, pol in enumerate("XY"):
                    for fi, freq in enumerate((100000000, 120000000)):
                        k = np.arange(16)
                        data = np.array(
                            [
                                (1 + 0.11 * antenna + 0.09 * pi + 0.07 * fi) / (1 + k),
                                (13 * k + 19 * antenna + 31 * pi + 7 * fi) % 360,
                            ]
                        )
                        target[f"{pol}{antenna}_{freq}"] = data
    return provenance


def run(args):
    iers.conf.auto_download = False
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Never leave an earlier successful summary looking current after a failure.
    (root / "report.json").write_text(json.dumps({"status": "running", "scope": __doc__}) + "\n")
    fixture = args.fixtures.resolve() if args.fixtures else root / "base"
    if not args.fixtures:
        make_fixtures(argparse.Namespace(build=args.build, output=fixture, ms=False))
    eop = np.loadtxt(fixture / "eop.txt")
    template = configparser.ConfigParser(interpolation=None)
    template.read(fixture / "icrs_aa_full/settings.ini")
    phase = np.deg2rad(
        [float(template["observation"]["phase_centre_ra_deg"]), float(template["observation"]["phase_centre_dec_deg"])]
    )
    osm = np.loadtxt(fixture / "sky.osm")
    sky = np.column_stack((np.deg2rad(osm[:, :2]), osm[:, 2:6], osm[:, 7]))
    pointing = phase + np.array([[0.02, -0.01], [-0.03, 0.02], [0.01, 0.025]])
    cases = [
        {"name": "feko_icrs"},
        {"name": "feko_apparent", "frame": "apparent"},
        {"name": "feko_fk5", "frame": "fk5_j2000"},
        {"name": "feko_offset", "offset": True},
        {"name": "feko_normalised", "offset": True, "normalise": True, "element_norm": True},
        {"name": "feko_swap", "offset": True, "swap": True, "casa": False},
        {"name": "feko_station_azel", "offset": True, "station_azel": True},
        {"name": "feko_hierarchy", "offset": True, "hierarchy": True},
        {"name": "feko_permitted", "offset": True, "permitted": True},
        {"name": "feko_gaussian", "offset": True, "gaussian": True},
        {"name": "feko_integrated", "offset": True, "dt": 10.0, "df": 50000.0, "nt": 8, "nf": 8},
        {"name": "feko_integrated16", "offset": True, "dt": 10.0, "df": 50000.0, "nt": 16, "nf": 16},
        {"name": "feko_chunks", "offset": True, "workers": 2, "chunk": 1},
        {"name": "analytic_control", "numerical": False},
    ]
    if args.real_feko:
        cases += [
            {"name": "real_feko_offset", "offset": True, "real": True},
            {"name": "real_feko_normalised", "offset": True, "normalise": True, "element_norm": True, "real": True},
        ]
    library = args.build / "oskar/liboskar.so"
    reader = Reader(library)
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    env.pop("LD_LIBRARY_PATH", None)
    records, products, provenance = [], {}, []
    telescopes = {}
    for label in ("synthetic", "hierarchy", "permitted", "analytic") + (("real",) if args.real_feko else ()):
        telescope = root / (label + ".tm")
        shutil.copytree(fixture / "telescope.tm", telescope, dirs_exist_ok=True)
        rows = write_patterns(telescope / "acceptance_FEKO.h5", args.real_feko if label == "real" else None)
        provenance.extend(rows)
        for index in range(3):
            station = telescope / f"station{index:03d}"
            if label == "hierarchy":
                np.savetxt(station / "layout.txt", [[0.0, 0.0, 0.0], [12.0, 8.0, 1.0]])
                for tile in range(2):
                    child = station / f"tile{tile}"
                    child.mkdir(exist_ok=True)
                    shutil.copyfile(fixture / f"telescope.tm/station{index:03d}/layout.txt", child / "layout.txt")
                    np.savetxt(child / "element_types.txt", [0, 1, 2], fmt="%d")
            else:
                np.savetxt(station / "element_types.txt", [0, 0, 0] if label == "analytic" else [0, 1, 2], fmt="%d")
            if label == "permitted":
                np.savetxt(station / "permitted_beams.txt", [[30.0, 80.0], [120.0, 75.0], [250.0, 70.0]])
        telescopes[label] = telescope

    for case in cases:
        case.update(full=True, aa=True)
        directory = root / case["name"]
        directory.mkdir(exist_ok=True)
        label = next((name for name in ("real", "hierarchy", "permitted") if case.get(name)), "synthetic")
        if not case.get("numerical", True):
            label = "analytic"
        telescope = telescopes[label]
        config = configparser.ConfigParser(interpolation=None)
        config.read_dict(template)
        case_sky = sky
        if case.get("gaussian"):
            sky_path = fixture / "gaussian_aa/sky.osm"
            config["sky"]["oskar_sky_model/file"] = str(sky_path)
            rows = np.loadtxt(sky_path)
            case_sky = np.column_stack((np.deg2rad(rows[:, :2]), rows[:, 2:6], rows[:, 7]))
        config["simulator"]["num_devices"] = str(case.get("workers", 1))
        config["simulator"]["max_sources_per_chunk"] = str(case.get("chunk", 64))
        tel = config["telescope"]
        tel["input_directory"] = str(telescope)
        tel["aperture_array/element_pattern/functional_type"] = "Dipole"
        tel["aperture_array/element_pattern/enable_numerical"] = str(case.get("numerical", True)).lower()
        tel["aperture_array/element_pattern/normalise"] = str(case.get("element_norm", False)).lower()
        tel["aperture_array/element_pattern/swap_xy"] = str(case.get("swap", False)).lower()
        tel["normalise_beams_at_phase_centre"] = str(case.get("normalise", False)).lower()
        settings = config["interferometer"]
        settings["astrometry/frame"] = case.get("frame", "icrs")
        settings["time_average_sec"] = str(case.get("dt", 0))
        settings["channel_bandwidth_hz"] = str(case.get("df", 0))
        settings["astrometry/time_samples"] = str(case.get("nt", 8))
        settings["astrometry/frequency_samples"] = str(case.get("nf", 8))
        settings["use_casa_phase_convention"] = str(case.get("casa", True)).lower()
        settings["oskar_vis_filename"] = str(directory / "result.vis")
        settings.pop("ms_filename", None)
        if case.get("offset"):
            lines = [f"{j} RADEC {np.rad2deg(p[0]):.17g} {np.rad2deg(p[1]):.17g}" for j, p in enumerate(pointing)]
            if case.get("station_azel"):
                lines[1] = "1 AZEL 45 80"
            if case.get("hierarchy"):
                lines.append("* * AZEL 45 80")
            (directory / "pointing.txt").write_text("\n".join(lines) + "\n")
            config["observation"]["pointing_file"] = str(directory / "pointing.txt")
        with (directory / "settings.ini").open("w") as stream:
            config.write(stream, space_around_delimiters=False)
        with (directory / "run.log").open("w") as log:
            subprocess.run(
                [str(args.build / "apps/oskar_sim_interferometer"), str(directory / "settings.ini")],
                cwd=directory,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=180,
            )
        assert "WARNING: Ignoring '" not in (directory / "run.log").read_text()
        actual, uvw, xyz, _, _ = reader.read(directory / "result.vis", True)
        products[case["name"]] = actual
        if not case.get("numerical", True):
            continue
        lon, lat, alt = np.loadtxt(telescope / "position.txt")
        centre = erfa.gd2gc(1, *np.deg2rad([lon, lat]), alt)
        geo = np.array([erfa.gc2gd(1, centre + p) for p in xyz])
        beam = Beam(library, telescope, case)
        try:

            def evaluate(time, freq, p, phase_dir, station_geo):
                directions_itrs = directions(
                    time, pointing if case.get("offset") else [phase] * 3, case.get("frame", "icrs"), eop
                )
                result = []
                for j, station in enumerate(beam.stations):
                    enu = enu_matrix(*station_geo[j, :2])
                    beam_angles = horizon_angles(enu @ directions_itrs[j])
                    if case.get("station_azel") and j == 1:
                        beam_angles = np.deg2rad([45.0, 80.0])
                    beam.set_pointing(station, AZEL, beam_angles)
                    if case.get("hierarchy"):
                        for tile in range(2):
                            beam.set_pointing(
                                beam.lib.oskar_station_child(station, tile), AZEL, np.deg2rad([45.0, 80.0])
                            )
                    result.append(beam.evaluate(station, p @ enu.T, time, freq, horizon_angles(enu @ phase_dir)))
                return np.array(result)

            predicted, uvw_ref = expected(case, xyz, eop, case_sky, phase, geo, None, beam_evaluator=evaluate)
        finally:
            beam.close()
        error = float(np.max(np.abs(actual - predicted)))
        np.testing.assert_allclose(actual, predicted, rtol=2e-8, atol=2e-8)
        np.testing.assert_allclose(uvw, uvw_ref, rtol=0, atol=3e-5)
        record = {
            "case": case["name"],
            "max_abs_jy": error,
            "normalised_max_error": error / float(np.max(np.abs(predicted))),
        }
        records.append(record)
        np.savez_compressed(directory / "comparison.npz", actual=actual, reference=predicted, uvw=uvw)
        print(record, flush=True)
    controls = {}
    for left, right in (
        ("feko_icrs", "analytic_control"),
        ("feko_icrs", "feko_offset"),
        ("feko_offset", "feko_station_azel"),
    ):
        difference = float(np.max(np.abs(products[left] - products[right])))
        assert difference > 1e-4, (left, right, difference)
        controls[left + "_vs_" + right] = difference
    np.testing.assert_allclose(products["feko_offset"], products["feko_chunks"], rtol=0, atol=2e-12)
    convergence = float(np.max(np.abs(products["feko_integrated"] - products["feko_integrated16"])))
    assert convergence < 2e-8, convergence
    moving = check_moving_api(library, telescopes["synthetic"], fixture / "eop.txt", eop, phase, sky, geo)
    legacy = check_legacy(args, root, reader, env) if args.reference_build else None
    report = {
        "status": "passed",
        "scope": __doc__,
        "cases": records,
        "controls": controls,
        "quadrature_8_vs_16_max_jy": convergence,
        "moving_api": moving,
        "real_feko": str(args.real_feko.resolve()) if args.real_feko else None,
        "real_datasets": provenance,
        "legacy": legacy,
        "library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def check_moving_api(library, telescope, eop_path, eop, phase, sky, geo):
    records = []
    for frame_id, frame in enumerate(("apparent", "icrs", "fk5_j2000"), start=1):
        precise = Beam(library, telescope, {"normalise": True})
        reference = None
        try:
            reference = Beam(library, telescope, {"normalise": True})
            precise.astrometry = precise.lib.oskar_astrometry_create(frame_id, os.fsencode(eop_path), precise.err)
            precise.check()
            outputs, error = [], 0.0
            for index, offset in enumerate((900.0, 2700.0, 4500.0, 900.0)):
                time = Time(59000.25, format="mjd", scale="utc") + offset * u.s
                state = precise.lib.oskar_astrometry_state_create(
                    precise.astrometry, 59000.25, offset, *phase, precise.err
                )
                precise.check()
                try:
                    precise.lib.oskar_station_work_set_astrometry(precise.work, state)
                    # Revisit the first time/pointing to detect stale work caches.
                    step = index % 3
                    pointing = phase + np.array([0.01 * step, -0.015 * step])
                    enu = enu_matrix(*geo[0, :2])
                    points = directions(time, sky[:, :2], frame, eop) @ enu.T
                    centre = directions(time, [phase], frame, eop)[0] @ enu.T
                    beam_dir = directions(time, [pointing], frame, eop)[0] @ enu.T
                    precise.set_pointing(precise.stations[0], RADEC, pointing)
                    reference.set_pointing(reference.stations[0], AZEL, horizon_angles(beam_dir))
                    actual = precise.evaluate(
                        precise.stations[0],
                        np.column_stack((sky[:, :2], np.zeros(len(sky)))),
                        time,
                        110e6,
                        phase,
                        RADEC,
                        RADEC,
                    )
                    wanted = reference.evaluate(reference.stations[0], points, time, 110e6, horizon_angles(centre))
                    error = max(error, float(np.max(np.abs(actual - wanted))))
                    np.testing.assert_allclose(actual, wanted, rtol=0, atol=2e-10)
                    outputs.append(actual)
                finally:
                    precise.lib.oskar_station_work_set_astrometry(precise.work, None)
                    precise.lib.oskar_astrometry_state_free(state)
            np.testing.assert_array_equal(outputs[0], outputs[-1])
            change = float(np.max(np.abs(outputs[0] - outputs[1])))
            assert change > 1e-4
            record = {
                "frame": frame,
                "max_abs_jones_error": error,
                "response_change": change,
                "revisited_pointing_bitwise_equal": True,
            }
            records.append(record)
            print(record, flush=True)
        finally:
            precise.close()
            if reference:
                reference.close()
    return records


def check_legacy(args, root, reader, env):
    records = []
    for name in ("feko_offset", "feko_normalised", "feko_hierarchy", "feko_gaussian"):
        config = configparser.ConfigParser(interpolation=None)
        config.read(root / name / "settings.ini")
        for key in list(config["interferometer"]):
            if key.startswith("astrometry/"):
                del config["interferometer"][key]
        results = []
        for label, build in (("patched", args.build), ("pristine", args.reference_build)):
            directory = root / f"legacy_{name}_{label}"
            directory.mkdir(exist_ok=True)
            config["interferometer"]["oskar_vis_filename"] = str(directory / "result.vis")
            with (directory / "settings.ini").open("w") as stream:
                config.write(stream, space_around_delimiters=False)
            with (directory / "run.log").open("w") as log:
                subprocess.run(
                    [str(build / "apps/oskar_sim_interferometer"), str(directory / "settings.ini")],
                    cwd=directory,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=180,
                )
            data, uvw, _, frame, _ = reader.read(directory / "result.vis", True)
            assert frame == 0
            results.append((data, uvw))
        np.testing.assert_array_equal(results[0][0], results[1][0])
        np.testing.assert_array_equal(results[0][1], results[1][1])
        records.append({"case": name, "visibility_and_uvw_bitwise_equal": True})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fixtures", type=Path, help="Reuse uvw_simulation_reference.py fixtures")
    parser.add_argument("--real-feko", type=Path, help="Optional real X1..3/Y1..3 100 MHz FEKO file")
    parser.add_argument("--reference-build", type=Path, help="Optional pristine OSKAR 2.13 CPU/HDF5 build")
    args = parser.parse_args()
    args.build = args.build.resolve(strict=True)
    if args.reference_build:
        args.reference_build = args.reference_build.resolve(strict=True)
    try:
        run(args)
    except Exception as exc:
        if args.output.is_dir():
            (args.output / "report.json").write_text(json.dumps({"status": "failed", "error": str(exc)}) + "\n")
        raise
