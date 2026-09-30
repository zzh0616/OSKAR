"""Offline CUDA/HDF5/MS output-contract acceptance using existing fixtures.

MS reading runs in a separate process so Python bindings cannot change the
native simulator's casacore/HDF5 library resolution. This is not downstream
predictor compatibility or an absolute astrometric accuracy certificate.
"""

import argparse
import configparser
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def inspect_ms(path):
    import casacore
    from casacore.tables import table

    arrays, metadata = {}, {"python_casacore_version": casacore.__version__}
    with table(str(path), ack=False) as ms:
        for column in (
            "DATA", "UVW", "TIME", "TIME_CENTROID", "INTERVAL", "EXPOSURE",
            "ANTENNA1", "ANTENNA2", "FLAG", "FLAG_ROW", "WEIGHT", "SIGMA",
        ):
            arrays[column] = ms.getcol(column)
        metadata["keywords"] = {k: v for k, v in ms.getkeywords().items() if k.startswith("OSKAR_")}
        metadata["direction_refs"] = {}
        for sub, columns in (
            ("FIELD", ["PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"]),
            ("POINTING", ["DIRECTION", "TARGET"]), ("SOURCE", ["DIRECTION"]),
        ):
            with table(str(path / sub), ack=False) as tab:
                for column in columns:
                    key = sub + "/" + column
                    metadata["direction_refs"][key] = tab.getcolkeyword(column, "MEASINFO")["Ref"]
                    if tab.nrows():
                        arrays[key] = tab.getcol(column)
    with table(str(path / "SPECTRAL_WINDOW"), ack=False) as tab:
        for column in ("CHAN_FREQ", "CHAN_WIDTH", "EFFECTIVE_BW", "RESOLUTION", "TOTAL_BANDWIDTH"):
            arrays[column] = np.asarray(tab.getcell(column, 0))
    with table(str(path / "POLARIZATION"), ack=False) as tab:
        arrays["CORR_TYPE"] = tab.getcell("CORR_TYPE", 0)
    maps = Path("/proc/self/maps")
    if maps.exists():
        metadata["reader_libraries"] = sorted({
            line.split()[-1] for line in maps.read_text().splitlines()
            if "libcasa_" in line or "libhdf5" in line
        })
    np.savez_compressed(path.parent / "ms_readback.npz", **arrays)
    (path.parent / "ms_readback.json").write_text(json.dumps(metadata, indent=2) + "\n")


def run_case(args, fixture, label, mode, devices, write_ms=True):
    from uvw_simulation_reference import Reader

    target = args.output / label
    # Do not accidentally append rows to a previous successful Measurement Set.
    target.mkdir()
    config = configparser.ConfigParser(interpolation=None)
    config.read(fixture / "settings.ini")
    sim, settings = config["simulator"], config["interferometer"]
    sim["use_gpus"] = str(devices is not None).lower()
    sim["double_precision"] = "true"
    sim["num_devices"] = str(len(devices.split(","))) if devices else "1"
    if devices:
        sim["cuda_device_ids"] = devices
        if "," in devices:
            sim["max_sources_per_chunk"] = "1"
    settings["astrometry/mode"] = mode
    if mode == "legacy":
        for key in list(settings):
            if key.startswith("astrometry/") and key != "astrometry/mode":
                del settings[key]
    settings["oskar_vis_filename"] = str(target / "result.vis")
    settings.pop("ms_filename", None)
    if write_ms:
        settings["ms_filename"] = str(target / "result.ms")
    with (target / "settings.ini").open("w") as stream:
        config.write(stream, space_around_delimiters=False)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("LD_LIBRARY_PATH", None)
    native_env = {"PATH": os.defpath, "HOME": str(target), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    if "CUDA_VISIBLE_DEVICES" in env:
        native_env["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"]
    with (target / "run.log").open("w") as log:
        subprocess.run(
            [str(args.binary), str(target / "settings.ini")],
            cwd=target, env=native_env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300,
        )
    log = (target / "run.log").read_text()
    assert "WARNING: Ignoring '" not in log
    full = config["telescope"]["pol_mode"].lower() == "full"
    result = Reader(args.library).read(target / "result.vis", full)
    assert np.isfinite(result[0]).all() and np.isfinite(result[1]).all()
    record = {"case": label}
    if write_ms:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--inspect-ms", str(target / "result.ms")],
            cwd=target, env=env, check=True, timeout=60,
        )
        record.update(check_ms(target, config, result, mode))
    return result, record


def check_ms(target, config, result, mode):
    from astropy.time import Time

    vis, uvw, xyz, frame, eop = result
    nt, nc, nb, npol = vis.shape
    meta = json.loads((target / "ms_readback.json").read_text())
    with np.load(target / "ms_readback.npz") as ms:
        data = ms["DATA"].reshape(nt, nb, nc, -1).transpose(0, 2, 1, 3)
        assert data.dtype == np.complex64 and ms["UVW"].dtype == np.float64
        expected = vis if data.shape[-1] == npol else np.concatenate(
            (vis, np.zeros_like(vis), np.zeros_like(vis), vis), axis=-1,
        )
        # Exact round-to-complex64 checks isolate storage quantisation from
        # simulation errors, including XY/YX ordering and scalar expansion.
        np.testing.assert_array_equal(data, expected.astype(np.complex64))
        np.testing.assert_array_equal(ms["UVW"].reshape(nt, nb, 3), uvw)
        pairs = [(a, b) for a in range(len(xyz)) for b in range(a + 1, len(xyz))]
        np.testing.assert_array_equal(ms["ANTENNA1"], np.tile(np.array(pairs)[:, 0], nt))
        np.testing.assert_array_equal(ms["ANTENNA2"], np.tile(np.array(pairs)[:, 1], nt))
        # The existing OSKAR MS writer labels scalar output XX, not Stokes I.
        np.testing.assert_array_equal(ms["CORR_TYPE"], [9, 10, 11, 12] if data.shape[-1] == 4 else [9])
        assert not ms["FLAG"].any() and not ms["FLAG_ROW"].any()
        assert np.isfinite(ms["WEIGHT"]).all() and (ms["WEIGHT"] > 0).all()
        assert np.isfinite(ms["SIGMA"]).all() and (ms["SIGMA"] > 0).all()
        obs, settings = config["observation"], config["interferometer"]
        step = float(obs["length"]) / nt
        start = Time(datetime.strptime(obs["start_time_utc"], "%d-%m-%Y %H:%M:%S.%f"), scale="utc")
        times = np.repeat(start.mjd * 86400 + (np.arange(nt) + 0.5) * step, nb)
        np.testing.assert_allclose(ms["TIME"], times, rtol=0, atol=2e-6)
        np.testing.assert_array_equal(ms["TIME"], ms["TIME_CENTROID"])
        np.testing.assert_array_equal(ms["INTERVAL"], step)
        np.testing.assert_array_equal(ms["EXPOSURE"], float(settings["time_average_sec"]))
        spacing = float(obs["frequency_inc_hz"])
        frequencies = float(obs["start_frequency_hz"]) + np.arange(nc) * spacing
        np.testing.assert_array_equal(ms["CHAN_FREQ"], frequencies)
        if mode == "precision":
            ref = {1: "APP", 2: "ICRS", 3: "J2000"}[frame]
            assert meta["keywords"]["OSKAR_ASTROMETRY_FRAME"] == ref
            assert meta["keywords"]["OSKAR_ASTROMETRY_EOP"] == eop.decode("ascii")
            assert "transported catalogue north" in meta["keywords"]["OSKAR_UVW_BASIS"]
            width = float(settings["channel_bandwidth_hz"])
            assert meta["keywords"]["OSKAR_ASTROMETRY_CHANNEL_WIDTH_HZ"] == width
            if width > 0:
                for column in ("CHAN_WIDTH", "EFFECTIVE_BW", "RESOLUTION"):
                    np.testing.assert_array_equal(ms[column], width)
                np.testing.assert_array_equal(ms["TOTAL_BANDWIDTH"], np.ptp(frequencies) + width)
        else:
            ref = "J2000"
            assert not any(k.startswith("OSKAR_ASTROMETRY") for k in meta["keywords"])
            assert frame == 0 and not eop
        assert all(value == ref for value in meta["direction_refs"].values())
        phase = np.deg2rad([float(obs["phase_centre_ra_deg"]), float(obs["phase_centre_dec_deg"])])
        for column in ("PHASE_DIR", "DELAY_DIR", "REFERENCE_DIR"):
            np.testing.assert_allclose(ms["FIELD/" + column].reshape(-1, 2), [phase], rtol=0, atol=1e-14)
        return {"ms_quantisation_max_jy": float(np.max(np.abs(data - expected))),
                "ms_data_exact_complex64_cast": True, "ms_uvw_bitwise_equal": True}


def compare(actual, expected):
    from uvw_gpu_reference import compare as compare_gpu

    record = compare_gpu("full-stack", actual, expected, 0)
    return {key: record[key] for key in ("max_abs_jy", "normalised_max_error")}


def run(args):
    from uvw_simulation_reference import Reader

    args.build = args.build.resolve(strict=True)
    built = (args.build / "oskar/liboskar.so").exists()
    args.library = args.build / ("oskar/liboskar.so" if built else "lib/liboskar.so")
    args.binary = args.build / ("apps/oskar_sim_interferometer" if built else "bin/oskar_sim_interferometer")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise ValueError("Use a fresh output directory; existing acceptance products are never overwritten")
    report = {"status": "running", "scope": __doc__, "cases": [], "comparisons": [],
              "binary": str(args.binary), "native_environment": "minimal; fresh HOME per case; no LD_LIBRARY_PATH"}
    try:
        fixtures = []
        for root in args.fixtures:
            fixtures.extend(sorted(root.resolve(strict=True).glob("*/settings.ini")))
        for path in fixtures:
            if path.parent.name not in (
                "icrs_aa_full", "apparent_scalar", "gaussian_fk5", "feko_icrs", "feko_apparent",
                "feko_fk5", "feko_gaussian", "feko_integrated", "feko_station_azel",
                "real_feko_offset", "real_feko_normalised",
            ):
                continue
            name = path.parent.name
            for mode in ("legacy", "precision"):
                label = name + "_" + mode
                cpu, record = run_case(args, path.parent, label + "_cpu", mode, None)
                report["cases"].append(record)
                gpu, record = run_case(args, path.parent, label + "_cuda", mode, args.gpu)
                record.update(compare(gpu, cpu))
                report["cases"].append(record)
                if mode == "precision":
                    config = configparser.ConfigParser(interpolation=None)
                    config.read(path)
                    full = config["telescope"]["pol_mode"].lower() == "full"
                    reference = Reader(args.library).read(path.parent / "result.vis", full)
                    report["comparisons"].append({"case": label + "_original_fixture", **compare(cpu, reference)})
                if name in ("feko_gaussian", "real_feko_offset"):
                    output, record = run_case(args, path.parent, label + "_no_ms", mode, args.gpu, False)
                    np.testing.assert_array_equal(output[0], gpu[0])
                    np.testing.assert_array_equal(output[1], gpu[1])
                    report["comparisons"].append({"case": record["case"], "ms_toggle_bitwise_equal": True})
                    if args.multi_gpu:
                        output, record = run_case(args, path.parent, label + "_multi", mode, args.multi_gpu)
                        record.update(compare(output, cpu))
                        report["cases"].append(record)
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        assert report["cases"], "No matching fixtures"
        report["library_sha256"] = hashlib.sha256(args.library.read_bytes()).hexdigest()
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-ms", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--build", type=Path)
    parser.add_argument("--fixtures", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--multi-gpu")
    args = parser.parse_args()
    if args.inspect_ms:
        inspect_ms(args.inspect_ms)
    elif not all((args.build, args.fixtures, args.output)):
        parser.error("--build, --fixtures and --output are required")
    else:
        run(args)
