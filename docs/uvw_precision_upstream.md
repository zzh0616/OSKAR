# Draft upstream issue: explicit celestial frames and offline precision geometry

This is an upstream proposal draft; no upstream issue or merge request has been filed.

## Scope and motivation

The legacy simulator uses fast sidereal rotation with apparent coordinates.
A mean FK5 J2000 or ICRS catalogue cannot be made consistent by correcting only
the output UVW: source phase, station steering, horizon selection and the
polarization basis must use the same frame and Earth orientation.

Propose an opt-in `interferometer/astrometry/mode=precision` with explicit
`apparent`, `icrs` and `fk5_j2000` frames and a local, immutable EOP snapshot.
The default remains legacy. Neither simulation nor EOP export downloads data.
Missing coverage and unsupported combinations fail explicitly.

## Proposed review units

1. CPU foundation and consistent interferometer integration: optional ERFA,
   coordinate conversion, local EOP contract, beam/phase/polarization geometry,
   compact Gaussian morphology, numerical time/frequency averaging, metadata
   and offline reference/regression tests.
2. CUDA implementation of the same measurement equation, with host ERFA state
   preparation, batched transfers and CPU/CUDA equivalence/performance tests.
3. Separate extensions for standalone beam maps, AZ/EL phase tracking and
   array-level HARP. These must not delay review of the bounded RA/Dec route.

No ionosphere, atmosphere or downstream pipeline changes belong in these units.
The accepted CPU/CUDA checkpoint is preserved as a development commit and tag
in the personal fork. These are proposed upstream review units; a smaller
review series can be prepared after agreeing on the interface with maintainers.

## CPU checkpoint before CUDA changes

On 2026-09-22, the local source and library were snapshotted before GPU edits.
The existing CPU suite passed 17/17 targets. Offline tests include independent
ITRS visibility formation, FEKO numerical responses, changing beam pointings,
compact Gaussian sources and unchanged legacy comparisons against 2.13.0.
Both astrometry reference routes use ERFA; FEKO references share the native
electromagnetic kernel. They establish implementation consistency, not an
independent absolute physical accuracy guarantee.

## Questions for maintainers

- Preferred configuration/API names and supported input frame contract?
- Is the optional ERFA and caller-provided EOP approach acceptable?
- Preferred metadata representation, including non-legacy frame labels?
- Is a bounded CPU-first change followed by CUDA acceptable for review?

## Explicit limits

This is distant-source geocentric apparent-direction geometry, not a complete
relativistic VLBI delay model. Observer diurnal aberration, dX/dY, finite
distance, proper motion and parallax are not included. Compact Gaussians retain
the centroid-response and local tangent approximations. Unsupported backends
must not silently fall back to fast geometry.

## Minimal reproduction on unmodified 2.13.0

Run the attached `uvw_issue_reproducer.py` with NumPy, Astropy and PyERFA:

```sh
python uvw_issue_reproducer.py --library /path/to/upstream/liboskar.so --output reproduction.json
```

The script uses only existing public C functions, CPU execution, and bundled
historical IERS-B data with automatic downloads disabled. The test dates are
MJD UTC 59000.25, 60000.5 and 60544.583333333336, with three apparent RA/Dec
directions and station offsets up to 66.858 km. It does not compare identical
numbers interpreted in different reference frames.

On the unmodified 2.13.0 library (source `15776f87`), the range of maximum
coordinate changes across these nine cases is:

| Comparison | Maximum UVW-coordinate change per case |
| --- | ---: |
| Existing projection vs the same analytic rotation | 1.82e-12 to 1.46e-11 m |
| Fast GAST using UT1 instead of UTC | 0.0512 to 1.144 m |
| ERFA GAST vs fast GAST, after using UT1 | 0.0192 to 0.0414 m |
| Combined GAST change | 0.0319 to 1.105 m |

These changes are direction-, baseline- and epoch-dependent. The two maxima
must not simply be added: their signed effects can partly cancel. The test
does not include polar motion or a full visibility calculation. It shows why
an optional explicit EOP/precision route is useful, not that the existing
projection algebra is wrong. These numbers are not an EoR power-spectrum
error estimate.

## Prototype acceptance evidence

The prototype is maintained on the personal fork's `feature/uvw-precision`
branch, with checkpoint tag `uvw-precision-2.13.0-20260930`. A combined
CUDA 12.9, HDF5 1.10.10, casacore 3.8.0 and ERFA build passed 19/19 CTest targets.
An additional output-contract matrix passed 48 CPU/CUDA, legacy/precision and
multi-GPU cases, including synthetic and real FEKO, compact Gaussians, all three
frames, finite integration widths and channel/time blocking. The same matrix
passed after installation with a minimal process environment and fresh HOME.

The maximum CPU/CUDA visibility difference divided by the peak visibility of
each case was 2.15e-12. MS DATA matched the corresponding binary output cast to
complex64 exactly; MS and binary UVW matched bitwise. Four checks confirmed that
enabling MS output did not change binary visibility or UVW. All 11 precision
CPU cases also matched the previously validated binary fixtures bitwise.
Five default-legacy cases remained bitwise identical to unmodified 2.13.0.
These are implementation and storage-contract tests, not an independent
physical-accuracy bound or a downstream calibration compatibility claim.

The minimal public issue attachment needs no telescope files or proprietary
FEKO coefficients. Real FEKO validation data are not included in it.

## Existing reports and submission sequence

On 2026-09-23 the public GitLab issue/MR lists and GitHub issue/PR list were
checked; no matching precision-astrometry proposal was found. This is limited
to publicly visible records, not a guarantee that no internal work exists.
Related but distinct changes include
[MR 29](https://gitlab.com/ska-telescope/sim/oskar/-/merge_requests/29)
(beam coordinate/polarization conventions) and
[MR 33](https://gitlab.com/ska-telescope/sim/oskar/-/merge_requests/33)
(operator precedence in the TEC-screen coordinate transform).

Current public master `5f4423cb` is two macOS build-related commits beyond the
tested 2.13.0 base; neither changes the relevant geometry. Refresh this check
immediately before submission. Prefer one initial discussion issue in the
[canonical GitLab project](https://gitlab.com/ska-telescope/sim/oskar/-/issues),
confirm the maintainer's preferred location, and do not duplicate it on GitHub.

The current implementation is backed up at
[zzh0616/OSKAR](https://github.com/zzh0616/OSKAR/tree/feature/uvw-precision).
First agree on scope/API and optional dependency/metadata contracts. Then
prepare independently reviewable CPU and CUDA commit series. Standalone beam
maps, AZ/EL tracking, HARP and HIP/DCU remain separate follow-up work.
The personal-fork checkpoint is separate from the pending upstream issue/MR.
