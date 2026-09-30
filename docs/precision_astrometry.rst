Offline Precision Astrometry
============================

Development implementation based on OSKAR 2.13.0, September 2026. This is a
separate geometry change, not an ionosphere or atmosphere implementation.
The default is still ``legacy`` and does not require ERFA or EOP files.
No production installation is changed by an out-of-tree build.

Build and configuration
-----------------------

Enable the optional native ERFA dependency with ``-DOSKAR_USE_ERFA=ON``.
If necessary set ``ERFA_INCLUDE_DIR`` and ``ERFA_LIBRARY`` to a consistent
installation. Use a clean compiler environment; mixing a system compiler
with conda compiler flags, sysroots or incompatible HDF5 libraries is unsafe.

Minimal additional settings, using OSKAR's INI syntax::

    [simulator]
    use_gpus=false
    double_precision=true

    [telescope]
    allow_station_beam_duplication=false
    ionosphere_screen_type=None

    [interferometer]
    astrometry/mode=precision
    astrometry/frame=icrs
    astrometry/eop_file=/absolute/path/eop.txt
    astrometry/time_samples=8
    astrometry/frequency_samples=8

The reference frame applies consistently to the supplied sky, phase centre,
station RA/Dec pointings and catalogue polarization position angles:

* ``apparent``: equinox-based, geocentric apparent RA/Dec. Fixed input numbers
  remain fixed apparent coordinates during a run; they are not a time series
  of apparent places for a fixed inertial catalogue.
* ``icrs``: distant, stationary ICRS sources. ERFA supplies solar deflection,
  annual aberration and IAU 2006/2000A precession/nutation at each sample.
* ``fk5_j2000``: FK5 mean equator/equinox J2000 positions, converted to ICRS
  with ``eraFk5hz`` before the same apparent-place transformation. This assumes
  zero ICRS proper motion, not arbitrary stellar catalogue space motion.

ICRS, mean FK5 J2000 and apparent coordinates are not interchangeable. This
switch declares input coordinates; it does not discover mixed frames in
catalogues, automatically convert FITS WCS, or add proper motion/parallax.
``c2t06a`` alone is not an ICRS-to-apparent-place transformation.

Offline Earth orientation
-------------------------

An immutable ASCII table has five columns::

    # MJD_UTC DUT1_seconds xp_arcsec yp_arcsec TAI_minus_UTC_seconds
    59000 0.10 0.20 -0.10 37
    59001 0.11 0.21 -0.09 37

These illustrative rows are not measured EOP. Export real historical rows
from a local IERS-B table, without network access::

    python apps/oskar_prepare_eop.py \
      --start 2020-05-31T06:00:00 --end 2020-05-31T07:30:00 \
      --output eop.txt

The exporter uses Astropy's bundled table or ``--iers-b /local/table`` and
records the source hash. Python/Astropy are preparation tools only; simulation
uses native ERFA. Neither program downloads EOP. Replace/update the local
table explicitly when necessary, rather than silently using predicted or
out-of-range values. The exporter refuses to overwrite an existing file.

Rows are finite, ordered daily midnight samples. Every integration boundary
must be bracketed; missing days and extrapolation fail. The leap-second column
must match ERFA. DUT1 interpolation goes through continuous UT1-TAI so a leap
does not become a false drift. Time offsets are SI seconds from two-part UTC,
converted through TAI to TT and UT1. The geometry API handles leap seconds;
the current simulator rejects an observation crossing one because its output
uses uniformly spaced UTC metadata. Split such an observation explicitly.

Geometry and integration
------------------------

At each quadrature time, the same vacuum celestial-to-terrestrial mapping
supplies the phase direction, transported catalogue north, source directions,
station UVW, Jones K, horizon selection, aperture steering and polarization
rotation. If R contains the terrestrial east/north/phase-direction rows::

    uvw = R b
    b . (s - s0) = uvw . (l, m, n - 1)

Station-specific geodetic ENU frames are retained. The output baseline sign
and complex polarization convention still follow ``use_casa_phase_convention``.
Measured coordinates supply output UVW; true coordinates supply simulation
phase, preserving intentional station-position errors.

Positive time/channel widths use tensor-product Gauss-Legendre quadrature of
the full response, including frequency-scaled flux and beam. The old analytic
smearing factors are disabled in this mode to avoid averaging twice. Sample
counts are 1--1024 (at least 2 for positive widths). Increase them to verify
convergence for the actual longest baseline, field and channel width. Eight
samples are not a universal accuracy guarantee, especially across a horizon
discontinuity. Time-indexed gain/error inputs are constant within their native
dump; this change does not invent sub-dump gain variability. Thermal noise is
added once after integration, as before.

Gaussian sources
----------------

Precision mode supports native compact elliptical Gaussians, as well as point
sources, with scalar or full-polarization CPU/CUDA correlation. The catalogue FWHM
axes and position angle (north through east) belong to the declared input frame.
Their local tangent vectors are pushed through the same celestial mapping as
the centroid, retaining its angular magnification/shear. They are recomputed at
each quadrature time, not held in the initial phase-centre projection.

For the mapping s(x), major/minor unit tangent vectors e1/e2 and FWHM theta1/
theta2, the two shape factors in UVW are::

    a = pi / (2 sqrt(ln 2)) theta1 R Ds[e1]
    c = pi / (2 sqrt(ln 2)) theta2 R Ds[e2]
    V_gaussian = V_centroid exp[-(b_lambda . a)^2 - (b_lambda . c)^2]

The factors retain all three UVW components, including the off-axis source
tangents' w projection, and make the attenuation nonnegative by construction.
The legacy fitted two-dimensional ellipse and correlator remain unchanged.
The standard sky loader still uses its existing semantics: native Gaussians
have two positive FWHM axes, with omitted/zero sizes representing point sources.

This remains a **compact-source approximation**, not exact spherical integration
of an arbitrarily large source. The shape uses a local first derivative of the
direction mapping; Jones/beam, polarization rotation and horizon selection are
evaluated at the centroid. Curvature of phase across the source, beam gradients,
and partial horizon clipping are not integrated spatially. For a resolved or
beam-edge source, compare against a sufficiently sampled point representation;
no single angular size guarantees a desired visibility accuracy on all baselines.

The included spherical point-quadrature check uses 1 Jy unpolarized Gaussians,
2--8 arcsec major FWHM, a roughly 16-degree offset, 100/120 MHz and baselines up
to about 51 km. Its maximum absolute differences, divided by total source flux,
are 1.4e-6--5.4e-6 for isotropic stations and 1.1e-5--3.7e-5 with the test
three-element aperture beam. Point quadrature itself agrees to below 4e-12 Jy
when increased from 24 to 40 nodes per axis. These are example approximation
errors, **not universal bounds**, and are distinct from implementation agreement
with the same analytic Gaussian equation (about 1e-10 Jy in these tests).

Output and compatibility
------------------------

Double-precision binary visibility remains complex128. Extended named header
tags ``OSKAR_ASTROMETRY/FRAME_V1`` and ``EOP_V1`` record the input frame and the
exact EOP snapshot. Legacy files have no new tags. An incomplete precision
metadata payload is an error, not permission to infer legacy geometry.

MSv2 DATA remains complex64; UVW remains double. FIELD/POINTING/SOURCE direction
metadata uses APP, ICRS or J2000 as appropriate. MS keywords record the EOP,
input frame and transported geocentric UVW basis. Positive integration channel
widths are written separately from the channel-centre spacing. A zero-width
monochromatic diagnostic retains the legacy positive MS width placeholder and
records zero in ``OSKAR_ASTROMETRY_CHANNEL_WIDTH_HZ``; it is not a physical
finite-bandwidth noise observation. Standard coordinate labels
alone do not guarantee that a third-party predictor implements the identical
delay convention. Downstream DP3/WSClean scientific compatibility is outside
this upstream OSKAR change, not a prerequisite added to its CPU implementation.
The writer round-trip checks below test OSKAR's own output contract only.

The optional ``oskar/convert/test/uvw_full_stack_reference.py`` exercises
CPU/CUDA, legacy/precision, FEKO/HDF5 input and simultaneous binary/MS output
in one build. Supply ``--build`` (a build tree or install prefix), ``--fixtures``
(the analytic and beam acceptance output directories), a fresh ``--output``
directory, ``--gpu`` and optionally ``--multi-gpu``. It requires Python-casacore
for independent-process MS readback in addition to the reference dependencies.
The native simulator runs with a minimal environment and fresh HOME per case;
Python bindings do not control its library loading. Verify that casacore and
OSKAR use a compatible HDF5 dependency stack before interpreting failures.
The test checks exact complex64 quantisation, UVW, direction/frame/EOP metadata,
channel widths, time centres, baseline/polarization order and MS-on/off parity.
Scalar MS output retains the existing XX label, rather than relabelling it I.

Scope and remaining work
------------------------

The current supported geometry is CPU/CUDA, far-field point/compact Gaussian sources
and RA/Dec phase tracking. OpenCL execution, array-level HARP,
ignore-w, shared station beams and ionospheric screens are rejected rather
than silently evaluated with legacy geometry. The standalone beam-pattern
application is not yet wired to the new settings. Multi-element aperture
steering inside the interferometer is wired and tested. FEKO/HDF5 numerical
patterns and the specific pointing combinations below are now covered by an
offline acceptance test. This does not qualify every optional station model.
Native Gaussian morphology no longer requires rejection in precision mode.
The remaining CPU items fall into two categories:

* Extend the acceptance matrix for other numerical pattern formats, station
  configurations and applications before claiming these combinations work.
* Independently extend the standalone beam-pattern app, AZ/EL tracking and HARP
  if desired. They need not be silently supported to submit a scoped RA/Dec
  interferometer change; unsupported combinations must stay explicit.

This is a geocentric vacuum apparent-direction measurement equation, not a
complete relativistic VLBI delay model. It does not add finite source distance,
proper motion, diurnal observer aberration, celestial-pole offsets dX/dY, or
atmospheric refraction. Numerical agreement of two ERFA implementations is not
an absolute physical-error bound, nor an EoR signal-recovery certificate.
Their scientific importance must be evaluated separately before such claims.

CUDA execution
--------------

With a CUDA build, set ``simulator/use_gpus=true`` and explicitly select
``simulator/cuda_device_ids``. Each worker owns its geometry state and staging
buffers. EOP interpolation, ERFA apparent-place transformations, sky clipping,
Gaussian shape derivatives and polarization transport are evaluated on the
host. Source directions and factors are uploaded in blocks per time node and
reused over frequency nodes. There are no per-source device reads for ERFA.

The GPU projects station UVW and ITRS directions into each station's ENU frame,
then evaluates the beam, phase, Jones products and correlations. Gaussian
correlation includes all three UVW components and preserves the full Jones
source stride. Its CUDA reduction has a different summation order from CPU;
bitwise equality is not promised. Single precision remains available but is
not an EoR-accuracy mode. Use double precision for quantitative comparisons.

The timing log reports ``Host astrometry incl. upload`` separately from beam
and correlation timings. This is a sum over workers and includes host work
and bulk copies, not a pure transfer benchmark. CPU worker count is controlled
by ``simulator/num_devices``; OSKAR sets one OpenMP thread per worker.

Offline GPU acceptance reuses the analytic and FEKO inputs after their CPU
reference tests pass. To enable both through CTest, set
``OSKAR_ASTROMETRY_TEST_PYTHON`` and ``OSKAR_ASTROMETRY_TEST_CUDA_IDS=0`` (or
``0,1`` for the additional multi-card cases). The GPU test requires the beam
fixture automatically, including when selected alone with ``ctest -R``.
The fixtures require real CUDA device access; they are not emulation tests.

The standalone ``uvw_gpu_reference.py`` accepts ``--build``, ``--fixtures``
(one or more CPU fixture directories), ``--output``, ``--gpu`` and optional
``--multi-gpu``. ``uvw_gpu_benchmark.py`` adds a bounded 64-station, 4096-source
benchmark with selectable ``--cpu-workers``; its timings are not a forecast
for a production SKA simulation. CUDA Compute Sanitizer can be run on the
Gaussian unit test and a generated FEKO settings file.

Input validation also rejects undefined element-type indices. The earlier
numerical-disabled FEKO test fixture incorrectly retained types 1 and 2 after
only type 0 was loaded. Its historical control amplitude is invalid; the
corrected analytic fixture uses type 0 for every element. This does not
invalidate the independent comparisons with numerical FEKO enabled.

FEKO and pointing acceptance
----------------------------

The FEKO/HDF5 suite exercises the full-polarization interferometer with three
embedded element types, three stations, three times and two frequencies. It
covers apparent/ICRS/FK5-J2000, different station pointings, element/final-beam
normalization, swapped feeds, baseline signs, permitted discrete beams,
two-level beamforming, Gaussian sources, finite time/channel integration and
source chunking across workers. A numerical-disabled control verifies that the
numerical data are actually used, rather than merely loaded and ignored.

For this route set ``telescope/pol_mode=Full``, enable numerical patterns and
use a non-isotropic functional element type (e.g. ``Dipole``). Upstream OSKAR's
scalar element path evaluates an analytic response, and an explicitly isotropic
type takes precedence even with numerical data loaded. This suite does not
certify scalar FEKO Jones simulation. It also does not add frequency
interpolation: native element evaluation selects the closest coefficient set.

The reference first computes ITRS/ENU directions with PyERFA, then calls the
existing beam kernel with **explicit ENU inputs and AZ/EL steering, with no
precision state attached**. It forms visibility directly from ITRS baselines.
This avoids sharing the new coordinate/steering plumbing while deliberately
sharing the electromagnetic beam kernel. It is not independent validation of
the FEKO electromagnetic model. Synthetic coefficients are small, reproducible
regression fixtures, not a claimed SKA beam. Optionally, the test extracts just
X1--3/Y1--3 from a supplied real 100 MHz FEKO HDF5 file and records dataset
hashes. These are real coefficient samples in a toy array, not a complete SKA
station-model validation. Off-table frequencies still use nearest coefficients;
they do not validate the physical response at those frequencies.

There are three distinct meanings of a changing pointing:

* A fixed catalogue RA/Dec tracked through changing hour angle is supported and
  tested over multiple times.
* The native station-beam API is tested with explicitly changed RA/Dec
  pointings and times, reusing work buffers and revisiting the original state.
  It does not retain stale beam coordinates.
* A time-tagged arbitrary scan/ephemeris scheduler is **not** provided by this
  change. OSKAR's existing pointing file has station addresses and coordinates,
  not time columns. The API test is not a new end-to-end scan scheduler.

Likewise, an RA/Dec interferometer phase centre with fixed AZ/EL station or tile
beams is supported and tested. This is different from an **AZ/EL interferometer
phase centre** (the drift-scan observation mode), which precision preflight
still rejects. Standalone ``oskar_sim_beam_pattern`` has not acquired these
precision settings; visibility simulation uses its own, already connected
station-beam evaluation and does not require that standalone application.
Array-level HARP remains a separate unsupported route, not a requirement for
ordinary element-by-element aperture synthesis using FEKO coefficients.

Validation
----------

Native ``astrometry_test`` checks EOP and leap semantics, frames, poles,
orthonormal bases, polynomial quadrature moments and binary metadata. Existing
``convert_test`` and ``jones_test`` exercise the projection/phase infrastructure.
Gaussian tests additionally cover position angles, poles, off-axis w projection,
non-unit angular magnification, zero-size limits, float/double scalar/matrix
correlation, baseline signs, output offsets, source strides and UV filtering.
Build and test both ``OSKAR_USE_ERFA=ON`` and ``OFF``.

Additional offline checks (NumPy, Astropy and PyERFA)::

    python oskar/convert/test/uvw_simulation_reference.py \
      --build /absolute/build --output /absolute/test-output [--ms]

    python oskar/convert/test/uvw_integration_contract.py \
      --build /absolute/build --fixtures /absolute/test-output \
      --reference-build /absolute/pristine-2.13-build --output /absolute/contracts

    python oskar/convert/test/uvw_gaussian_point_reference.py \
      --build /absolute/build --fixtures /absolute/test-output \
      --output /absolute/gaussian-point-check

The first forms visibility directly from terrestrial baselines, rather than
reusing OSKAR UVW/Jones/beam routines. It covers all three frames, both signs,
full polarization, multi-element stations, horizon crossing, finite integration,
source chunking, mixed Gaussian/point skies and optional MS output. The second
checks preflight rejections and unchanged legacy outputs, including Gaussians,
against a separately built pristine release. The third quantifies the compact
Gaussian approximation against spatial point quadrature in the same simulator.
Both astrometry references use ERFA: they are independent implementations of
the measurement equation, not independent astrometric theories.

FEKO acceptance additionally needs h5py and a CPU build with native HDF5::

    python oskar/convert/test/uvw_beam_reference.py \
      --build /absolute/build --output /absolute/beam-check \
      [--fixtures /absolute/test-output] \
      [--real-feko /local/SKA-Low_FEKO_beam_100.00000MHz.h5] \
      [--reference-build /absolute/pristine-2.13-build]

Without ``--fixtures`` it generates and validates the base fixtures locally.
Nothing is downloaded, including EOP or beam files. The optional pristine
build checks legacy FEKO outputs, including normalization, hierarchy and
Gaussian sources, for bitwise equality. Real coefficients and the pristine
build are optional and are not needed by automated CI.

To register the self-contained synthetic suite in CTest, configure with::

    -DOSKAR_USE_ERFA=ON \
    -DOSKAR_ASTROMETRY_TEST_PYTHON=/path/to/test-environment/python

Then run ``ctest -R astrometry_beam_reference --output-on-failure``.
This opt-in requires HDF5 and the Python test dependencies; it does not add a
Python runtime dependency or change the default legacy build.
