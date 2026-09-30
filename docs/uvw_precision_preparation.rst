UVW Precision: Development Contract
===================================

Historical foundation checkpoint, 13-14 September 2026. The subsequent CPU
integration is documented in ``precision_astrometry.rst``; the remaining-work
list below describes the initial projection-only checkpoint, not current status.
At that foundation checkpoint, the default simulator, GPU kernels and atmosphere
interfaces were unchanged.

Implemented foundation
----------------------

``oskar_convert_ecef_to_station_uvw_matrix()`` projects terrestrial station
offsets onto explicitly supplied, orthonormal UVW axes. This initial helper
supports real single/double CPU arrays, output offsets and the existing
ignore-w convention. It does not assume an Earth orientation, celestial
frame, observing time or atmosphere. Unsupported devices fail explicitly.

Let the rows of B be the celestial tangent axes (east, north, source) in
a specified geocentric celestial frame, and C map that frame to the
terrestrial coordinate frame used by the station offsets. Then::

    R = B C^T
    uvw_station = R xyz_station
    uvw_baseline = R (xyz_q - xyz_p)

Both sides of the geometrical phase identity must use these same axes::

    baseline_terrestrial . (s_terrestrial - s0_terrestrial)
        = uvw_baseline . (l, m, n - 1)

The helper tests this identity, not OSKAR's end-to-end DATA phase or a
Measurement Set sign convention. Neither a valid rotation matrix nor small
round-off guarantees that its caller supplied the right epoch or frame.

Do not reinterpret an apparent RA/Dec catalogue as ICRS by changing the UVW
matrix alone. ERFA ``c2t06a`` rotates GCRS to ITRS. It does not perform the
additional ICRS-to-apparent-place transformation, including aberration.

Tests
-----

The native tests are part of the existing ``convert_test`` CMake target.
They require no new dependency. Build out of tree with CPU-only options
and run ``ctest -R '^convert_test$' --output-on-failure``.

``oskar/convert/test/uvw_astrometry_reference.py`` is an additional offline
development check, requiring NumPy, Astropy and PyERFA. Run it in a fresh
Python process with ``--library /absolute/build/oskar/liboskar.so`` and
optionally ``--output report.json``. It uses bundled IERS-B historical dates
only, refuses out-of-range EOP values and records the table/library hashes.
It separates legacy formula reproduction, UTC-as-UT1 differences, residual
GAST approximations and the non-equivalent apparent/GCRS frame contrast.

Astropy and the direct route both use ERFA. They cross-check conventions and
implementation but do not provide independent absolute physical accuracy.
The test coordinates are synthetic offsets, not a new SKA science run.

Remaining integration, in order
-------------------------------

1. Agree on explicit source/phase-centre frames with upstream. Preserve the
   documented legacy apparent-coordinate mode. Do not label a bare GCRS
   rotation as a complete ICRS apparent-place solution.
2. Add an optional common time/EOP state: two-part UTC/UT1/TT dates, leap
   seconds, pole coordinates and a documented interpolation/coverage policy.
   No silent zero-EOP fallback in the precision mode.
3. Use this common state for station/baseline UVW, phase tracking, source
   directions, horizon selection, actual aperture steering weights and
   polarization bases. Test isotropic and multi-element stations, on-axis
   and off-axis sources, both baseline orientations and DATA phase signs.
4. Propagate the same state into finite time/frequency integration and
   supported CPU/GPU kernels. Until implemented, reject unsupported precise
   combinations rather than dispatching old smearing formulae silently.
5. Verify unchanged legacy outputs and optional-backend builds, then submit
   a focused MR. Keep external propagation/ionosphere changes separate.

No production executable, global environment or existing dataset should be
replaced on the strength of this projection-only preparation gate.

References
----------

* `OSKAR fixed-format sky coordinate contract
  <https://ska-telescope.gitlab.io/sim/oskar/sky_model/sky_model_fixed_format.html>`_
* `ERFA celestial-to-terrestrial rotation definition
  <https://github.com/liberfa/erfa/blob/master/src/c2t06a.c>`_
