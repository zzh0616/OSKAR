[![GitHub release](https://img.shields.io/github/release/OxfordSKA/OSKAR.svg?style=flat-square)](https://github.com/OxfordSKA/OSKAR/releases)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3758491.svg)](https://doi.org/10.5281/zenodo.3758491)

# OSKAR: A GPU-accelerated simulator for the Square Kilometre Array

## Precision astrometry development fork

The `feature/uvw-precision` branch in
[zzh0616/OSKAR](https://github.com/zzh0616/OSKAR/tree/feature/uvw-precision)
adds optional offline precision geometry to the OSKAR 2.13.0 interferometer.
It supports explicit apparent, ICRS and FK5 J2000 inputs, consistent geometry
through the beam/phase/polarization calculation, and CPU/CUDA execution.
The existing legacy mode remains the default. See
[the configuration and acceptance documentation](docs/precision_astrometry.rst).

The development checkpoint is tagged `uvw-precision-2.13.0-20260930`.
It has passed the combined CUDA, HDF5 and Measurement Set acceptance described
in [the upstream proposal draft](docs/uvw_precision_upstream.md). This branch
has not been accepted upstream. Standalone beam-pattern, AZ/EL phase tracking,
HARP and HIP/DCU remain separate extensions.

Development commits are maintained in this fork. Upstream updates come from
`https://gitlab.com/ska-telescope/sim/oskar.git`; merge them on a separate update
branch, rerun legacy and opt-in tests, then merge the tested update into
`feature/uvw-precision`. Published checkpoint tags identify fixed source states.

OSKAR has been designed to produce simulated visibility data from radio
telescopes containing aperture arrays, such as those envisaged for the
Square Kilometre Array.

A source code archive, and pre-built binary packages for Linux (using
Singularity), macOS and Windows platforms are available to download from

- [https://github.com/OxfordSKA/OSKAR/releases](https://github.com/OxfordSKA/OSKAR/releases)

macOS builds support Apple Silicon (arm64) on macOS 14 or later. Intel Macs
are not supported.

OSKAR is licensed under the terms of the 3-clause BSD License,
and makes use of a number of third-party libraries.
Please see the OSKAR documentation for license details
of third-party libraries used by OSKAR.

## Documentation

Package documentation is built automatically as part of the
GitLab CI/CD pipeline, and can be viewed at

- [https://ska-telescope.gitlab.io/sim/oskar/](https://ska-telescope.gitlab.io/sim/oskar/)

## Installation Guide

If compiling from source, see the
[Installation Guide](https://ska-telescope.gitlab.io/sim/oskar/install/install.html)
in the documentation.
