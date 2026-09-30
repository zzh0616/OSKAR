/*
 * Copyright (c) 2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */
#ifndef OSKAR_CROSS_CORRELATE_GAUSSIAN_UVW_H_
#define OSKAR_CROSS_CORRELATE_GAUSSIAN_UVW_H_

#include <correlate/oskar_cross_correlate.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief CPU/CUDA correlation of local tangent-plane Gaussian sources in UVW.
 *
 * The six real source arrays hold two three-vector Gaussian factors from
 * oskar_astrometry_gaussian_axes(). For a baseline b in wavelengths the
 * envelope is exp(-dot(b, a)^2 - dot(b, c)^2). Zero factors represent a point.
 * Jones matrices (including the phase) are evaluated at the source centroid.
 * No analytic bandwidth/time smearing is applied: the caller must integrate
 * samples explicitly. Fluxes, Jones matrices and station coordinates refer to
 * the same sample and coordinate basis. Results are added at offset_out.
 * Arrays share the Jones precision and CPU/CUDA location; the telescope is
 * used only for station count and UV filtering metadata. CUDA uses block
 * reduction, so CPU/GPU sums need not be bitwise equal.
 */
OSKAR_EXPORT
void oskar_cross_correlate_gaussian_uvw(int use_casa_phase_convention,
        int num_sources, const oskar_Jones* jones,
        const oskar_Mem* const src_flux[4],
        const oskar_Mem* const axes[6], const oskar_Telescope* tel,
        const oskar_Mem* const station_uvw[3], double frequency_hz,
        int offset_out, oskar_Mem* vis, int* status);

#ifdef __cplusplus
}
#endif
#endif
