/* See the LICENSE file at the top-level directory of this distribution. */

#ifndef OSKAR_CONVERT_ECEF_TO_STATION_UVW_MATRIX_H_
#define OSKAR_CONVERT_ECEF_TO_STATION_UVW_MATRIX_H_

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Projects terrestrial station offsets onto a supplied UVW basis.
 *
 * @details
 * The row-major matrix contains the u, v and w unit vectors expressed in
 * the SAME terrestrial frame as x, y and z. It must be finite, orthonormal
 * and right-handed (checked to 1e-12). Coordinates and output share a unit,
 * normally metres. The phase centre points along positive w.
 *
 * No time-scale, sky-frame, Earth-orientation or atmosphere model is assumed.
 * For example, a caller can form the matrix from an explicitly defined
 * celestial tangent basis and a GCRS-to-ITRS rotation. This projection does
 * not itself transform ICRS catalogue positions to apparent directions.
 *
 * This implementation supports CPU/CUDA real single/double arrays.
 * Unsupported locations return an error; they do not fall back to legacy
 * astrometry. Input and output arrays must not overlap. All arrays must have
 * the same type and location. Outputs grow to offset_out + num_stations, preserving the
 * prefix. The matrix is still validated when ignoring w or processing zero
 * stations. A non-zero incoming status leaves all outputs untouched.
 *
 * @param[in] num_stations Number of station coordinates to project.
 * @param[in] x Input terrestrial x offsets.
 * @param[in] y Input terrestrial y offsets.
 * @param[in] z Input terrestrial z offsets.
 * @param[in] matrix Row-major 3 by 3 terrestrial-to-UVW rotation.
 * @param[in] ignore_w_components If non-zero, set output w to zero.
 * @param[in] offset_out Start index in output arrays.
 * @param[in,out] u Output u coordinates.
 * @param[in,out] v Output v coordinates.
 * @param[in,out] w Output w coordinates.
 * @param[in,out] status OSKAR status code.
 */
OSKAR_EXPORT
void oskar_convert_ecef_to_station_uvw_matrix(int num_stations,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        const double matrix[9], int ignore_w_components, int offset_out,
        oskar_Mem* u, oskar_Mem* v, oskar_Mem* w, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_ECEF_TO_STATION_UVW_MATRIX_H_ */
