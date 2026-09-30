/* See the LICENSE file at the top-level directory of this distribution. */
#ifndef OSKAR_ASTROMETRY_H_
#define OSKAR_ASTROMETRY_H_

#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct oskar_Astrometry oskar_Astrometry;
typedef struct oskar_AstrometryState oskar_AstrometryState;

enum OSKAR_ASTROMETRY_FRAME {
    OSKAR_ASTROMETRY_APPARENT = 1,
    OSKAR_ASTROMETRY_ICRS = 2,
    OSKAR_ASTROMETRY_FK5_J2000 = 3
};

/* Immutable, owned EOP table. No network access or process-global state.
 * ASCII columns: MJD_UTC DUT1_seconds xp_arcsec yp_arcsec TAI_minus_UTC_seconds.
 * Rows must bracket the observation; interpolation never extrapolates.
 * The leap-second column must agree with the linked ERFA leap-second table.
 * Catalogue sources are distant and stationary (zero ICRS proper motion).
 */
OSKAR_EXPORT
oskar_Astrometry* oskar_astrometry_create(int frame, const char* eop_file,
        int* status);
OSKAR_EXPORT
void oskar_astrometry_free(oskar_Astrometry* data);
OSKAR_EXPORT
int oskar_astrometry_frame(const oskar_Astrometry* data);
OSKAR_EXPORT
const char* oskar_astrometry_eop_snapshot(const oskar_Astrometry* data);

/* offset_sec is elapsed SI time from the UTC reference epoch. Two-part dates
 * are retained internally. Each worker owns its state; the table is shared.
 */
OSKAR_EXPORT
oskar_AstrometryState* oskar_astrometry_state_create(
        const oskar_Astrometry* data, double reference_mjd_utc,
        double offset_sec, double phase_ra_rad, double phase_dec_rad,
        int* status);
OSKAR_EXPORT
void oskar_astrometry_state_free(oskar_AstrometryState* state);
OSKAR_EXPORT
const double* oskar_astrometry_uvw_matrix(const oskar_AstrometryState* state);
OSKAR_EXPORT
double oskar_astrometry_gast(const oskar_AstrometryState* state);
OSKAR_EXPORT
double oskar_astrometry_tai_minus_utc(const oskar_AstrometryState* state);

/* Geocentric vacuum direction in ITRS, optionally with transported north.
 * No atmosphere, refraction, or ionospheric assumption enters this interface.
 */
OSKAR_EXPORT
void oskar_astrometry_direction(const oskar_AstrometryState* state,
        double ra, double dec, double direction[3], double north[3],
        int* status);

/* Transport a compact Gaussian's FWHM axes (PA north through east) into the
 * UVW frame. Output is two 3-vectors a,b such that attenuation at baseline q
 * in wavelengths is exp(-dot(q,a)^2-dot(q,b)^2). Keeps the Jacobian scale.
 * This is a local tangent Gaussian, not an exact integral on the sphere.
 */
OSKAR_EXPORT
void oskar_astrometry_gaussian_axes(const oskar_AstrometryState* state,
        double ra, double dec, double major_fwhm, double minor_fwhm,
        double position_angle, double axes[6], int* status);
OSKAR_EXPORT
void oskar_astrometry_enu(const double direction[3], double lon, double lat,
        double enu[3]);
/* Celestial inputs are transformed on the host. OSKAR_COORDS_ITRS_DIR accepts
 * already transformed Cartesian direction blocks on CPU/CUDA and rotates them
 * to station-local ENU without applying aberration/precession a second time.
 */
OSKAR_EXPORT
void oskar_astrometry_convert_to_enu(const oskar_AstrometryState* state,
        int coord_type, int num_points, const oskar_Mem* const coords[3],
        double ref_ra, double ref_dec, double lon, double lat,
        oskar_Mem* const enu[3], int* status);

/* Gauss-Legendre nodes on [-1/2, 1/2], with weights summing to one.
 * The caller supplies arrays of length n. Independent of ERFA support.
 */
OSKAR_EXPORT
void oskar_astrometry_quadrature(int n, double* nodes, double* weights,
        int* status);

#ifdef __cplusplus
}
#endif
#endif
