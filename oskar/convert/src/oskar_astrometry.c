/* See the LICENSE file at the top-level directory of this distribution. */
#include "convert/oskar_astrometry.h"
#include "convert/oskar_convert_ecef_to_station_uvw_matrix.h"
#include "convert/oskar_convert_any_to_enu_directions.h"
#include "math/oskar_cmath.h"
#include <ctype.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef OSKAR_HAVE_ERFA
#include <erfa.h>
#endif

typedef struct {
    double mjd, dut1, xp, yp, dat;
} EopRow;

struct oskar_Astrometry {
    int frame, num_rows;
    EopRow* rows;
    char* snapshot;
};

struct oskar_AstrometryState {
    int frame;
    double gast, eo, dat, cirs_to_itrs[3][3], basis[9];
#ifdef OSKAR_HAVE_ERFA
    eraASTROM astrom;
#endif
};

#ifdef OSKAR_HAVE_ERFA
static double dot(const double a[3], const double b[3])
{
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

static void cross(const double a[3], const double b[3], double c[3])
{
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
    c[2] = a[0]*b[1] - a[1]*b[0];
}

static void unit(double a[3], int* status)
{
    int k = 0;
    const double norm = sqrt(dot(a, a));
    if (!isfinite(norm) || norm < 1e-14)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    for (k = 0; k < 3; ++k) a[k] /= norm;
}
#endif

oskar_Astrometry* oskar_astrometry_create(int frame, const char* file,
        int* status)
{
    if (*status) return 0;
#ifndef OSKAR_HAVE_ERFA
    (void) frame;
    (void) file;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    return 0;
#else
    oskar_Astrometry* a = 0;
    FILE* stream = 0;
    char line[2048];
    size_t length = 0;
    if (frame < OSKAR_ASTROMETRY_APPARENT ||
            frame > OSKAR_ASTROMETRY_FK5_J2000 || !file || !file[0])
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return 0;
    }
    stream = fopen(file, "r");
    if (!stream)
    {
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }
    a = (oskar_Astrometry*) calloc(1, sizeof(*a));
    if (!a)
    {
        fclose(stream);
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }
    a->frame = frame;
    while (!*status && fgets(line, sizeof(line), stream))
    {
        EopRow row, *rows = 0;
        char *p = line, *snapshot = 0;
        int n = 0, year = 0, month = 0, day = 0;
        double fraction = 0., dat = 0.;
        const size_t size = strlen(line);
        if (!strchr(line, '\n') && !feof(stream))
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            break;
        }
        snapshot = (char*) realloc(a->snapshot, length + size + 1);
        if (!snapshot)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            break;
        }
        a->snapshot = snapshot;
        memcpy(a->snapshot + length, line, size + 1);
        length += size;
        while (isspace((unsigned char)*p)) ++p;
        if (!*p || *p == '#') continue;
        if (sscanf(p, "%lf %lf %lf %lf %lf %n", &row.mjd, &row.dut1,
                &row.xp, &row.yp, &row.dat, &n) != 5)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            break;
        }
        p += n;
        if ((*p && *p != '#') || !isfinite(row.mjd) ||
                !isfinite(row.dut1) || !isfinite(row.xp) ||
                !isfinite(row.yp) || !isfinite(row.dat) ||
                row.mjd != floor(row.mjd) || fabs(row.dut1) > 1.0 ||
                fabs(row.xp) > 10.0 || fabs(row.yp) > 10.0 ||
                (a->num_rows && row.mjd <= a->rows[a->num_rows-1].mjd) ||
                eraJd2cal(2400000.5, row.mjd, &year, &month, &day,
                        &fraction) ||
                eraDat(year, month, day, 0., &dat) != 0 ||
                fabs(dat - row.dat) > 1e-9)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            break;
        }
        rows = (EopRow*) realloc(a->rows, (a->num_rows + 1)*sizeof(*rows));
        if (!rows)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            break;
        }
        a->rows = rows;
        a->rows[a->num_rows++] = row;
    }
    if (ferror(stream)) *status = OSKAR_ERR_FILE_IO;
    fclose(stream);
    if (a->num_rows < 2 && !*status) *status = OSKAR_ERR_INVALID_ARGUMENT;
    if (*status)
    {
        oskar_astrometry_free(a);
        return 0;
    }
    return a;
#endif
}

void oskar_astrometry_free(oskar_Astrometry* a)
{
    if (!a) return;
    free(a->rows);
    free(a->snapshot);
    free(a);
}

int oskar_astrometry_frame(const oskar_Astrometry* a)
{
    return a ? a->frame : 0;
}

const char* oskar_astrometry_eop_snapshot(const oskar_Astrometry* a)
{
    return a ? a->snapshot : 0;
}

oskar_AstrometryState* oskar_astrometry_state_create(
        const oskar_Astrometry* a, double reference_mjd, double offset_sec,
        double ra, double dec, int* status)
{
    oskar_AstrometryState* s = 0;
    if (*status) return 0;
#ifndef OSKAR_HAVE_ERFA
    (void) a; (void) reference_mjd; (void) offset_sec; (void) ra; (void) dec;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
#else
    double utc1 = 0., utc2 = 0., tai1 = 0., tai2 = 0., tt1 = 0., tt2 = 0.;
    double ut11 = 0., ut12 = 0., mjd = 0., f = 0., dut1 = 0., pm[3][3];
    double xp = 0., yp = 0., dat = 0., fraction = 0.;
    int lo = 0, hi = 0, year = 0, month = 0, day = 0;
    const EopRow *r0 = 0, *r1 = 0;
    if (!a || !isfinite(reference_mjd) || !isfinite(offset_sec))
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return 0;
    }
    utc1 = 2400000.5 + floor(reference_mjd);
    utc2 = reference_mjd - floor(reference_mjd);
    if (eraUtctai(utc1, utc2, &tai1, &tai2) != 0 ||
            eraTaiutc(tai1, tai2 + offset_sec/86400., &utc1, &utc2) != 0)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return 0;
    }
    tai2 += offset_sec/86400.;
    mjd = (utc1 - 2400000.5) + utc2;
    if (mjd < a->rows[0].mjd || mjd > a->rows[a->num_rows-1].mjd ||
            eraJd2cal(utc1, utc2, &year, &month, &day, &fraction) ||
            eraDat(year, month, day, fraction, &dat) != 0)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return 0;
    }
    hi = a->num_rows - 1;
    while (hi - lo > 1)
    {
        const int mid = (hi + lo)/2;
        if (a->rows[mid].mjd <= mjd) lo = mid;
        else hi = mid;
    }
    r0 = &a->rows[lo]; r1 = &a->rows[hi];
    /* Daily EOP only: do not silently interpolate across missing days. */
    if (r1->mjd - r0->mjd > 1.0)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return 0;
    }
    f = ((utc1 - 2400000.5 - r0->mjd) + utc2)/(r1->mjd - r0->mjd);
    /* UT1-TAI is continuous across a UTC leap; DUT1 itself is not. */
    dut1 = (1.-f)*(r0->dut1-r0->dat) + f*(r1->dut1-r1->dat) + dat;
    xp = ((1.-f)*r0->xp + f*r1->xp) * (M_PI/(180.*3600.));
    yp = ((1.-f)*r0->yp + f*r1->yp) * (M_PI/(180.*3600.));
    if (eraTaitt(tai1, tai2, &tt1, &tt2) ||
            eraUtcut1(utc1, utc2, dut1, &ut11, &ut12) != 0)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return 0;
    }
    s = (oskar_AstrometryState*) calloc(1, sizeof(*s));
    if (!s)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }
    s->frame = a->frame;
    s->dat = dat;
    s->gast = eraGst06a(ut11, ut12, tt1, tt2);
    eraApci13(tt1, tt2, &s->astrom, &s->eo);
    eraPom00(xp, yp, eraSp00(tt1, tt2), pm);
    eraIr(s->cirs_to_itrs);
    eraRz(eraEra00(ut11, ut12), s->cirs_to_itrs);
    eraRxr(pm, s->cirs_to_itrs, s->cirs_to_itrs);
    oskar_astrometry_direction(s, ra, dec, s->basis + 6, s->basis + 3,
            status);
    cross(s->basis + 3, s->basis + 6, s->basis);
    unit(s->basis, status);
    if (*status)
    {
        free(s);
        return 0;
    }
#endif
    return s;
}

void oskar_astrometry_state_free(oskar_AstrometryState* s) { free(s); }
const double* oskar_astrometry_uvw_matrix(const oskar_AstrometryState* s)
{ return s ? s->basis : 0; }
double oskar_astrometry_gast(const oskar_AstrometryState* s)
{ return s ? s->gast : 0.; }
double oskar_astrometry_tai_minus_utc(const oskar_AstrometryState* s)
{ return s ? s->dat : 0.; }

#ifdef OSKAR_HAVE_ERFA
static void direction(const oskar_AstrometryState* s, double ra, double dec,
        double out[3])
{
    int k = 0;
    double ri = 0., di = 0., v[3];
    if (s->frame == OSKAR_ASTROMETRY_APPARENT)
    {
        ri = ra + s->eo; di = dec;
    }
    else
    {
        if (s->frame == OSKAR_ASTROMETRY_FK5_J2000)
            eraFk5hz(ra, dec, 2451545., 0., &ra, &dec);
        /* Keep the shared state immutable even with ERFA's non-const API. */
        eraASTROM astrom = s->astrom;
        eraAtciqz(ra, dec, &astrom, &ri, &di);
    }
    eraS2c(ri, di, v);
    for (k = 0; k < 3; ++k) out[k] = dot(s->cirs_to_itrs[k], v);
}
#endif

void oskar_astrometry_direction(const oskar_AstrometryState* s,
        double ra, double dec, double out[3], double north[3], int* status)
{
    if (*status) return;
#ifndef OSKAR_HAVE_ERFA
    (void) s; (void) ra; (void) dec; (void) out; (void) north;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
#else
    if (!s || !out || !isfinite(ra) || !isfinite(dec) || fabs(dec) > M_PI/2.)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    direction(s, ra, dec, out);
    if (north)
    {
        const double step = 1e-4;
        const double input[3] = {cos(dec)*cos(ra), cos(dec)*sin(ra), sin(dec)};
        const double tangent[3] = {-sin(dec)*cos(ra), -sin(dec)*sin(ra), cos(dec)};
        const int offsets[4] = {-2, -1, 1, 2};
        const double weights[4] = {1., -8., 8., -1.};
        double radial = 0.;
        int i = 0, k = 0;
        north[0] = north[1] = north[2] = 0.;
        /* Transport the catalogue polarization basis through the same map,
         * including aberration. Great-circle offsets avoid RA pole singularities.
         * A fourth-order stencil has O(step^4) truncation error. */
        for (i = 0; i < 4; ++i)
        {
            const double a = offsets[i]*step;
            double p[3], mapped[3];
            for (k = 0; k < 3; ++k)
                p[k] = cos(a)*input[k] + sin(a)*tangent[k];
            direction(s, atan2(p[1], p[0]), atan2(p[2], hypot(p[0], p[1])), mapped);
            for (k = 0; k < 3; ++k) north[k] += weights[i]*mapped[k];
        }
        radial = dot(north, out);
        for (k = 0; k < 3; ++k) north[k] -= radial*out[k];
        unit(north, status);
    }
#endif
}

void oskar_astrometry_enu(const double p[3], double lon, double lat,
        double enu[3])
{
    const double t = cos(lon)*p[0] + sin(lon)*p[1];
    enu[0] = -sin(lon)*p[0] + cos(lon)*p[1];
    enu[1] = -sin(lat)*t + cos(lat)*p[2];
    enu[2] = cos(lat)*t + sin(lat)*p[2];
}

void oskar_astrometry_gaussian_axes(const oskar_AstrometryState* s,
        double ra, double dec, double major, double minor, double pa,
        double axes[6], int* status)
{
    if (*status) return;
#ifndef OSKAR_HAVE_ERFA
    (void) s; (void) ra; (void) dec; (void) major; (void) minor; (void) pa;
    (void) axes;
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
#else
    const double step = 1e-4;
    const double factor = M_PI/(2.*sqrt(log(2.)));
    const int offsets[4] = {-2, -1, 1, 2};
    const double weights[4] = {1., -8., 8., -1.};
    double centre[3], input[3], north[3], east[3];
    int a = 0, k = 0, j = 0;
    if (!s || !axes || !isfinite(ra) || !isfinite(dec) ||
            fabs(dec) > M_PI/2. || !isfinite(major) || !isfinite(minor) ||
            !isfinite(pa) || major < 0. || minor < 0.)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    direction(s, ra, dec, centre);
    input[0] = cos(dec)*cos(ra); input[1] = cos(dec)*sin(ra);
    input[2] = sin(dec);
    north[0] = -sin(dec)*cos(ra); north[1] = -sin(dec)*sin(ra);
    north[2] = cos(dec);
    east[0] = -sin(ra); east[1] = cos(ra); east[2] = 0.;
    for (a = 0; a < 2; ++a)
    {
        double tangent[3], derivative[3] = {0., 0., 0.};
        const double width = a ? minor : major;
        for (k = 0; k < 3; ++k)
            tangent[k] = a ? cos(pa)*east[k] - sin(pa)*north[k] :
                    sin(pa)*east[k] + cos(pa)*north[k];
        for (j = 0; j < 4 && width > 0.; ++j)
        {
            double p[3], mapped[3];
            const double angle = offsets[j]*step;
            for (k = 0; k < 3; ++k)
                p[k] = cos(angle)*input[k] + sin(angle)*tangent[k];
            direction(s, atan2(p[1], p[0]),
                    atan2(p[2], hypot(p[0], p[1])), mapped);
            for (k = 0; k < 3; ++k)
                derivative[k] += weights[j]*mapped[k]/(12.*step);
        }
        /* Remove finite-difference radial roundoff, not the physical angular
         * magnification/shear from aberration and light deflection. */
        const double radial = dot(derivative, centre);
        for (k = 0; k < 3; ++k) derivative[k] -= radial*centre[k];
        for (k = 0; k < 3; ++k)
            axes[3*a+k] = width*factor*dot(s->basis+3*k, derivative);
    }
#endif
}

void oskar_astrometry_convert_to_enu(const oskar_AstrometryState* s,
        int type, int n, const oskar_Mem* const coords[3],
        double ra0, double dec0, double lon, double lat,
        oskar_Mem* const enu[3], int* status)
{
    int i = 0, k = 0, precision = 0;
    double basis[3][3] = {{-sin(ra0), cos(ra0), 0.},
        {-sin(dec0)*cos(ra0), -sin(dec0)*sin(ra0), cos(dec0)},
        {cos(dec0)*cos(ra0), cos(dec0)*sin(ra0), sin(dec0)}};
    if (*status) return;
    if (!s || n < 0 || !coords || !enu || !coords[0] || !coords[1])
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    if (type == OSKAR_COORDS_ITRS_DIR)
    {
        const double sl = sin(lon), cl = cos(lon), sp = sin(lat), cp = cos(lat);
        const double rotation[] = {-sl, cl, 0., -sp*cl, -sp*sl, cp,
                cp*cl, cp*sl, sp};
        oskar_convert_ecef_to_station_uvw_matrix(n,
                coords[0], coords[1], coords[2], rotation, 0, 0,
                enu[0], enu[1], enu[2], status);
        return;
    }
    if (type != OSKAR_COORDS_RADEC && type != OSKAR_COORDS_REL_DIR &&
            type != OSKAR_COORDS_AZEL && type != OSKAR_COORDS_ENU_DIR)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    precision = oskar_mem_type(coords[0]);
    for (k = 0; k < 3; ++k)
    {
        if (!enu[k] || oskar_mem_location(enu[k]) != OSKAR_CPU ||
                oskar_mem_type(enu[k]) != precision ||
                (precision != OSKAR_DOUBLE && precision != OSKAR_SINGLE))
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        if (k < 2 || type == OSKAR_COORDS_REL_DIR || type == OSKAR_COORDS_ENU_DIR)
        {
            if (!coords[k] || oskar_mem_location(coords[k]) != OSKAR_CPU ||
                    oskar_mem_type(coords[k]) != precision ||
                    oskar_mem_length(coords[k]) < (size_t)n)
            {
                *status = OSKAR_ERR_DIMENSION_MISMATCH;
                return;
            }
        }
        oskar_mem_ensure(enu[k], n, status);
    }
    if (type == OSKAR_COORDS_AZEL || type == OSKAR_COORDS_ENU_DIR)
    {
        oskar_Mem* output[3] = {enu[0], enu[1], enu[2]};
        oskar_convert_any_to_enu_directions(type, n, coords, ra0, dec0,
                0., lat, output, status);
        return;
    }
    for (i = 0; i < n && !*status; ++i)
    {
        double ra = oskar_mem_get_element(coords[0], i, status);
        double dec = oskar_mem_get_element(coords[1], i, status);
        double p[3], local[3];
        if (type == OSKAR_COORDS_REL_DIR)
        {
            const double z = oskar_mem_get_element(coords[2], i, status);
            for (k = 0; k < 3; ++k)
                p[k] = ra*basis[0][k] + dec*basis[1][k] + z*basis[2][k];
            ra = atan2(p[1], p[0]); dec = atan2(p[2], hypot(p[0], p[1]));
        }
        oskar_astrometry_direction(s, ra, dec, p, 0, status);
        oskar_astrometry_enu(p, lon, lat, local);
        for (k = 0; k < 3; ++k)
            oskar_mem_set_element_real(enu[k], i, local[k], status);
    }
}

void oskar_astrometry_quadrature(int n, double* nodes, double* weights,
        int* status)
{
    int i = 0;
    if (*status) return;
    if (n < 1 || n > 1024 || !nodes || !weights)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    for (i = 0; i < (n + 1)/2; ++i)
    {
        int iteration = 0;
        double z = cos(M_PI*(i + 0.75)/(n + 0.5)), derivative = 0.;
        for (iteration = 0; iteration < 100; ++iteration)
        {
            int k = 0;
            double p = 1., prev = 0., step = 0.;
            for (k = 1; k <= n; ++k)
            {
                const double next = ((2.*k - 1.)*z*p - (k - 1.)*prev)/k;
                prev = p; p = next;
            }
            derivative = n*(z*p - prev)/(z*z - 1.);
            step = p/derivative;
            if (fabs(step) <= 4.*DBL_EPSILON) break;
            z -= step;
        }
        if (iteration == 100)
        {
            *status = OSKAR_ERR_INVALID_ARGUMENT;
            return;
        }
        nodes[i] = -0.5*z;
        nodes[n-1-i] = 0.5*z;
        weights[i] = weights[n-1-i] = 1./((1.-z*z)*derivative*derivative);
    }
}
