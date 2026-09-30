/* See the LICENSE file at the top-level directory of this distribution. */
#include "interferometer/private_interferometer.h"
#include "interferometer/oskar_interferometer.h"
#include "convert/oskar_convert_ecef_to_station_uvw_matrix.h"
#include "math/oskar_cmath.h"
#include <string.h>

static void invalid(oskar_Interferometer* h, const char* message, int* status)
{
    oskar_log_error(h->log, "Precision astrometry: %s", message);
    *status = OSKAR_ERR_INVALID_ARGUMENT;
}

static void check_element_types(oskar_Interferometer* h,
        const oskar_Station* station, int* status)
{
    int i = 0;
    if (*status) return;
    if (oskar_station_has_child(station))
    {
        for (i = 0; i < oskar_station_num_elements(station) && !*status; ++i)
            check_element_types(h, oskar_station_child_const(station, i), status);
    }
    else
    {
        const int* types = oskar_station_element_types_cpu_const(station);
        const int count = oskar_station_num_element_types(station);
        for (i = 0; types && i < oskar_station_num_elements(station); ++i)
            if (types[i] < 0 || types[i] >= count)
            {
                invalid(h, "element_types.txt references an undefined element model", status);
                return;
            }
    }
}

void oskar_interferometer_set_astrometry(oskar_Interferometer* h,
        const char* mode, const char* frame, const char* file,
        int time_samples, int frequency_samples, int* status)
{
    int frame_id = 0;
    oskar_Astrometry* a = 0;
    if (*status) return;
    if (!h || !mode || !frame)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    if (h->header)
    {
        invalid(h, "set options before initialisation", status);
        return;
    }
    if (!strcmp(mode, "legacy"))
    {
        if (strcmp(frame, "apparent") || (file && file[0]))
        {
            invalid(h, "legacy mode requires apparent coordinates and no EOP file", status);
            return;
        }
        oskar_astrometry_free(h->astrometry);
        h->astrometry = 0;
        return;
    }
    if (strcmp(mode, "precision") || time_samples < 1 || frequency_samples < 1 ||
            time_samples > 1024 || frequency_samples > 1024)
    {
        invalid(h, "invalid mode or integration sample count", status);
        return;
    }
    if (!strcmp(frame, "apparent")) frame_id = OSKAR_ASTROMETRY_APPARENT;
    else if (!strcmp(frame, "icrs")) frame_id = OSKAR_ASTROMETRY_ICRS;
    else if (!strcmp(frame, "fk5_j2000")) frame_id = OSKAR_ASTROMETRY_FK5_J2000;
    a = oskar_astrometry_create(frame_id, file, status);
    if (*status)
    {
        oskar_log_error(h->log, "Cannot initialise precision astrometry: "
                "check ERFA support, frame, daily EOP columns, and leap seconds.");
        return;
    }
    oskar_astrometry_free(h->astrometry);
    h->astrometry = a;
    h->astrometry_time_samples = time_samples;
    h->astrometry_frequency_samples = frequency_samples;
}

void oskar_interferometer_check_astrometry(oskar_Interferometer* h, int* status)
{
    int i = 0;
    double dat = 0.;
    if (*status || !h->astrometry) return;
    if (oskar_telescope_mem_location(h->tel) != OSKAR_CPU ||
            (h->num_gpus && h->dev_loc != OSKAR_GPU))
        invalid(h, "host telescope and CPU or CUDA devices are required", status);
    else if (oskar_telescope_phase_centre_coord_type(h->tel) != OSKAR_COORDS_RADEC)
        invalid(h, "only RA/Dec phase tracking is currently supported", status);
    else if (oskar_telescope_ionosphere_screen_type(h->tel) != 'N')
        invalid(h, "ionosphere coupling is intentionally outside this branch", status);
    else if (oskar_telescope_allow_station_beam_duplication(h->tel))
        invalid(h, "disable station beam duplication to retain each station's horizon", status);
    else if (h->ignore_w_components)
        invalid(h, "ignore_w_components is incompatible with precise phase geometry", status);
    else if (oskar_telescope_time_average_sec(h->tel) > 0. &&
            h->astrometry_time_samples < 2)
        invalid(h, "nonzero time averaging requires time_samples >= 2", status);
    else if (oskar_telescope_channel_bandwidth_hz(h->tel) > 0. &&
            h->astrometry_frequency_samples < 2)
        invalid(h, "nonzero bandwidth requires frequency_samples >= 2", status);
    if (*status) return;
    for (i = 0; i < oskar_telescope_num_station_models(h->tel) && !*status; ++i)
        check_element_types(h, oskar_telescope_station_const(h->tel, i), status);
    if (h->num_time_steps < 1 || h->num_channels < 1 ||
            !isfinite(h->time_inc_sec) || h->time_inc_sec <= 0. ||
            !isfinite(oskar_telescope_time_average_sec(h->tel)) ||
            oskar_telescope_time_average_sec(h->tel) < 0. ||
            !isfinite(oskar_telescope_channel_bandwidth_hz(h->tel)) ||
            oskar_telescope_channel_bandwidth_hz(h->tel) < 0.)
        invalid(h, "invalid observation dimensions or integration widths", status);
    for (i = 0; i < h->num_sky_chunks && !*status; ++i)
    {
        const oskar_Sky* sky = h->sky_chunks[i];
        const int columns[] = {OSKAR_SKY_MAJOR_RAD, OSKAR_SKY_MINOR_RAD, OSKAR_SKY_PA_RAD};
        /* Validate even when a malformed width did not set USE_EXTENDED. */
        for (int col = 0; col < 3 && !*status; ++col)
        {
            const oskar_Mem* values = oskar_sky_column_const(sky, columns[col], 0);
            if (!values) continue;
            for (int j = 0; j < oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES) && !*status; ++j)
            {
                const double v = oskar_mem_get_element(values, j, status);
                if (!isfinite(v) || (col < 2 && v < 0.))
                    invalid(h, "invalid Gaussian FWHM or position angle", status);
            }
        }
    }
    for (i = 0; i < h->num_channels && !*status; ++i)
    {
        const double f = h->freq_start_hz + i*h->freq_inc_hz;
        if (!isfinite(f) || f - 0.5*oskar_telescope_channel_bandwidth_hz(h->tel) <= 0.)
            invalid(h, "frequency integration extends to nonpositive frequency", status);
        if (oskar_telescope_harp_data(h->tel, f))
            invalid(h, "array-level HARP geometry is not yet supported", status);
    }
    /* Validate all dump boundaries before creating outputs. Restrict the
     * simulator's uniform-UTC metadata to observations without a leap second. */
    for (i = -1; i < h->num_time_steps && !*status; ++i)
    {
        int edge = 0;
        for (edge = -1; edge <= 1; ++edge)
        {
            const double offset = i < 0 ? 0. : (i + 0.5)*h->time_inc_sec +
                    edge*0.5*oskar_telescope_time_average_sec(h->tel);
            oskar_AstrometryState* s = oskar_astrometry_state_create(h->astrometry,
                    h->time_start_mjd_utc, offset,
                    oskar_telescope_phase_centre_longitude_rad(h->tel),
                    oskar_telescope_phase_centre_latitude_rad(h->tel), status);
            if (s)
            {
                const double current = oskar_astrometry_tai_minus_utc(s);
                if (dat && current != dat)
                    invalid(h, "split observations that cross a UTC leap second", status);
                dat = current;
            }
            oskar_astrometry_state_free(s);
        }
    }
    if (*status)
        oskar_log_error(h->log, "Precision astrometry preflight failed; "
                "no legacy fallback will be used.");
}

void oskar_interferometer_astrometry_uvw(const oskar_Telescope* tel,
        const oskar_AstrometryState* state, int use_true, int ignore_w,
        int offset, oskar_Mem* u, oskar_Mem* v, oskar_Mem* w, int* status)
{
    const oskar_Mem* xyz[3];
    int k = 0;
    if (*status) return;
    for (k = 0; k < 3; ++k)
        xyz[k] = use_true ? oskar_telescope_station_true_offset_ecef_metres_const(tel, k) :
                oskar_telescope_station_measured_offset_ecef_metres_const(tel, k);
    oskar_convert_ecef_to_station_uvw_matrix(oskar_telescope_num_stations(tel),
            xyz[0], xyz[1], xyz[2], oskar_astrometry_uvw_matrix(state),
            ignore_w, offset, u, v, w, status);
}

void oskar_interferometer_astrometry_prepare(oskar_Interferometer* h,
        DeviceData* d, const oskar_Sky* sky, int* status)
{
    int i = 0, k = 0, station = 0;
    const int n = oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES);
    const int ns = oskar_telescope_num_stations(d->tel);
    const double* b = oskar_astrometry_uvw_matrix(d->astrometry_state);
    const int extended = oskar_sky_int(sky, OSKAR_SKY_USE_EXTENDED);
    if (*status) return;
    if (oskar_sky_int(sky, OSKAR_SKY_MEM_LOCATION) != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }
    for (k = 0; k < 3; ++k)
    {
        oskar_mem_ensure(d->lmn_cpu[k], n, status);
        oskar_mem_ensure(d->itrs_cpu[k], n, status);
    }
    if (extended)
        for (k = 0; k < 6; ++k) oskar_mem_ensure(d->gaussian_axes_cpu[k], n, status);
    if (d->R_cpu) oskar_jones_set_size(d->R_cpu, ns, n, status);
    oskar_interferometer_astrometry_uvw(d->tel, d->astrometry_state,
            1, 0, 0, d->uvw[0], d->uvw[1], d->uvw[2], status);
    for (i = 0; i < n && !*status; ++i)
    {
        double p[3] = {0., 0., 0.}, north[3] = {0., 0., 0.}, east[3];
        const double ra = oskar_mem_get_element(
                oskar_sky_column_const(sky, OSKAR_SKY_RA_RAD, 0), i, status);
        const double dec = oskar_mem_get_element(
                oskar_sky_column_const(sky, OSKAR_SKY_DEC_RAD, 0), i, status);
        oskar_astrometry_direction(d->astrometry_state, ra, dec, p,
                d->R ? north : 0, status);
        for (k = 0; k < 3; ++k)
        {
            oskar_mem_set_element_real(d->itrs_cpu[k], i, p[k], status);
            oskar_mem_set_element_real(d->lmn_cpu[k], i,
                    b[3*k]*p[0]+b[3*k+1]*p[1]+b[3*k+2]*p[2], status);
        }
        if (extended)
        {
            double axes[6] = {0., 0., 0., 0., 0., 0.};
            const oskar_Mem* major = oskar_sky_column_const(sky, OSKAR_SKY_MAJOR_RAD, 0);
            const oskar_Mem* minor = oskar_sky_column_const(sky, OSKAR_SKY_MINOR_RAD, 0);
            const oskar_Mem* pa = oskar_sky_column_const(sky, OSKAR_SKY_PA_RAD, 0);
            oskar_astrometry_gaussian_axes(d->astrometry_state, ra, dec,
                    major ? oskar_mem_get_element(major, i, status) : 0.,
                    minor ? oskar_mem_get_element(minor, i, status) : 0.,
                    pa ? oskar_mem_get_element(pa, i, status) : 0., axes, status);
            for (k = 0; k < 6; ++k)
                oskar_mem_set_element_real(d->gaussian_axes_cpu[k], i, axes[k], status);
        }
        if (!d->R) continue;
        east[0] = north[1]*p[2]-north[2]*p[1];
        east[1] = north[2]*p[0]-north[0]*p[2];
        east[2] = north[0]*p[1]-north[1]*p[0];
        for (station = 0; station < ns && !*status; ++station)
        {
            const oskar_Station* st = oskar_telescope_station_const(h->tel, station);
            double ne[3], ee[3];
            double angle = 0., c = 0., s = 0.;
            oskar_astrometry_enu(north, oskar_station_lon_rad(st),
                    oskar_station_lat_rad(st), ne);
            oskar_astrometry_enu(east, oskar_station_lon_rad(st),
                    oskar_station_lat_rad(st), ee);
            angle = atan2(ee[2], ne[2]);
            c = cos(angle); s = sin(angle);
            if (h->prec == OSKAR_DOUBLE)
            {
                double4c* r = oskar_mem_double4c(oskar_jones_mem(d->R_cpu), status) + station*n + i;
                r->a.x = c; r->b.x = -s; r->c.x = s; r->d.x = c;
                r->a.y = r->b.y = r->c.y = r->d.y = 0.;
            }
            else
            {
                float4c* r = oskar_mem_float4c(oskar_jones_mem(d->R_cpu), status) + station*n + i;
                r->a.x = (float)c; r->b.x = (float)-s;
                r->c.x = (float)s; r->d.x = (float)c;
                r->a.y = r->b.y = r->c.y = r->d.y = 0.f;
            }
        }
    }
    /* Upload whole source blocks once per time node, never per source/station. */
    for (k = 0; k < 3; ++k)
    {
        oskar_mem_ensure(d->lmn[k], n, status);
        oskar_mem_ensure(d->itrs[k], n, status);
        oskar_mem_copy_contents(d->lmn[k], d->lmn_cpu[k], 0, 0, n, status);
        oskar_mem_copy_contents(d->itrs[k], d->itrs_cpu[k], 0, 0, n, status);
    }
    if (extended)
        for (k = 0; k < 6; ++k)
        {
            oskar_mem_ensure(d->gaussian_axes[k], n, status);
            oskar_mem_copy_contents(d->gaussian_axes[k], d->gaussian_axes_cpu[k], 0, 0, n, status);
        }
    if (d->R)
    {
        oskar_jones_set_size(d->R, ns, n, status);
        oskar_mem_copy_contents(oskar_jones_mem(d->R), oskar_jones_mem_const(d->R_cpu),
                0, 0, (size_t)ns*n, status);
    }
}
