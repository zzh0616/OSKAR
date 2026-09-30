/* See the LICENSE file at the top-level directory of this distribution. */

#include "convert/oskar_convert_ecef_to_station_uvw_matrix.h"
#include "utility/oskar_device.h"

#include <limits.h>
#include <math.h>

static int is_rotation(const double* m)
{
    int i, j, k;
    if (!m) return 0;
    for (i = 0; i < 9; ++i)
        if (!isfinite(m[i])) return 0;
    for (i = 0; i < 3; ++i)
    {
        for (j = i; j < 3; ++j)
        {
            double dot = 0.;
            for (k = 0; k < 3; ++k) dot += m[3*i+k] * m[3*j+k];
            if (fabs(dot - (i == j ? 1. : 0.)) > 1e-12) return 0;
        }
    }
    return m[0]*(m[4]*m[8] - m[5]*m[7]) +
            m[1]*(m[5]*m[6] - m[3]*m[8]) +
            m[2]*(m[3]*m[7] - m[4]*m[6]) > 0.;
}

void oskar_convert_ecef_to_station_uvw_matrix(int num_stations,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        const double matrix[9], int ignore_w_components, int offset_out,
        oskar_Mem* u, oskar_Mem* v, oskar_Mem* w, int* status)
{
    int i, type, location;
    size_t required;
    if (*status) return;
    if (!x || !y || !z || !u || !v || !w ||
            num_stations < 0 || offset_out < 0 ||
            num_stations > INT_MAX - offset_out || !is_rotation(matrix))
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    location = oskar_mem_location(x);
    if (location != OSKAR_CPU && location != OSKAR_GPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }
    if (oskar_mem_location(y) != location || oskar_mem_location(z) != location ||
            oskar_mem_location(u) != location || oskar_mem_location(v) != location ||
            oskar_mem_location(w) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    type = oskar_mem_type(x);
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (oskar_mem_type(y) != type || oskar_mem_type(z) != type ||
            oskar_mem_type(u) != type || oskar_mem_type(v) != type ||
            oskar_mem_type(w) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (oskar_mem_length(x) < (size_t)num_stations ||
            oskar_mem_length(y) < (size_t)num_stations ||
            oskar_mem_length(z) < (size_t)num_stations)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    if (num_stations == 0) return;
    required = (size_t)offset_out + (size_t)num_stations;
    oskar_mem_ensure(u, required, status);
    oskar_mem_ensure(v, required, status);
    oskar_mem_ensure(w, required, status);
    if (*status) return;
    if (location == OSKAR_GPU)
    {
        size_t local[] = {256, 1, 1}, global[] = {1, 1, 1};
        const char* name = type == OSKAR_DOUBLE ?
                "convert_uvw_matrix_double" : "convert_uvw_matrix_float";
        oskar_device_check_local_size(location, 0, local);
        global[0] = oskar_device_global_size(num_stations, local[0]);
        const oskar_Arg args[] = {
            {INT_SZ, &num_stations},
            {PTR_SZ, oskar_mem_buffer_const(x)},
            {PTR_SZ, oskar_mem_buffer_const(y)},
            {PTR_SZ, oskar_mem_buffer_const(z)},
            {DBL_SZ, matrix}, {DBL_SZ, matrix+1}, {DBL_SZ, matrix+2},
            {DBL_SZ, matrix+3}, {DBL_SZ, matrix+4}, {DBL_SZ, matrix+5},
            {DBL_SZ, matrix+6}, {DBL_SZ, matrix+7}, {DBL_SZ, matrix+8},
            {INT_SZ, &ignore_w_components}, {INT_SZ, &offset_out},
            {PTR_SZ, oskar_mem_buffer(u)}, {PTR_SZ, oskar_mem_buffer(v)},
            {PTR_SZ, oskar_mem_buffer(w)}
        };
        oskar_device_launch_kernel(name, location, 1, local, global,
                sizeof(args)/sizeof(oskar_Arg), args, 0, 0, status);
        return;
    }
    if (type == OSKAR_DOUBLE)
    {
        const double *px = oskar_mem_double_const(x, status);
        const double *py = oskar_mem_double_const(y, status);
        const double *pz = oskar_mem_double_const(z, status);
        double *pu = oskar_mem_double(u, status);
        double *pv = oskar_mem_double(v, status);
        double *pw = oskar_mem_double(w, status);
        if (*status) return;
        for (i = 0; i < num_stations; ++i)
        {
            const double a = px[i], b = py[i], c = pz[i];
            const int j = i + offset_out;
            pu[j] = matrix[0]*a + matrix[1]*b + matrix[2]*c;
            pv[j] = matrix[3]*a + matrix[4]*b + matrix[5]*c;
            pw[j] = ignore_w_components ? 0. :
                    matrix[6]*a + matrix[7]*b + matrix[8]*c;
        }
    }
    else
    {
        const float *px = oskar_mem_float_const(x, status);
        const float *py = oskar_mem_float_const(y, status);
        const float *pz = oskar_mem_float_const(z, status);
        float *pu = oskar_mem_float(u, status);
        float *pv = oskar_mem_float(v, status);
        float *pw = oskar_mem_float(w, status);
        if (*status) return;
        for (i = 0; i < num_stations; ++i)
        {
            const double a = px[i], b = py[i], c = pz[i];
            const int j = i + offset_out;
            pu[j] = (float)(matrix[0]*a + matrix[1]*b + matrix[2]*c);
            pv[j] = (float)(matrix[3]*a + matrix[4]*b + matrix[5]*c);
            pw[j] = ignore_w_components ? 0.f :
                    (float)(matrix[6]*a + matrix[7]*b + matrix[8]*c);
        }
    }
}
