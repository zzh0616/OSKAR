/*
 * Copyright (c) 2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */
#include "correlate/oskar_cross_correlate_gaussian_uvw.h"
#include "correlate/private_gaussian_uvw.h"
#include "utility/oskar_device.h"
#include <cfloat>
#include <cmath>

#ifdef OSKAR_HAVE_CUDA
void oskar_cross_correlate_gaussian_uvw_cuda(int casa, int n, int ns, int stride,
        const oskar_Jones* jones, const oskar_Mem* const flux[4],
        const oskar_Mem* const axes[6], const oskar_Mem* const uvw[3],
        double inv_wave, double uv_min, double uv_max, int offset, oskar_Mem* vis);
#endif

namespace {
using oskar_gaussian_uvw::Scalar;
using oskar_gaussian_uvw::Matrix;

template<typename R, typename V, typename Op>
void correlate(int casa, int n, int ns, int stride, const oskar_Jones* jones,
        const oskar_Mem* const flux_mem[4], const oskar_Mem* const axes_mem[6],
        const oskar_Mem* const uvw_mem[3], double inv_wave,
        double uv_min, double uv_max, int offset, oskar_Mem* vis)
{
    const R *flux[4] = {}, *axes[6] = {}, *uvw[3] = {};
    const V* j = static_cast<const V*>(
            oskar_mem_void_const(oskar_jones_mem_const(jones)));
    V* out = static_cast<V*>(oskar_mem_void(vis));
    for (int k = 0; k < 4; ++k)
        if (flux_mem[k]) flux[k] = static_cast<const R*>(oskar_mem_void_const(flux_mem[k]));
    for (int k = 0; k < 6; ++k)
        axes[k] = static_cast<const R*>(oskar_mem_void_const(axes_mem[k]));
    for (int k = 0; k < 3; ++k)
        uvw[k] = static_cast<const R*>(oskar_mem_void_const(uvw_mem[k]));
#pragma omp parallel for schedule(dynamic, 1)
    for (int q = 0; q < ns; ++q)
    {
        for (int p = q + 1; p < ns; ++p)
        {
            double b[3];
            for (int k = 0; k < 3; ++k)
                b[k] = (double(uvw[k][p]) - double(uvw[k][q]))*inv_wave;
            const double uv = std::hypot(b[0], b[1]);
            if (uv < uv_min || uv > uv_max) continue;
            const V* jp = j + size_t(casa ? q : p)*stride;
            const V* jq = j + size_t(casa ? p : q)*stride;
            V sum = {}, guard = {};
            for (int i = 0; i < n; ++i)
            {
                double a = 0., c = 0.;
                for (int k = 0; k < 3; ++k)
                {
                    a += b[k]*axes[k][i];
                    c += b[k]*axes[k+3][i];
                }
                const R envelope = R(std::exp(-(a*a + c*c)));
                Op::add(sum, Op::product(jp[i], jq[i], flux, i), envelope, guard);
            }
            V unused = {};
            const size_t index = size_t(q)*(2*size_t(ns)-q-1)/2 + p-q-1 + offset;
            Op::add(out[index], sum, R(1), unused);
        }
    }
}

bool check(const oskar_Mem* m, int type, int location, size_t length, int* status)
{
    if (!m) *status = OSKAR_ERR_INVALID_ARGUMENT;
    else if (oskar_mem_location(m) != location) *status = OSKAR_ERR_LOCATION_MISMATCH;
    else if (oskar_mem_type(m) != type) *status = OSKAR_ERR_TYPE_MISMATCH;
    else if (oskar_mem_length(m) < length) *status = OSKAR_ERR_DIMENSION_MISMATCH;
    return !*status;
}
} // namespace

void oskar_cross_correlate_gaussian_uvw(int casa, int n, const oskar_Jones* jones,
        const oskar_Mem* const flux[4], const oskar_Mem* const axes[6],
        const oskar_Telescope* tel, const oskar_Mem* const uvw[3],
        double frequency, int offset, oskar_Mem* vis, int* status)
{
    if (*status) return;
    if (!jones || !flux || !axes || !tel || !uvw || !vis || n < 0 ||
            offset < 0 || !std::isfinite(frequency) || frequency <= 0.)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    const int ns = oskar_telescope_num_stations(tel);
    const int stride = oskar_jones_num_sources(jones);
    const int type = oskar_jones_type(jones), real = oskar_type_precision(type);
    const int location = oskar_mem_location(vis);
    if (location != OSKAR_CPU && location != OSKAR_GPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }
    if (ns < 0 || stride < n || oskar_jones_num_stations(jones) != ns)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    if (type != OSKAR_SINGLE_COMPLEX && type != OSKAR_DOUBLE_COMPLEX &&
            type != OSKAR_SINGLE_COMPLEX_MATRIX && type != OSKAR_DOUBLE_COMPLEX_MATRIX)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (!check(oskar_jones_mem_const(jones), type, location, size_t(ns)*stride, status) ||
            !check(vis, type, location, size_t(offset) + size_t(ns)*(ns ? ns-1 : 0)/2, status)) return;
    const int nf = oskar_type_is_matrix(type) ? 4 : 1;
    for (int k = 0; k < nf; ++k) if (!check(flux[k], real, location, n, status)) return;
    for (int k = 0; k < 6; ++k) if (!check(axes[k], real, location, n, status)) return;
    for (int k = 0; k < 3; ++k) if (!check(uvw[k], real, location, ns, status)) return;
    if (!n || ns < 2) return;
    const double inv_wave = frequency/299792458.;
    double uv_min = oskar_telescope_uv_filter_min(tel);
    double uv_max = oskar_telescope_uv_filter_max(tel);
    if (oskar_telescope_uv_filter_units(tel) == OSKAR_METRES)
    {
        uv_min *= inv_wave; uv_max *= inv_wave;
    }
    if (uv_max < 0. || uv_max > FLT_MAX) uv_max = FLT_MAX;
    if (location == OSKAR_GPU)
    {
#ifdef OSKAR_HAVE_CUDA
        oskar_cross_correlate_gaussian_uvw_cuda(casa, n, ns, stride, jones,
                flux, axes, uvw, inv_wave, uv_min, uv_max, offset, vis);
        oskar_device_check_error_cuda(status);
#else
        *status = OSKAR_ERR_CUDA_NOT_AVAILABLE;
#endif
        return;
    }
#define RUN(R, V, ...) correlate<R, V, __VA_ARGS__>(casa, n, ns, stride, jones, \
        flux, axes, uvw, inv_wave, uv_min, uv_max, offset, vis)
    switch (type)
    {
    case OSKAR_SINGLE_COMPLEX: RUN(float, float2, Scalar<float, float2>); break;
    case OSKAR_DOUBLE_COMPLEX: RUN(double, double2, Scalar<double, double2>); break;
    case OSKAR_SINGLE_COMPLEX_MATRIX:
        RUN(float, float4c, Matrix<float, float2, float4c>); break;
    case OSKAR_DOUBLE_COMPLEX_MATRIX:
        RUN(double, double4c, Matrix<double, double2, double4c>); break;
    }
#undef RUN
}
