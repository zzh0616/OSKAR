/* See the LICENSE file at the top-level directory of this distribution. */
#include "correlate/oskar_cross_correlate_gaussian_uvw.h"
#include "correlate/private_gaussian_uvw.h"
#include <cuda_runtime.h>

namespace {
using oskar_gaussian_uvw::Scalar;
using oskar_gaussian_uvw::Matrix;

template<typename R> struct Inputs
{
    const R* flux[4];
    const R* axes[6];
    const R* uvw[3];
};

/* One block per baseline; threads sum disjoint source subsets. No atomics. */
template<typename R, typename V, typename Op>
__global__ void correlate(int casa, int n, int ns, int stride, const V* j,
        Inputs<R> input, double inv_wave, double uv_min, double uv_max,
        int offset, V* out)
{
    const int p = blockIdx.x, q = blockIdx.y, t = threadIdx.x;
    if (p <= q) return;
    double b[3];
    for (int k = 0; k < 3; ++k)
        b[k] = (double(input.uvw[k][p]) - double(input.uvw[k][q]))*inv_wave;
    const double uv = hypot(b[0], b[1]);
    if (uv < uv_min || uv > uv_max) return;
    const V* jp = j + size_t(casa ? q : p)*stride;
    const V* jq = j + size_t(casa ? p : q)*stride;
    V sum = {}, guard = {};
    for (int i = t; i < n; i += blockDim.x)
    {
        double a = 0., c = 0.;
        for (int k = 0; k < 3; ++k)
        {
            a += b[k]*input.axes[k][i];
            c += b[k]*input.axes[k+3][i];
        }
        const R envelope = R(exp(-(a*a + c*c)));
        Op::add(sum, Op::product(jp[i], jq[i], input.flux, i), envelope, guard);
    }
    __shared__ V partial[128];
    partial[t] = sum;
    __syncthreads();
    for (int step = blockDim.x/2; step > 0; step /= 2)
    {
        if (t < step)
        {
            V unused = {};
            Op::add(partial[t], partial[t+step], R(1), unused);
        }
        __syncthreads();
    }
    if (t == 0)
    {
        V unused = {};
        const size_t index = size_t(q)*(2*size_t(ns)-q-1)/2 + p-q-1 + offset;
        Op::add(out[index], partial[0], R(1), unused);
    }
}

template<typename R, typename V, typename Op>
void launch(int casa, int n, int ns, int stride, const oskar_Jones* jones,
        const oskar_Mem* const flux[4], const oskar_Mem* const axes[6],
        const oskar_Mem* const uvw[3], double inv_wave, double uv_min,
        double uv_max, int offset, oskar_Mem* vis)
{
    Inputs<R> input = {};
    for (int k = 0; k < 4; ++k)
        if (flux[k]) input.flux[k] = static_cast<const R*>(oskar_mem_void_const(flux[k]));
    for (int k = 0; k < 6; ++k)
        input.axes[k] = static_cast<const R*>(oskar_mem_void_const(axes[k]));
    for (int k = 0; k < 3; ++k)
        input.uvw[k] = static_cast<const R*>(oskar_mem_void_const(uvw[k]));
    correlate<R, V, Op><<<dim3(ns, ns), 128>>>(casa, n, ns, stride,
            static_cast<const V*>(oskar_mem_void_const(oskar_jones_mem_const(jones))),
            input, inv_wave, uv_min, uv_max, offset, static_cast<V*>(oskar_mem_void(vis)));
}
} // namespace

void oskar_cross_correlate_gaussian_uvw_cuda(int casa, int n, int ns, int stride,
        const oskar_Jones* jones, const oskar_Mem* const flux[4],
        const oskar_Mem* const axes[6], const oskar_Mem* const uvw[3],
        double inv_wave, double uv_min, double uv_max, int offset, oskar_Mem* vis)
{
#define RUN(R, V, ...) launch<R, V, __VA_ARGS__>(casa, n, ns, stride, jones, \
        flux, axes, uvw, inv_wave, uv_min, uv_max, offset, vis)
    switch (oskar_mem_type(vis))
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
