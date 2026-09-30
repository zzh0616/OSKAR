/* See the LICENSE file at the top-level directory of this distribution. */
#ifndef OSKAR_PRIVATE_GAUSSIAN_UVW_H_
#define OSKAR_PRIVATE_GAUSSIAN_UVW_H_

#include "correlate/define_correlate_utils.h"
#include "math/define_multiply.h"
#include "math/oskar_kahan_sum.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __CUDACC__
#define OSKAR_GAUSSIAN_HD __host__ __device__
#else
#define OSKAR_GAUSSIAN_HD
#endif

namespace oskar_gaussian_uvw {
template<typename R, typename V> struct Scalar
{
    OSKAR_GAUSSIAN_HD static V product(V p, V q, const R* const flux[4], int i)
    {
        V v = {(p.x*q.x + p.y*q.y)*flux[0][i],
                (p.y*q.x - p.x*q.y)*flux[0][i]};
        return v;
    }
    OSKAR_GAUSSIAN_HD static void add(V& sum, V v, R scale, V& guard)
    {
        OSKAR_KAHAN_SUM_MULTIPLY_COMPLEX(R, sum, v, scale, guard)
    }
};

template<typename R, typename C, typename V> struct Matrix
{
    OSKAR_GAUSSIAN_HD static V product(V p, V q, const R* const flux[4], int i)
    {
        V b = {};
        OSKAR_CONSTRUCT_B(R, b, flux[0][i], flux[1][i], flux[2][i], flux[3][i])
        OSKAR_MUL_COMPLEX_MATRIX_HERMITIAN_IN_PLACE(C, p, b)
        OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(C, p, q)
        return p;
    }
    OSKAR_GAUSSIAN_HD static void add(V& sum, V v, R scale, V& guard)
    {
        OSKAR_KAHAN_SUM_MULTIPLY_COMPLEX_MATRIX(R, sum, v, scale, guard)
    }
};
} // namespace
#undef OSKAR_GAUSSIAN_HD
#endif
