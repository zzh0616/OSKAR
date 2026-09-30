/* See the LICENSE file at the top-level directory of this distribution. */
#ifndef OSKAR_DEFINE_CONVERT_UVW_MATRIX_H_
#define OSKAR_DEFINE_CONVERT_UVW_MATRIX_H_

#define OSKAR_CONVERT_UVW_MATRIX(NAME, R) \
KERNEL(NAME)(const int n, GLOBAL_IN(R, x), GLOBAL_IN(R, y), GLOBAL_IN(R, z), \
        const double a, const double b, const double c, \
        const double d, const double e, const double f, \
        const double g, const double h, const double k, \
        const int ignore_w, const int offset, \
        GLOBAL_OUT(R, u), GLOBAL_OUT(R, v), GLOBAL_OUT(R, w)) \
{ \
    KERNEL_LOOP_X(int, i, 0, n) \
    const double xx = x[i], yy = y[i], zz = z[i]; \
    u[i + offset] = (R)(a*xx + b*yy + c*zz); \
    v[i + offset] = (R)(d*xx + e*yy + f*zz); \
    w[i + offset] = ignore_w ? (R)0 : (R)(g*xx + h*yy + k*zz); \
    KERNEL_LOOP_END \
} \
OSKAR_REGISTER_KERNEL(NAME)

#endif
