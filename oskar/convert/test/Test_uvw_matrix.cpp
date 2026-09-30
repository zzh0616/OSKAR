/* See the LICENSE file at the top-level directory of this distribution. */

#include <gtest/gtest.h>

#include "convert/oskar_convert_ecef_to_station_uvw.h"
#include "convert/oskar_convert_ecef_to_station_uvw_matrix.h"
#include "utility/oskar_device.h"

#include <cmath>
#include <climits>
#include <limits>

namespace {

const double identity[] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};

class UvwMatrix : public ::testing::Test
{
protected:
    oskar_Mem *xyz[3], *uvw[3];
    int status;

    void SetUp()
    {
        status = 0;
        for (int d = 0; d < 3; ++d)
        {
            xyz[d] = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 3, &status);
            uvw[d] = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 1, &status);
            double* p = oskar_mem_double(xyz[d], &status);
            p[0] = 0.;
            p[1] = 1000. * (d + 1);
            p[2] = -17000. * (d + 1);
            oskar_mem_set_value_real(uvw[d], -123., 0, 1, &status);
        }
        ASSERT_EQ(0, status);
    }

    void TearDown()
    {
        int cleanup = 0;
        for (int d = 0; d < 3; ++d)
        {
            oskar_mem_free(xyz[d], &cleanup);
            oskar_mem_free(uvw[d], &cleanup);
        }
    }

    void project(const double* matrix, int offset = 0, int ignore_w = 0,
            int count = 3)
    {
        oskar_convert_ecef_to_station_uvw_matrix(count,
                xyz[0], xyz[1], xyz[2], matrix, ignore_w, offset,
                uvw[0], uvw[1], uvw[2], &status);
    }
};

TEST_F(UvwMatrix, identity_and_offset_growth)
{
    project(identity, 2);
    ASSERT_EQ(0, status);
    for (int d = 0; d < 3; ++d)
    {
        ASSERT_GE(oskar_mem_length(uvw[d]), 5u);
        const double* p = oskar_mem_double_const(uvw[d], &status);
        EXPECT_EQ(-123., p[0]);
        for (int i = 0; i < 3; ++i)
            EXPECT_EQ(oskar_mem_double_const(xyz[d], &status)[i], p[i+2]);
    }
}

TEST_F(UvwMatrix, legacy_apparent_formula)
{
    for (int t = 0; t < 12; ++t)
    {
        const double ra = .37, dec = -.51, gast = .17 + .5*t;
        const double h = gast - ra;
        const double matrix[] = {
            sin(h), cos(h), 0.,
            -sin(dec)*cos(h), sin(dec)*sin(h), cos(dec),
            cos(dec)*cos(h), -cos(dec)*sin(h), sin(dec)
        };
        project(matrix);
        ASSERT_EQ(0, status);
        oskar_Mem* legacy[3];
        for (int d = 0; d < 3; ++d)
            legacy[d] = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 3, &status);
        oskar_convert_ecef_to_station_uvw(3, xyz[0], xyz[1], xyz[2],
                ra, dec, gast, 0, 0, legacy[0], legacy[1], legacy[2], &status);
        EXPECT_EQ(0, status);
        for (int d = 0; d < 3; ++d)
        {
            for (int i = 0; i < 3; ++i)
                EXPECT_NEAR(oskar_mem_double_const(legacy[d], &status)[i],
                        oskar_mem_double_const(uvw[d], &status)[i], 1e-9);
            int cleanup = 0;
            oskar_mem_free(legacy[d], &cleanup);
        }
    }
}

TEST_F(UvwMatrix, baseline_and_phase_invariant)
{
    const double angle = .123;
    const double matrix[] = {cos(angle), sin(angle), 0.,
            -sin(angle), cos(angle), 0., 0., 0., 1.};
    const double local_direction[] = {.1, -.2, sqrt(.95) - 1.};
    project(matrix);
    ASSERT_EQ(0, status);
    double in_norm = 0., out_norm = 0., direct_phase = 0., uvw_phase = 0.;
    for (int d = 0; d < 3; ++d)
    {
        const double* p = oskar_mem_double_const(xyz[d], &status);
        const double* q = oskar_mem_double_const(uvw[d], &status);
        const double b = p[2] - p[1];
        const double uvw_baseline = q[2] - q[1];
        double delta_s = 0.;
        for (int j = 0; j < 3; ++j)
            delta_s += matrix[3*j+d] * local_direction[j];
        direct_phase += b * delta_s;
        uvw_phase += uvw_baseline * local_direction[d];
        in_norm += b*b;
        out_norm += uvw_baseline*uvw_baseline;
    }
    EXPECT_NEAR(sqrt(in_norm), sqrt(out_norm), 1e-9);
    EXPECT_NEAR(direct_phase, uvw_phase, 1e-9);
}

TEST_F(UvwMatrix, ignore_w)
{
    project(identity, 1, 1);
    ASSERT_EQ(0, status);
    EXPECT_EQ(-123., oskar_mem_double_const(uvw[2], &status)[0]);
    for (int i = 1; i < 4; ++i)
        EXPECT_EQ(0., oskar_mem_double_const(uvw[2], &status)[i]);
}

TEST_F(UvwMatrix, single_precision)
{
    for (int d = 0; d < 3; ++d)
    {
        oskar_mem_free(xyz[d], &status);
        oskar_mem_free(uvw[d], &status);
        xyz[d] = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 3, &status);
        uvw[d] = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 0, &status);
        oskar_mem_set_value_real(xyz[d], 10.*d, 0, 3, &status);
    }
    project(identity, 2);
    ASSERT_EQ(0, status);
    for (int d = 0; d < 3; ++d)
        for (int i = 2; i < 5; ++i)
            EXPECT_FLOAT_EQ(10.f*d, oskar_mem_float_const(uvw[d], &status)[i]);
}

TEST_F(UvwMatrix, invalid_rotation)
{
    double bad[9];
    for (int mode = 0; mode < 5; ++mode)
    {
        for (int i = 0; i < 9; ++i) bad[i] = identity[i];
        if (mode == 0) bad[0] = 2.;
        if (mode == 1) bad[0] = -1.;
        if (mode == 2) bad[0] = std::numeric_limits<double>::quiet_NaN();
        if (mode == 3) bad[1] = .01;
        if (mode == 4) bad[0] = std::numeric_limits<double>::infinity();
        project(bad);
        EXPECT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
        status = 0;
        EXPECT_EQ(-123., oskar_mem_double_const(uvw[0], &status)[0]);
    }
    project(0);
    EXPECT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
}

TEST_F(UvwMatrix, invalid_ranges)
{
    project(identity, -1);
    EXPECT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
    status = 0;
    project(identity, INT_MAX);
    EXPECT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
    status = 0;
    project(identity, 0, 0, -1);
    EXPECT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
    status = 0;
    project(identity, 0, 0, 4);
    EXPECT_EQ(OSKAR_ERR_DIMENSION_MISMATCH, status);
}

TEST_F(UvwMatrix, type_mismatch)
{
    oskar_mem_free(uvw[0], &status);
    uvw[0] = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 1, &status);
    project(identity);
    EXPECT_EQ(OSKAR_ERR_TYPE_MISMATCH, status);
}

TEST_F(UvwMatrix, complex_input_rejected)
{
    oskar_mem_free(xyz[0], &status);
    xyz[0] = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, 3, &status);
    project(identity);
    EXPECT_EQ(OSKAR_ERR_BAD_DATA_TYPE, status);
}

TEST_F(UvwMatrix, empty_and_incoming_status)
{
    project(identity, 0, 0, 0);
    ASSERT_EQ(0, status);
    EXPECT_EQ(1u, oskar_mem_length(uvw[0]));
    status = OSKAR_ERR_OUT_OF_RANGE;
    project(0);
    EXPECT_EQ(OSKAR_ERR_OUT_OF_RANGE, status);
    status = 0;
    EXPECT_EQ(-123., oskar_mem_double_const(uvw[0], &status)[0]);
}

#ifdef OSKAR_HAVE_CUDA
TEST_F(UvwMatrix, cuda_rotation_offset_and_ignore_w)
{
    const double a = .23;
    const double matrix[] = {cos(a), sin(a), 0., -sin(a), cos(a), 0., 0., 0., 1.};
    oskar_device_set(OSKAR_GPU, 0, &status);
    ASSERT_EQ(0, status);
    for (int single = 0; single < 2; ++single)
    for (int ignore = 0; ignore < 2; ++ignore)
    {
        const int type = single ? OSKAR_SINGLE : OSKAR_DOUBLE;
        oskar_Mem *input[3], *output[3];
        for (int k = 0; k < 3; ++k)
        {
            oskar_Mem* host = oskar_mem_convert_precision(xyz[k], type, &status);
            input[k] = oskar_mem_create_copy(host, OSKAR_GPU, &status);
            output[k] = oskar_mem_create(type, OSKAR_GPU, 5, &status);
            oskar_mem_set_value_real(output[k], -123., 0, 5, &status);
            oskar_mem_free(host, &status);
        }
        oskar_convert_ecef_to_station_uvw_matrix(3, input[0], input[1], input[2],
                matrix, ignore, 2, output[0], output[1], output[2], &status);
        project(matrix, 2, ignore);
        ASSERT_EQ(0, status);
        for (int k = 0; k < 3; ++k)
        {
            oskar_Mem* host = oskar_mem_create_copy(output[k], OSKAR_CPU, &status);
            EXPECT_EQ(-123., oskar_mem_get_element(host, 0, &status));
            EXPECT_EQ(-123., oskar_mem_get_element(host, 1, &status));
            for (int i = 2; i < 5; ++i)
                EXPECT_NEAR(oskar_mem_get_element(uvw[k], i, &status),
                        oskar_mem_get_element(host, i, &status), single ? .004 : 1e-10);
            oskar_mem_free(host, &status);
            oskar_mem_free(input[k], &status);
            oskar_mem_free(output[k], &status);
        }
    }
}
#endif

} // namespace
