/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "utility/oskar_timer.h"

#include "correlate/oskar_cross_correlate.h"
#include "correlate/oskar_cross_correlate_gaussian_uvw.h"
#include "utility/oskar_get_error_string.h"
#include "math/oskar_kahan_sum.h"
#include <cstdlib>
#include <cmath>

// Comment out this line to disable benchmark timer printing.
 #define ALLOW_PRINTING 1

static void check_values(const oskar_Mem* approx, const oskar_Mem* accurate)
{
    int status = 0;
    double min_rel_error = 0.0, max_rel_error = 0.0;
    double avg_rel_error = 0.0, std_rel_error = 0.0, tol = 0.0;
    oskar_mem_evaluate_relative_error(approx, accurate, &min_rel_error,
            &max_rel_error, &avg_rel_error, &std_rel_error, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    tol = oskar_mem_is_double(approx) &&
            oskar_mem_is_double(accurate) ? 1e-10 : 2e-2;
    EXPECT_LT(max_rel_error, tol) << std::setprecision(5) <<
            "RELATIVE ERROR" <<
            " MIN: " << min_rel_error << " MAX: " << max_rel_error <<
            " AVG: " << avg_rel_error << " STD: " << std_rel_error;
    tol = oskar_mem_is_double(approx) &&
            oskar_mem_is_double(accurate) ? 1e-12 : 1e-5;
    EXPECT_LT(avg_rel_error, tol) << std::setprecision(5) <<
            "RELATIVE ERROR" <<
            " MIN: " << min_rel_error << " MAX: " << max_rel_error <<
            " AVG: " << avg_rel_error << " STD: " << std_rel_error;
}

static const char* loc_to_str(int loc)
{
    switch (loc) {
    case OSKAR_CPU: return "CPU";
    case OSKAR_GPU: return "GPU";
    case OSKAR_CL:  return "CL";
    default: return "";
    }
}

#ifdef OSKAR_HAVE_CUDA
static void check_gaussian_cuda(int casa, int count, const oskar_Jones* jones,
        const oskar_Mem* const flux[4], const oskar_Mem* const axes[6],
        const oskar_Telescope* tel, const oskar_Mem* const uvw[3],
        int offset, const oskar_Mem* expected)
{
    int status = 0;
    oskar_Jones* j = oskar_jones_create(oskar_jones_type(jones), OSKAR_GPU,
            oskar_jones_num_stations(jones), oskar_jones_num_sources(jones), &status);
    oskar_mem_copy(oskar_jones_mem(j), oskar_jones_mem_const(jones), &status);
    oskar_Mem *f[4], *a[6], *u[3];
    const oskar_Mem *fc[4], *ac[6], *uc[3];
    for (int k = 0; k < 4; ++k)
        fc[k] = f[k] = oskar_mem_create_copy(flux[k], OSKAR_GPU, &status);
    for (int k = 0; k < 6; ++k)
        ac[k] = a[k] = oskar_mem_create_copy(axes[k], OSKAR_GPU, &status);
    for (int k = 0; k < 3; ++k)
        uc[k] = u[k] = oskar_mem_create_copy(uvw[k], OSKAR_GPU, &status);
    oskar_Mem* out = oskar_mem_create(oskar_mem_type(expected), OSKAR_GPU,
            oskar_mem_length(expected), &status);
    ASSERT_EQ(0, status);
    for (int repeat = 1; repeat <= 2; ++repeat)
    {
        oskar_cross_correlate_gaussian_uvw(casa, count, j, fc, ac, tel, uc,
                100e6, offset, out, &status);
        oskar_Mem* result = oskar_mem_create_copy(out, OSKAR_CPU, &status);
        ASSERT_EQ(0, status);
        const bool single = oskar_mem_is_single(result);
        const int components = oskar_mem_is_matrix(result) ? 8 : 2;
        for (size_t k = 0; k < components*oskar_mem_length(result); ++k)
        {
            const double x = repeat*(single ? oskar_mem_float_const(expected, &status)[k] :
                    oskar_mem_double_const(expected, &status)[k]);
            const double y = single ? oskar_mem_float(result, &status)[k] :
                    oskar_mem_double(result, &status)[k];
            EXPECT_NEAR(x, y, (single ? 3e-6 : 1e-12)*std::fabs(x) + (single ? .03 : 1e-9));
        }
        oskar_mem_free(result, &status);
    }
    oskar_mem_free(out, &status);
    oskar_jones_free(j, &status);
    for (int k = 0; k < 4; ++k) oskar_mem_free(f[k], &status);
    for (int k = 0; k < 6; ++k) oskar_mem_free(a[k], &status);
    for (int k = 0; k < 3; ++k) oskar_mem_free(u[k], &status);
    EXPECT_EQ(0, status);
}
#endif

class cross_correlate : public ::testing::Test
{
protected:
    static const int num_sources = 277;
    static const int num_stations = 19;
    oskar_Mem *src_dir[3], *src_ext[3], *src_flux[4], *uvw[3];
    oskar_Telescope* tel;
    oskar_Jones* jones;

protected:
    void create_test_data(int precision, int location, int matrix)
    {
        int status = 0, type = 0;

        // Allocate memory for data structures.
        type = precision | OSKAR_COMPLEX;
        if (matrix) type |= OSKAR_MATRIX;
        jones = oskar_jones_create(type, location, num_stations, num_sources,
                &status);
        for (int i = 0; i < 3; ++i)
        {
            src_dir[i] = oskar_mem_create(
                    precision, location, num_sources, &status);
            src_ext[i] = oskar_mem_create(
                    precision, location, num_sources, &status);
            uvw[i] = oskar_mem_create(
                    precision, location, num_stations, &status);
        }
        for (int i = 0; i < 4; ++i)
        {
            src_flux[i] = oskar_mem_create(
                    precision, location, num_sources, &status);
        }
        tel = oskar_telescope_create(precision, location,
                num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Fill data structures with random data in sensible ranges.
#ifdef _MSC_VER
        srand(1);
#else
        srand(2);
#endif
        oskar_mem_random_range(oskar_jones_mem(jones), 1.0, 5.0, &status);
        oskar_mem_random_range(uvw[0], 1.0, 5.0, &status);
        oskar_mem_random_range(uvw[1], 1.0, 5.0, &status);
        oskar_mem_random_range(uvw[2], 1.0, 5.0, &status);
        oskar_mem_random_range(
                oskar_telescope_station_true_offset_ecef_metres(tel, 0),
                0.1, 1000.0, &status);
        oskar_mem_random_range(
                oskar_telescope_station_true_offset_ecef_metres(tel, 1),
                0.1, 1000.0, &status);
        oskar_mem_random_range(
                oskar_telescope_station_true_offset_ecef_metres(tel, 2),
                0.1, 1000.0, &status);
        oskar_mem_random_range(src_flux[0], 1.0, 2.0, &status);
        oskar_mem_random_range(src_flux[1], 0.1, 1.0, &status);
        oskar_mem_random_range(src_flux[2], 0.1, 0.5, &status);
        oskar_mem_random_range(src_flux[3], 0.1, 0.2, &status);
        oskar_mem_random_range(src_dir[0], 0.1, 0.9, &status);
        oskar_mem_random_range(src_dir[1], 0.1, 0.9, &status);
        oskar_mem_random_range(src_dir[2], 0.1, 0.9, &status);
        oskar_mem_random_range(src_ext[0], 0.1e-6, 0.2e-6, &status);
        oskar_mem_random_range(src_ext[1], 0.1e-6, 0.2e-6, &status);
        oskar_mem_random_range(src_ext[2], 0.1e-6, 0.2e-6, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    void destroy_test_data()
    {
        int status = 0;
        oskar_jones_free(jones, &status);
        for (int i = 0; i < 3; ++i)
        {
            oskar_mem_free(src_dir[i], &status);
            oskar_mem_free(src_ext[i], &status);
            oskar_mem_free(uvw[i], &status);
        }
        for (int i = 0; i < 4; ++i)
        {
            oskar_mem_free(src_flux[i], &status);
        }
        oskar_telescope_free(tel, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    void run_test(int prec1, int prec2, int loc1, int loc2, int matrix,
            int extended, double time_average, double freq_average)
    {
        int num_baselines = 0, status = 0, type = 0;
        oskar_Mem *vis1 = 0, *vis2 = 0;
        oskar_Timer *timer1 = 0, *timer2 = 0;
        double time1 = 0.0, time2 = 0.0, gast = 1.0, frequency = 100e6;
        int use_casa_phase_convention = 1;

        // Create the timers.
        timer1 = oskar_timer_create(loc1);
        timer2 = oskar_timer_create(loc2);

        // Run first part.
        create_test_data(prec1, loc1, matrix);
        num_baselines = oskar_telescope_num_baselines(tel);
        type = prec1 | OSKAR_COMPLEX;
        if (matrix) type |= OSKAR_MATRIX;
        vis1 = oskar_mem_create(type, loc1, num_baselines, &status);
        oskar_mem_clear_contents(vis1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_telescope_set_channel_bandwidth(tel, freq_average);
        oskar_telescope_set_time_average(tel, time_average);
        oskar_timer_start(timer1);
        oskar_cross_correlate(
                use_casa_phase_convention,
                extended,
                num_sources,
                jones,
                src_flux,
                src_dir,
                src_ext,
                tel,
                uvw,
                gast,
                frequency,
                0,
                vis1,
                &status
        );
        time1 = oskar_timer_elapsed(timer1);
        destroy_test_data();
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Run second part.
        create_test_data(prec2, loc2, matrix);
        num_baselines = oskar_telescope_num_baselines(tel);
        type = prec2 | OSKAR_COMPLEX;
        if (matrix) type |= OSKAR_MATRIX;
        vis2 = oskar_mem_create(type, loc2, num_baselines, &status);
        oskar_mem_clear_contents(vis2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_telescope_set_channel_bandwidth(tel, freq_average);
        oskar_telescope_set_time_average(tel, time_average);
        oskar_timer_start(timer2);
        oskar_cross_correlate(
                use_casa_phase_convention,
                extended,
                num_sources,
                jones,
                src_flux,
                src_dir,
                src_ext,
                tel,
                uvw,
                gast,
                frequency,
                0,
                vis2,
                &status
        );
        time2 = oskar_timer_elapsed(timer2);
        destroy_test_data();
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Destroy the timers.
        oskar_timer_free(timer1);
        oskar_timer_free(timer2);

        // Compare results.
        check_values(vis1, vis2);

        // Free memory.
        oskar_mem_free(vis1, &status);
        oskar_mem_free(vis2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Record properties for test.
        RecordProperty("SourceType", extended ? "Gaussian" : "Point");
        RecordProperty("JonesType", matrix ? "Matrix" : "Scalar");
        RecordProperty("TimeSmearing", time_average == 0.0 ? "off" : "on");
        RecordProperty("Prec1", prec1 == OSKAR_SINGLE ? "Single" : "Double");
        RecordProperty("Loc1", loc_to_str(loc1));
        RecordProperty("Time1_ms", int(time1 * 1000));
        RecordProperty("Prec2", prec2 == OSKAR_SINGLE ? "Single" : "Double");
        RecordProperty("Loc2", loc_to_str(loc2));
        RecordProperty("Time2_ms", int(time2 * 1000));

#ifdef ALLOW_PRINTING
        // Print times.
        printf("  > %s. %s sources. Time smearing %s. Bandwidth smearing %s.\n",
                matrix ? "Matrix" : "Scalar",
                extended ? "Gaussian" : "Point",
                time_average == 0.0 ? "off" : "on",
                freq_average == 0.0 ? "off" : "on");
        printf("    %s precision %s: %.2f ms, %s precision %s: %.2f ms\n",
                prec1 == OSKAR_SINGLE ? "Single" : "Double",
                loc_to_str(loc1),
                time1 * 1000.0,
                prec2 == OSKAR_SINGLE ? "Single" : "Double",
                loc_to_str(loc2),
                time2 * 1000.0);
#endif
    }
};


// CPU only.
// Check consistency between single and double precision.
TEST_F(cross_correlate, CPU)
{
    const int matrix_type[] = {0, 1};
    const int source_type[] = {0, 1};
    const double time_avg[] = {0.0, 10.0};
    const double freq_avg[] = {0.0, 1.e4};
    for (int i_matrix_type = 0; i_matrix_type < 2; ++i_matrix_type)
    {
        for (int i_source_type = 0; i_source_type < 2; ++i_source_type)
        {
            for (int i_time_avg = 0; i_time_avg < 2; ++i_time_avg)
            {
                for (int i_freq_avg = 0; i_freq_avg < 2; ++i_freq_avg)
                {
                    run_test(OSKAR_SINGLE, OSKAR_DOUBLE,
                            OSKAR_CPU, OSKAR_CPU,
                            matrix_type[i_matrix_type],
                            source_type[i_source_type],
                            time_avg[i_time_avg],
                            freq_avg[i_freq_avg]);
                }
            }
        }
    }
}

#ifdef OSKAR_HAVE_CUDA
// Check for consistency between CPU and CUDA versions.
TEST_F(cross_correlate, CUDA)
{
    const int precision[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    const int matrix_type[] = {0, 1};
    const int source_type[] = {0, 1};
    const double time_avg[] = {0.0, 10.0};
    const double freq_avg[] = {0.0, 1.e4};
    for (int i_prec = 0; i_prec < 2; ++i_prec)
    {
        for (int i_matrix_type = 0; i_matrix_type < 2; ++i_matrix_type)
        {
            for (int i_source_type = 0; i_source_type < 2; ++i_source_type)
            {
                for (int i_time_avg = 0; i_time_avg < 2; ++i_time_avg)
                {
                    for (int i_freq_avg = 0; i_freq_avg < 2; ++i_freq_avg)
                    {
                        run_test(precision[i_prec], precision[i_prec],
                                OSKAR_CPU, OSKAR_GPU,
                                matrix_type[i_matrix_type],
                                source_type[i_source_type],
                                time_avg[i_time_avg],
                                freq_avg[i_freq_avg]);
                    }
                }
            }
        }
    }
}
#endif

#ifdef OSKAR_HAVE_OPENCL
// Check for consistency between CPU and OpenCL versions.
TEST_F(cross_correlate, CL)
{
    const int precision[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    const int matrix_type[] = {0, 1};
    const int source_type[] = {0, 1};
    const double time_avg[] = {0.0, 10.0};
    const double freq_avg[] = {0.0, 1.e4};
    for (int i_prec = 0; i_prec < 2; ++i_prec)
    {
        for (int i_matrix_type = 0; i_matrix_type < 2; ++i_matrix_type)
        {
            for (int i_source_type = 0; i_source_type < 2; ++i_source_type)
            {
                for (int i_time_avg = 0; i_time_avg < 2; ++i_time_avg)
                {
                    for (int i_freq_avg = 0; i_freq_avg < 2; ++i_freq_avg)
                    {
                        run_test(precision[i_prec], precision[i_prec],
                                OSKAR_CPU, OSKAR_CL,
                                matrix_type[i_matrix_type],
                                source_type[i_source_type],
                                time_avg[i_time_avg],
                                freq_avg[i_freq_avg]);
                    }
                }
            }
        }
    }
}
#endif


TEST_F(cross_correlate, gaussian_uvw_point_limit_and_w_projection)
{
    const int nb = num_stations*(num_stations-1)/2, offset = 3;
    for (int single = 0; single < 2; ++single)
    for (int matrix = 0; matrix < 2; ++matrix)
    for (int casa = 0; casa < 2; ++casa)
    {
        int status = 0;
        const int real = single ? OSKAR_SINGLE : OSKAR_DOUBLE;
        create_test_data(real, OSKAR_CPU, matrix);
        oskar_telescope_set_time_average(tel, 0.);
        oskar_telescope_set_channel_bandwidth(tel, 0.);
        const int type = oskar_jones_type(jones);
        oskar_Mem* expected = oskar_mem_create(type, OSKAR_CPU, nb+offset, &status);
        oskar_Mem* actual = oskar_mem_create(type, OSKAR_CPU, nb+offset, &status);
        oskar_Mem* axes[6];
        const oskar_Mem *f[4], *d[3], *ext[3], *coords[3], *a[6];
        for (int k = 0; k < 4; ++k) f[k] = src_flux[k];
        for (int k = 0; k < 3; ++k)
        { d[k] = src_dir[k]; ext[k] = src_ext[k]; coords[k] = uvw[k]; }
        for (int k = 0; k < 6; ++k)
        { a[k] = axes[k] = oskar_mem_create(real, OSKAR_CPU, num_sources, &status); }
        oskar_cross_correlate(casa, 0, num_sources, jones, f, d, ext,
                tel, coords, 0., 100e6, offset, expected, &status);
        ASSERT_EQ(0, status);
        for (int shape = 0; shape < 2; ++shape)
        {
            oskar_mem_clear_contents(actual, &status);
            const double width = shape ? 0.4 : 0.;
            oskar_mem_set_value_real(axes[2], width, 0, num_sources, &status);
            oskar_cross_correlate_gaussian_uvw(casa, num_sources, jones,
                    f, a, tel, coords, 100e6, offset, actual, &status);
            ASSERT_EQ(0, status);
#ifdef OSKAR_HAVE_CUDA
            check_gaussian_cuda(casa, num_sources, jones, f, a, tel, coords, offset, actual);
#endif
            int index = offset;
            const int components = matrix ? 8 : 2;
            for (int q = 0; q < num_stations; ++q)
            for (int p = q+1; p < num_stations; ++p, ++index)
            {
                const double w = (oskar_mem_get_element(uvw[2], p, &status) -
                        oskar_mem_get_element(uvw[2], q, &status))*100e6/299792458.;
                const double envelope = std::exp(-w*w*width*width);
                for (int k = 0; k < components; ++k)
                {
                    const int j = components*index+k;
                    const double x = single ? oskar_mem_float(expected, &status)[j] :
                            oskar_mem_double(expected, &status)[j];
                    const double y = single ? oskar_mem_float(actual, &status)[j] :
                            oskar_mem_double(actual, &status)[j];
                    EXPECT_NEAR(x*envelope, y, (single ? 2e-6 : 3e-14)*std::fabs(x) +
                            (single ? 0.03 : 1e-9));
                }
            }
            for (int j = 0; j < components*offset; ++j)
                EXPECT_EQ(0., single ? oskar_mem_float(actual, &status)[j] :
                        oskar_mem_double(actual, &status)[j]);
        }
        /* A prefix of the source list must still use the full Jones stride. */
        oskar_Jones* packed = oskar_jones_create(type, OSKAR_CPU,
                num_stations, 2, &status);
        for (int st = 0; st < num_stations; ++st)
            oskar_mem_copy_contents(oskar_jones_mem(packed), oskar_jones_mem(jones),
                    st*2, st*num_sources, 2, &status);
        oskar_mem_clear_contents(axes[2], &status);
        oskar_mem_clear_contents(expected, &status);
        oskar_mem_clear_contents(actual, &status);
        oskar_cross_correlate(casa, 0, 2, packed, f, d, ext,
                tel, coords, 0., 100e6, offset, expected, &status);
        oskar_cross_correlate_gaussian_uvw(casa, 2, jones,
                f, a, tel, coords, 100e6, offset, actual, &status);
        ASSERT_EQ(0, status);
        check_values(actual, expected);
#ifdef OSKAR_HAVE_CUDA
        check_gaussian_cuda(casa, 2, jones, f, a, tel, coords, offset, actual);
#endif
        oskar_jones_free(packed, &status);
        /* The UV filter must leave excluded outputs untouched. */
        oskar_telescope_set_uv_filter(tel, 1e8, 1e9, "Wavelengths", &status);
        oskar_mem_clear_contents(actual, &status);
        oskar_cross_correlate_gaussian_uvw(casa, num_sources, jones,
                f, a, tel, coords, 100e6, offset, actual, &status);
        ASSERT_EQ(0, status);
#ifdef OSKAR_HAVE_CUDA
        check_gaussian_cuda(casa, num_sources, jones, f, a, tel, coords, offset, actual);
#endif
        for (int j = 0; j < (matrix ? 8 : 2)*(nb+offset); ++j)
            EXPECT_EQ(0., single ? oskar_mem_float(actual, &status)[j] :
                    oskar_mem_double(actual, &status)[j]);
        oskar_cross_correlate_gaussian_uvw(casa, num_sources+1, jones,
                f, a, tel, coords, 100e6, offset, actual, &status);
        EXPECT_EQ(OSKAR_ERR_DIMENSION_MISMATCH, status); status = 0;
        oskar_cross_correlate_gaussian_uvw(casa, num_sources, jones,
                f, a, tel, coords, 0., offset, actual, &status);
        EXPECT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status); status = 0;
        oskar_cross_correlate_gaussian_uvw(casa, num_sources, jones,
                f, a, tel, coords, 100e6, offset+1, actual, &status);
        EXPECT_EQ(OSKAR_ERR_DIMENSION_MISMATCH, status); status = 0;
        for (int k = 0; k < 6; ++k) oskar_mem_free(axes[k], &status);
        oskar_mem_free(actual, &status); oskar_mem_free(expected, &status);
        destroy_test_data();
    }
}

#if 0
TEST(KahanSum, sum)
{
    float sum_normal = 0.0f;
    int num = 100000;
    srand(0);
    for (int i = 0; i < num; ++i)
    {
        float r = (float)rand() / (float)RAND_MAX;
        sum_normal += r;
    }
    printf("Sum (normal, single): %.6f\n", sum_normal);

    float sum_kahan = 0.0f, kahan_guard = 0.0f;
    srand(0);
    for (int i = 0; i < num; ++i)
    {
        float r = (float)rand() / (float)RAND_MAX;
        oskar_kahan_sum_f(&sum_kahan, r, &kahan_guard);
    }
    printf("Sum (Kahan, single) : %.6f\n", sum_kahan);

    double sum_normal_double = 0.0;
    srand(0);
    for (int i = 0; i < num; ++i)
    {
        double r = (double)rand() / (double)RAND_MAX;
        sum_normal_double += r;
    }
    printf("Sum (normal, double): %.6f\n", sum_normal_double);
}
#endif
