/* See LICENSE at the top level of this distribution. */
#include <gtest/gtest.h>
#include "convert/oskar_convert_mjd_to_gast_fast.h"
#include "interferometer/oskar_evaluate_jones_E.h"
#include "interferometer/oskar_jones.h"
#include "telescope/oskar_telescope.h"
#include <cmath>
#include <cstdio>

static void check_station_phase(int location)
{
    // Three stations share one nominal beam. At transit on the equator
    // the physical (u,v) shifts are exactly the ECEF (y,z) shifts.
    const int n = 32, ns = 3, nd = 2;
    const double mjd = 59000., ra = oskar_convert_mjd_to_gast_fast(mjd);
    const int du[ns] = {0, 3, -2}, dv[ns] = {0, 1, -3};
    const int dl[nd] = {0, 4}, dm[nd] = {0, -2};
    const char* file = location == OSKAR_CPU ?
            "physical_station_tec_cpu.fits" : "physical_station_tec_gpu.fits";
    int err = 0;
    oskar_Mem* cube = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, n*n*3, &err);
    double* pix = oskar_mem_double(cube, &err);
    for (int t = 0; t < 3; ++t)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x)
                pix[(t*n+y)*n+x] = t * (x*x + 3*y) * 1.e-4;
    oskar_mem_write_fits_cube(cube, file, n, n, 3, -1, &err);
    oskar_Telescope* host = oskar_telescope_create(OSKAR_DOUBLE, OSKAR_CPU, ns, &err);
    oskar_telescope_set_allow_station_beam_duplication(host, 1);
    oskar_telescope_resize_station_array(host, 1, &err);
    oskar_Station* prototype = oskar_telescope_station(host, 0);
    oskar_station_set_station_type(prototype, OSKAR_STATION_TYPE_GAUSSIAN_BEAM);
    oskar_station_set_phase_centre(prototype, OSKAR_COORDS_RADEC, ra, 0.);
    for (int feed = 0; feed < 2; ++feed)
        oskar_station_set_gaussian_beam_values(prototype, .4, .4, 0., 0., feed);
    oskar_telescope_set_phase_centre(host, OSKAR_COORDS_RADEC, ra, 0.);
    for (int i = 0; i < ns; ++i)
    {
        const double pos[] = {0., du[i]*10., dv[i]*10.};
        const double enu[] = {pos[1], pos[2], 0.};
        const double geo[] = {pos[1]/6378137., pos[2]/6378137., 0.};
        oskar_telescope_set_station_coords(host, i, geo, pos, pos, enu, enu, &err);
    }
    oskar_telescope_set_station_ids_and_coords(host, &err);
    oskar_Telescope* tel = oskar_telescope_create_copy(host, location, &err);
    oskar_Mem* dirs[3];
    for (int j = 0; j < 3; ++j)
    {
        dirs[j] = oskar_mem_create(OSKAR_DOUBLE, location, nd, &err);
        for (int d = 0; d < nd; ++d)
        {
            const double l = dl[d]*.01, m = dm[d]*.01;
            const double val[] = {l, m, std::sqrt(1.-l*l-m*m)};
            oskar_mem_set_element_real(dirs[j], d, val[j], &err);
        }
    }
    const oskar_Mem* coords[] = {dirs[0], dirs[1], dirs[2]};
    oskar_Jones* E = oskar_jones_create(OSKAR_DOUBLE_COMPLEX_MATRIX,
            location, ns, nd, &err);
    oskar_StationWork* work = oskar_station_work_create(OSKAR_DOUBLE, location, &err);
    ASSERT_EQ(0, err);
    oskar_station_work_set_tec_screen_path(work, file);
    for (int f = 0; f < 2; ++f)
    {
        const double hz = (100. + 50.*f)*1.e6;
        oskar_station_work_set_tec_screen_common_params(work, 'N', 1., 10., 0.);
        oskar_evaluate_jones_E(E, OSKAR_COORDS_REL_DIR, nd, coords, ra, 0.,
                tel, 0, mjd, mjd, hz, work, &err);
        oskar_Mem* nominal = oskar_mem_create_copy(oskar_jones_mem(E), OSKAR_CPU, &err);
        const double4c* ref = oskar_mem_double4c_const(nominal, &err);
        ASSERT_EQ(0, err);
        EXPECT_GT(ref[0].a.x, .9);
        EXPECT_LT(ref[1].a.x, ref[0].a.x);
        oskar_station_work_set_tec_screen_common_params(work, 'P', 1., 10., 0.);
        for (int t = 0; t < 3; ++t)
        {
            oskar_evaluate_jones_E(E, OSKAR_COORDS_REL_DIR, nd, coords, ra, 0.,
                    tel, t, mjd, mjd, hz, work, &err);
            oskar_Mem* host_e = oskar_mem_create_copy(oskar_jones_mem(E), OSKAR_CPU, &err);
            const double4c* measured = oskar_mem_double4c_const(host_e, &err);
            ASSERT_EQ(0, err);
            for (int i = 0; i < ns; ++i)
                for (int d = 0; d < nd; ++d)
                {
                    const int x = n/2 + du[i] + dl[d], y = n/2 + dv[i] + dm[d];
                    const double phase = -8.44797245e9 * pix[(t*n+y)*n+x] / hz;
                    const double4c a = ref[i*nd+d], b = measured[i*nd+d];
                    const double2 in[] = {a.a, a.b, a.c, a.d};
                    const double2 out[] = {b.a, b.b, b.c, b.d};
                    for (int k = 0; k < 4; ++k)
                    {
                        EXPECT_NEAR(in[k].x*cos(phase)-in[k].y*sin(phase), out[k].x, 2e-12);
                        EXPECT_NEAR(in[k].x*sin(phase)+in[k].y*cos(phase), out[k].y, 2e-12);
                    }
                }
            oskar_mem_free(host_e, &err);
        }
        oskar_mem_free(nominal, &err);
    }
    for (int j = 0; j < 3; ++j) oskar_mem_free(dirs[j], &err);
    oskar_jones_free(E, &err);
    oskar_station_work_free(work, &err);
    oskar_telescope_free(tel, &err);
    oskar_telescope_free(host, &err);
    oskar_mem_free(cube, &err);
    std::remove(file);
    ASSERT_EQ(0, err);
}

TEST(physical_station_tec, cpu) { check_station_phase(OSKAR_CPU); }
#ifdef OSKAR_HAVE_CUDA
TEST(physical_station_tec, gpu) { check_station_phase(OSKAR_GPU); }
#endif
