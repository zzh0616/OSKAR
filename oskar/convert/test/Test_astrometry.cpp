/* See the LICENSE file at the top-level directory of this distribution. */
#include <gtest/gtest.h>
#include "convert/oskar_astrometry.h"
#include "interferometer/oskar_interferometer.h"
#include "vis/oskar_vis_header.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <vector>
#ifdef OSKAR_HAVE_ERFA
#include <erfa.h>
#endif

namespace {
const char* path = "oskar_astrometry_test_eop.txt";
const char* rows = "# MJD DUT1 xp yp DAT\n59000 0.1 0.2 -0.1 37\n"
        "59001 0.11 0.21 -0.09 37\n59002 0.12 0.22 -0.08 37\n";

class Astrometry : public ::testing::Test {
protected:
    int status;
    void SetUp() { status = 0; write(rows); }
    void TearDown() { std::remove(path); }
    void write(const char* contents) { std::ofstream f(path); f << contents; }
};

#ifdef OSKAR_HAVE_ERFA
TEST_F(Astrometry, apparent_matches_precise_rotation)
{
    oskar_Astrometry* a = oskar_astrometry_create(1, path, &status);
    oskar_AstrometryState* s = oskar_astrometry_state_create(a, 59000.25,
            0., 1.1, -0.5, &status);
    ASSERT_EQ(0, status);
    double t1, t2, tt1, tt2, u1, u2, pm[3][3], r[3][3], expected[3];
    eraUtctai(2459000.5, .25, &t1, &t2);
    eraTaitt(t1, t2, &tt1, &tt2);
    eraUtcut1(2459000.5, .25, .1025, &u1, &u2);
    const double gast = eraGst06a(u1, u2, tt1, tt2);
    eraPom00(.2025*M_PI/(180.*3600.), -.0975*M_PI/(180.*3600.),
            eraSp00(tt1, tt2), pm);
    eraIr(r); eraRz(gast, r); eraRxr(pm, r, r);
    double sky[3]; eraS2c(1.1, -.5, sky); eraRxp(r, sky, expected);
    const double* b = oskar_astrometry_uvw_matrix(s);
    for (int i = 0; i < 3; ++i) EXPECT_NEAR(expected[i], b[6+i], 5e-15);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            double v = 0.;
            for (int k = 0; k < 3; ++k) v += b[3*i+k]*b[3*j+k];
            EXPECT_NEAR(i == j ? 1. : 0., v, 3e-14);
        }
    oskar_astrometry_state_free(s); oskar_astrometry_free(a);
}

TEST_F(Astrometry, icrs_includes_aberration_not_just_c2t)
{
    oskar_Astrometry* a = oskar_astrometry_create(2, path, &status);
    oskar_AstrometryState* s = oskar_astrometry_state_create(a, 59000.25,
            0., 1.1, -.5, &status);
    ASSERT_EQ(0, status);
    double ta1, ta2, tt1, tt2, u1, u2, ri, di, eo, pm[3][3], r[3][3], p[3], out[3];
    eraUtctai(2459000.5, .25, &ta1, &ta2);
    eraTaitt(ta1, ta2, &tt1, &tt2);
    eraUtcut1(2459000.5, .25, .1025, &u1, &u2);
    eraAtci13(1.1, -.5, 0., 0., 0., 0., tt1, tt2, &ri, &di, &eo);
    eraPom00(.2025*M_PI/(180.*3600.), -.0975*M_PI/(180.*3600.),
            eraSp00(tt1, tt2), pm);
    eraIr(r); eraRz(eraEra00(u1,u2), r); eraRxr(pm, r, r);
    eraS2c(ri, di, p); eraRxp(r, p, out);
    const double* b = oskar_astrometry_uvw_matrix(s);
    for (int k = 0; k < 3; ++k) EXPECT_NEAR(out[k], b[6+k], 5e-15);
    oskar_astrometry_state_free(s); oskar_astrometry_free(a);
}

TEST_F(Astrometry, gaussian_axes_apparent_and_poles)
{
    const double scale = M_PI/(2.*std::sqrt(std::log(2.)));
    oskar_Astrometry* a = oskar_astrometry_create(1, path, &status);
    for (int pole = -1; pole <= 1; ++pole)
    {
        const double dec = pole*M_PI/2.;
        oskar_AstrometryState* s = oskar_astrometry_state_create(a,
                59000.25, 0., 1.1, dec, &status);
        ASSERT_EQ(0, status);
        for (int angle = 0; angle < 4; ++angle)
        {
            const double pa = angle*M_PI/4.;
            double axes[6];
            oskar_astrometry_gaussian_axes(s, 1.1, dec, 1e-4, 2e-5,
                    pa, axes, &status);
            ASSERT_EQ(0, status);
            EXPECT_NEAR(1e-4*scale*std::sin(pa), axes[0], 2e-15);
            EXPECT_NEAR(1e-4*scale*std::cos(pa), axes[1], 2e-15);
            EXPECT_NEAR(0., axes[2], 2e-15);
            EXPECT_NEAR(2e-5*scale*std::cos(pa), axes[3], 2e-15);
            EXPECT_NEAR(-2e-5*scale*std::sin(pa), axes[4], 2e-15);
            EXPECT_NEAR(0., axes[5], 2e-15);
        }
        double axes[6];
        oskar_astrometry_gaussian_axes(s, 1.1, dec, 0., 0., 0., axes, &status);
        for (int k = 0; k < 6; ++k) EXPECT_EQ(0., axes[k]);
        oskar_astrometry_gaussian_axes(s, 1.1, dec, -1., 0., 0., axes, &status);
        EXPECT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status); status = 0;
        oskar_astrometry_state_free(s);
    }
    oskar_astrometry_free(a);
}

TEST_F(Astrometry, gaussian_off_axis_jacobian_not_normalised)
{
    for (int frame = 1; frame <= 3; ++frame)
    {
        oskar_Astrometry* a = oskar_astrometry_create(frame, path, &status);
        oskar_AstrometryState* s = oskar_astrometry_state_create(a,
                59000.25, 0., 1.1, -0.5, &status);
        ASSERT_EQ(0, status);
        const double ra = 1.5, dec = -.3, step = 1e-6;
        double axes[6], plus[3], minus[3];
        oskar_astrometry_gaussian_axes(s, ra, dec, 1., 0., 0., axes, &status);
        oskar_astrometry_direction(s, ra, dec+step, plus, 0, &status);
        oskar_astrometry_direction(s, ra, dec-step, minus, 0, &status);
        ASSERT_EQ(0, status);
        const double* basis = oskar_astrometry_uvw_matrix(s);
        const double scale = M_PI/(2.*std::sqrt(std::log(2.)));
        for (int k = 0; k < 3; ++k)
        {
            double expected = 0.;
            for (int j = 0; j < 3; ++j)
                expected += basis[3*k+j]*(plus[j]-minus[j])/(2.*step);
            EXPECT_NEAR(expected*scale, axes[k], 2e-9);
        }
        EXPECT_GT(std::fabs(axes[2]), .1);
        const double magnification = std::sqrt(axes[0]*axes[0] +
                axes[1]*axes[1] + axes[2]*axes[2])/scale;
        if (frame != 1) { EXPECT_GT(std::fabs(magnification-1.), 1e-6); }
        oskar_astrometry_state_free(s); oskar_astrometry_free(a);
    }
}

TEST_F(Astrometry, fk5_and_poles)
{
    oskar_Astrometry* fk = oskar_astrometry_create(3, path, &status);
    oskar_Astrometry* ic = oskar_astrometry_create(2, path, &status);
    for (int i = -1; i <= 1; ++i)
    {
        const double dec = i*M_PI/2.;
        double ra_ic, dec_ic, p[3], q[3];
        eraFk5hz(1., dec, 2451545., 0., &ra_ic, &dec_ic);
        oskar_AstrometryState* f = oskar_astrometry_state_create(fk, 59000.25,
                0., 1., dec, &status);
        oskar_AstrometryState* c = oskar_astrometry_state_create(ic, 59000.25,
                0., ra_ic, dec_ic, &status);
        ASSERT_EQ(0, status);
        oskar_astrometry_direction(f, 1., dec, p, 0, &status);
        oskar_astrometry_direction(c, ra_ic, dec_ic, q, 0, &status);
        for (int k = 0; k < 3; ++k) EXPECT_NEAR(p[k], q[k], 5e-15);
        oskar_astrometry_state_free(f); oskar_astrometry_state_free(c);
    }
    oskar_astrometry_free(fk); oskar_astrometry_free(ic);
}

TEST_F(Astrometry, missing_coverage_is_an_error)
{
    oskar_Astrometry* a = oskar_astrometry_create(2, path, &status);
    oskar_AstrometryState* s = oskar_astrometry_state_create(a, 58999.,
            0., 1., -.5, &status);
    EXPECT_EQ(OSKAR_ERR_OUT_OF_RANGE, status); EXPECT_EQ((void*)0, s);
    oskar_astrometry_free(a);
}

TEST_F(Astrometry, malformed_eop_and_leaps_are_rejected)
{
    const char* invalid[] = {
        "59000 nan 0 0 37\n59001 0 0 0 37\n",
        "59000 0 0 0 36\n59001 0 0 0 36\n",
        "59001 0 0 0 37\n59000 0 0 0 37\n",
        "59000 0 0 0 37 garbage\n59001 0 0 0 37\n",
        "59000 0 0 0 37\n"};
    for (size_t i = 0; i < sizeof(invalid)/sizeof(invalid[0]); ++i)
    {
        write(invalid[i]); status = 0;
        oskar_Astrometry* a = oskar_astrometry_create(2, path, &status);
        EXPECT_NE(0, status); EXPECT_EQ((void*)0, a); oskar_astrometry_free(a);
    }
}

TEST_F(Astrometry, missing_day_is_not_interpolated)
{
    write("59000 0 0 0 37\n59002 0 0 0 37\n");
    oskar_Astrometry* a = oskar_astrometry_create(2, path, &status);
    oskar_AstrometryState* s = oskar_astrometry_state_create(a, 59001.,
            0., 1., -.5, &status);
    EXPECT_EQ(OSKAR_ERR_OUT_OF_RANGE, status);
    oskar_astrometry_state_free(s); oskar_astrometry_free(a);
}

TEST_F(Astrometry, leap_second_interpolates_continuous_ut1_tai)
{
    write("57753 -0.4 0 0 36\n57754 0.6 0 0 37\n57755 0.6 0 0 37\n");
    oskar_Astrometry* a = oskar_astrometry_create(1, path, &status);
    oskar_AstrometryState* before = oskar_astrometry_state_create(a,
            57753., 86399., 1., -.5, &status);
    oskar_AstrometryState* after = oskar_astrometry_state_create(a,
            57753., 86402., 1., -.5, &status);
    ASSERT_EQ(0, status);
    EXPECT_DOUBLE_EQ(36., oskar_astrometry_tai_minus_utc(before));
    EXPECT_DOUBLE_EQ(37., oskar_astrometry_tai_minus_utc(after));
    double delta = oskar_astrometry_gast(after)-oskar_astrometry_gast(before);
    EXPECT_NEAR(3.*7.292115e-5, delta, 2e-10);
    oskar_astrometry_state_free(before); oskar_astrometry_state_free(after);
    oskar_astrometry_free(a);
}
#else
TEST_F(Astrometry, unavailable_backend_never_falls_back)
{
    oskar_Astrometry* a = oskar_astrometry_create(2, path, &status);
    EXPECT_EQ(OSKAR_ERR_FUNCTION_NOT_AVAILABLE, status); EXPECT_EQ((void*)0, a);
}
#endif

TEST_F(Astrometry, header_round_trip_and_legacy_default)
{
    const char* filename = "oskar_astrometry_test.vis";
    oskar_VisHeader* h = oskar_vis_header_create(OSKAR_DOUBLE_COMPLEX,
            OSKAR_DOUBLE, 1, 1, 1, 1, 2, 0, 1, &status);
    EXPECT_EQ(0, oskar_vis_header_astrometry_frame(h));
    oskar_vis_header_set_astrometry(h, 2, rows, &status);
    oskar_Binary* writer = oskar_vis_header_write(h, filename, &status);
    oskar_binary_free(writer);
    oskar_Binary* reader = oskar_binary_create(filename, 'r', &status);
    oskar_VisHeader* copy = oskar_vis_header_read(reader, &status);
    ASSERT_EQ(0, status);
    EXPECT_EQ(2, oskar_vis_header_astrometry_frame(copy));
    EXPECT_STREQ(rows, oskar_vis_header_astrometry_eop(copy));
    oskar_vis_header_free(h, &status); oskar_vis_header_free(copy, &status);
    oskar_binary_free(reader); std::remove(filename);
}

TEST_F(Astrometry, legacy_rejects_inertial_frame_mislabel)
{
    oskar_Interferometer* h = oskar_interferometer_create(OSKAR_DOUBLE, &status);
    oskar_interferometer_set_astrometry(h, "legacy", "icrs", "", 1, 1, &status);
    EXPECT_NE(0, status);
    status = 0; oskar_interferometer_free(h, &status);
}

TEST_F(Astrometry, frame_tag_requires_eop_payload)
{
    const char* filename = "oskar_astrometry_incomplete.vis";
    oskar_VisHeader* h = oskar_vis_header_create(OSKAR_DOUBLE_COMPLEX,
            OSKAR_DOUBLE, 1, 1, 1, 1, 2, 0, 1, &status);
    oskar_Binary* writer = oskar_vis_header_write(h, filename, &status);
    oskar_binary_write_ext_int(writer, "OSKAR_ASTROMETRY", "FRAME_V1",
            0, 2, &status);
    oskar_binary_free(writer);
    oskar_Binary* reader = oskar_binary_create(filename, 'r', &status);
    oskar_VisHeader* copy = oskar_vis_header_read(reader, &status);
    EXPECT_NE(0, status);
    status = 0;
    oskar_vis_header_free(h, &status); oskar_vis_header_free(copy, &status);
    oskar_binary_free(reader); std::remove(filename);
}

TEST_F(Astrometry, quadrature_integrates_polynomial_moments)
{
    for (int n = 1; n <= 32; ++n)
    {
        std::vector<double> x(n), w(n);
        oskar_astrometry_quadrature(n, x.data(), w.data(), &status);
        ASSERT_EQ(0, status);
        for (int power = 0; power < 2*n; ++power)
        {
            double sum = 0.;
            for (int j = 0; j < n; ++j)
            {
                EXPECT_GT(w[j], 0.);
                EXPECT_LT(std::abs(x[j]), .5);
                sum += w[j]*std::pow(2.*x[j], power);
            }
            EXPECT_NEAR(power % 2 ? 0. : 1./(power + 1.), sum, 2e-14);
        }
    }
    double x = 0., w = 0.;
    oskar_astrometry_quadrature(0, &x, &w, &status);
    EXPECT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
}
}
