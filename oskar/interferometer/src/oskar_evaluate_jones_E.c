/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_any_to_enu_directions.h"
#include "convert/oskar_convert_enu_directions_to_theta_phi.h"
#include "convert/oskar_convert_enu_directions_to_relative_directions.h"
#include "telescope/private_telescope.h"
#include "convert/oskar_convert_mjd_to_gast_fast.h"
#include "interferometer/oskar_evaluate_jones_E.h"
#include "interferometer/oskar_jones_accessors.h"
#include "math/oskar_cmath.h"
#include "telescope/station/oskar_blank_below_horizon.h"
#include "telescope/station/private_station_work.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

static void oskar_evaluate_element_beams_harp(
        oskar_Harp* harp_data,
        int coord_type,
        int num_points,
        const oskar_Mem* const source_coords[3],
        double ref_lon_rad,
        double ref_lat_rad,
        oskar_Telescope* tel,
        double time_mjd_utc,
        double frequency_hz,
        oskar_StationWork* work,
        oskar_Mem* beam,
        int* status)
{
    int dim = 0, feed = 0, i_station = 0;
    oskar_Mem *enu[] = {0, 0, 0}, *theta = 0, *phi_x = 0, *phi_y = 0;

    /* Get source ENU coordinates. */
    for (dim = 0; dim < 3; ++dim)
    {
        enu[dim] = oskar_station_work_enu_direction(
                work, dim, num_points + 1, status);
    }
    const double gast_rad = oskar_convert_mjd_to_gast_fast(time_mjd_utc);
    const double lst_rad = gast_rad + oskar_telescope_lon_rad(tel);
    const double lat_rad = oskar_telescope_lat_rad(tel);
    oskar_convert_any_to_enu_directions(coord_type,
            num_points, source_coords, ref_lon_rad, ref_lat_rad,
            lst_rad, lat_rad, enu, status);

    /* Get theta and phi directions. */
    theta = work->theta_modified;
    phi_x = work->phi_x;
    phi_y = work->phi_y;
    oskar_mem_ensure(theta, num_points, status);
    oskar_mem_ensure(phi_x, num_points, status);
    oskar_mem_ensure(phi_y, num_points, status);
    oskar_convert_enu_directions_to_theta_phi(
                    0, num_points, enu[0], enu[1], enu[2], 0,
                    0.0, M_PI / 2.0, theta, phi_x, phi_y, status);
    oskar_harp_evaluate_smodes(harp_data, num_points, theta, phi_x,
            work->poly, work->ee, work->qq, work->dd,
            work->pth, work->pph, status);

    /* Evaluate all the element beams. */
    const int num_stations = oskar_telescope_num_stations(tel);
    for (feed = 0; feed < 2; ++feed)
    {
        oskar_harp_evaluate_element_beams(harp_data,
                num_points, theta, phi_x, frequency_hz,
                feed, num_stations,
                oskar_telescope_station_true_enu_metres_const(tel, 0),
                oskar_telescope_station_true_enu_metres_const(tel, 1),
                oskar_telescope_station_true_enu_metres_const(tel, 2),
                work->pth, work->pph, work->phase_fac,
                0, beam, status);
    }
    for (i_station = 0; i_station < num_stations; ++i_station)
    {
        const int offset_out = i_station * num_points;
        oskar_blank_below_horizon(0, num_points, enu[2],
                offset_out, beam, status);
    }
}


/* Sample the existing TEC evaluator on each physical line of sight. The
 * shared station models carry the reference station position; the telescope
 * coordinate arrays retain the positions of every physical station. */
static void apply_physical_station_phase(oskar_Jones* E, int coord_type,
        int num_points, const oskar_Mem* const source_coords[3],
        double ref_lon_rad, double ref_lat_rad, const oskar_Telescope* tel,
        int time_index, double time_start_mjd_utc, double time_mjd_utc,
        double frequency_hz, oskar_StationWork* work, int* status)
{
    if (*status || !num_points) return;
    const int tracking = tel->phase_centre_coord_type == OSKAR_COORDS_RADEC;
    if (!tracking && tel->phase_centre_coord_type != OSKAR_COORDS_AZEL)
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    const double gast = oskar_convert_mjd_to_gast_fast(time_mjd_utc);
    const double* lon = oskar_mem_double_const(tel->station_true_geodetic_rad[0], status);
    const double* lat = oskar_mem_double_const(tel->station_true_geodetic_rad[1], status);
    const double no_rotation[3] = {0., 0., 0.};
    oskar_Mem *enu[3], *lmn[3], *host_offsets[3] = {0, 0, 0};
    const oskar_Mem* offsets[3];
    for (int j = 0; j < 3; ++j)
    {
        offsets[j] = tel->station_true_offset_ecef_metres[j];
        if (oskar_mem_location(offsets[j]) != OSKAR_CPU)
        {
            host_offsets[j] = oskar_mem_create_copy(offsets[j], OSKAR_CPU, status);
            offsets[j] = host_offsets[j];
        }
        enu[j] = oskar_station_work_enu_direction(work, j, num_points, status);
        lmn[j] = oskar_station_work_lmn_direction(work, j, num_points, status);
    }
    for (int i = 0; i < tel->num_stations && !*status; ++i)
    {
        const double lst = gast + lon[i];
        const double gha = tracking ? gast - tel->phase_centre_rad[0] : -lon[i];
        const double dec = tracking ? tel->phase_centre_rad[1] : lat[i];
        const double x = oskar_mem_get_element(offsets[0], i, status);
        const double y = oskar_mem_get_element(offsets[1], i, status);
        const double z = oskar_mem_get_element(offsets[2], i, status);
        const double u = x * sin(gha) + y * cos(gha);
        const double v = z * cos(dec) - x * cos(gha) * sin(dec) + y * sin(gha) * sin(dec);
        oskar_convert_any_to_enu_directions(coord_type, num_points, source_coords,
                ref_lon_rad, ref_lat_rad, lst, lat[i], enu, status);
        if (tracking)
        {
            oskar_convert_enu_directions_to_relative_directions(
                    0, num_points, enu[0], enu[1], enu[2],
                    lst - tel->phase_centre_rad[0], dec, lat[i],
                    0, lmn[0], lmn[1], lmn[2], status);
        }
        oskar_Mem* beam = oskar_jones_mem(E);
        const oskar_Mem* phase = oskar_station_work_evaluate_tec_screen(work,
                num_points, tracking ? lmn[0] : enu[0], tracking ? lmn[1] : enu[1],
                enu[0], enu[1], enu[2], u, v, time_index, time_start_mjd_utc,
                time_mjd_utc, frequency_hz, no_rotation, beam, status);
        const size_t offset = (size_t)i * oskar_jones_num_sources(E);
        if (!*status) oskar_mem_multiply(beam, beam, phase, offset, offset, 0,
                num_points, status);
    }
    for (int j = 0; j < 3; ++j) oskar_mem_free(host_offsets[j], status);
}

/* NOLINTNEXTLINE(readability-identifier-naming) */
void oskar_evaluate_jones_E(
        oskar_Jones* E,
        int coord_type,
        int num_points,
        const oskar_Mem* const source_coords[3],
        double ref_lon_rad,
        double ref_lat_rad,
        oskar_Telescope* tel,
        int time_index,
        double time_start_mjd_utc,
        double time_mjd_utc,
        double frequency_hz,
        oskar_StationWork* work,
        int* status)
{
    int i = 0, j = 0;
    if (*status) return;
    const int num_stations = oskar_telescope_num_stations(tel);
    const int num_sources = oskar_jones_num_sources(E);
    if (num_stations == 0)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }
    if (num_stations != oskar_jones_num_stations(E))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Check if HARP data exist. */
    oskar_Harp* harp_data = oskar_telescope_harp_data(tel, frequency_hz);
    if (harp_data)
    {
        oskar_evaluate_element_beams_harp(harp_data, coord_type, num_points,
                source_coords, ref_lon_rad, ref_lat_rad, tel, time_mjd_utc,
                frequency_hz, work, oskar_jones_mem(E), status);
    }
    else if (!oskar_telescope_allow_station_beam_duplication(tel))
    {
        /* Evaluate all the station beams. */
        for (i = 0; i < num_stations; ++i)
        {
            oskar_station_beam(
                    oskar_telescope_station(tel, i),
                    work, coord_type, num_points, source_coords,
                    ref_lon_rad, ref_lat_rad,
                    oskar_telescope_phase_centre_coord_type(tel),
                    oskar_telescope_phase_centre_longitude_rad(tel),
                    oskar_telescope_phase_centre_latitude_rad(tel),
                    time_index, time_start_mjd_utc, time_mjd_utc, frequency_hz,
                    i * num_sources, oskar_jones_mem(E), status);
        }
    }
    else
    {
        /* Keep track of which station models have been evaluated. */
        int num_models_evaluated = 0;
        int *models_evaluated = 0, *model_offsets = 0;
        const int* type_map = oskar_mem_int_const(
                oskar_telescope_station_type_map_const(tel), status);
        for (i = 0; i < num_stations; ++i)
        {
            int station_to_copy = -1;
            const int station_model_type = type_map[i];
            for (j = 0; j < num_models_evaluated; ++j)
            {
                if (models_evaluated[j] == station_model_type)
                {
                    station_to_copy = model_offsets[j];
                    break;
                }
            }
            if (station_to_copy >= 0)
            {
                oskar_mem_copy_contents(
                        oskar_jones_mem(E), oskar_jones_mem(E),
                        (size_t)(i * num_sources),               /* Dest. */
                        (size_t)(station_to_copy * num_sources), /* Source. */
                        (size_t)num_sources, status);
            }
            else
            {
                oskar_station_beam(
                        oskar_telescope_station(tel, station_model_type),
                        work, coord_type, num_points, source_coords,
                        ref_lon_rad, ref_lat_rad,
                        oskar_telescope_phase_centre_coord_type(tel),
                        oskar_telescope_phase_centre_longitude_rad(tel),
                        oskar_telescope_phase_centre_latitude_rad(tel),
                        time_index, time_start_mjd_utc, time_mjd_utc,
                        frequency_hz, i * num_sources, oskar_jones_mem(E),
                        status
                );
                num_models_evaluated++;
                models_evaluated = (int*) realloc(models_evaluated,
                        num_models_evaluated * sizeof(int));
                model_offsets = (int*) realloc(model_offsets,
                        num_models_evaluated * sizeof(int));
                models_evaluated[num_models_evaluated - 1] = station_model_type;
                model_offsets[num_models_evaluated - 1] = i;
            }
        }
        free(models_evaluated);
        free(model_offsets);
    }
    if (work->screen_type == 'P')
    {
        apply_physical_station_phase(E, coord_type, num_points, source_coords,
                ref_lon_rad, ref_lat_rad, tel, time_index, time_start_mjd_utc,
                time_mjd_utc, frequency_hz, work, status);
    }
}

#ifdef __cplusplus
}
#endif
