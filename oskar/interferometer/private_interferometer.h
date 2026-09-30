/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_PRIVATE_INTERFEROMETER_H_
#define OSKAR_PRIVATE_INTERFEROMETER_H_

#include <binary/oskar_binary.h>
#include <interferometer/oskar_jones.h>
#include <log/oskar_log.h>
#include <mem/oskar_mem.h>
#include <ms/oskar_measurement_set.h>
#include <sky/oskar_sky.h>
#include <telescope/oskar_telescope.h>
#include <utility/oskar_thread.h>
#include <utility/oskar_timer.h>
#include <vis/oskar_vis_block.h>
#include <vis/oskar_vis_header.h>
#include <convert/oskar_astrometry.h>

/* Memory allocated per compute device (may be either CPU or GPU). */
struct DeviceData
{
    /* Host memory. */
    oskar_VisBlock* vis_block_cpu[2]; /* On host, for copy back & write. */
    oskar_Sky* astrometry_clip_cpu;
    oskar_StationWork* astrometry_work_cpu;
    oskar_Mem *lmn_cpu[3], *itrs_cpu[3], *gaussian_axes_cpu[6];
    oskar_Jones* R_cpu;

    /* Device memory. */
    int previous_chunk_index;
    oskar_VisBlock* vis_block;  /* Device memory block. */
    oskar_Mem *lmn[3], *uvw[3], *itrs[3];
    oskar_Sky* chunk;           /* The unmodified sky chunk being processed. */
    oskar_Sky* chunk_clip;      /* Copy of the chunk after horizon clipping. */
    oskar_Telescope* tel;       /* Telescope model, created as a copy. */
    oskar_Jones *J, *R, *E, *K;
    oskar_Mem *gains;
    oskar_StationWork* station_work;
    oskar_AstrometryState* astrometry_state;
    oskar_Mem* gaussian_axes[6]; /* Precision Gaussian factors in UVW. */

    /* Timers. */
    oskar_Timer* tmr_compute;   /* Total time spent filling vis blocks. */
    oskar_Timer* tmr_copy;      /* Time spent copying data. */
    oskar_Timer* tmr_clip;      /* Time spent in horizon clip. */
    oskar_Timer* tmr_correlate; /* Time spent correlating Jones matrices. */
    oskar_Timer* tmr_join;      /* Time spent combining Jones matrices. */
    oskar_Timer* tmr_E;         /* Time spent evaluating E-Jones. */
    oskar_Timer* tmr_K;         /* Time spent evaluating K-Jones. */
    oskar_Timer* tmr_astrometry; /* Host geometry, including bulk uploads. */
};
typedef struct DeviceData DeviceData;


struct oskar_Interferometer
{
    /* Settings. */
    int prec, num_devices, num_gpus_avail, dev_loc, num_gpus, *gpu_ids;
    int num_channels, num_time_steps;
    int max_sources_per_chunk, max_times_per_block, max_channels_per_block;
    int apply_horizon_clip, force_polarised_ms, zero_failed_gaussians;
    int coords_only, ignore_w_components;
    int casa_phase_convention;
    double freq_start_hz, freq_inc_hz, time_start_mjd_utc, time_inc_sec;
    double source_min_jy, source_max_jy;
    double ms_dish_diameter;
    char correlation_type, *vis_name, *ms_name, *settings_path;
    oskar_Astrometry* astrometry;
    int astrometry_time_samples, astrometry_frequency_samples;

    /* State. */
    int init_sky, work_unit_index;
    oskar_Mutex* mutex;
    oskar_Barrier* barrier;
    oskar_Log* log;

    /* Sky model and telescope model. */
    int num_sources_total, num_sky_chunks;
    oskar_Sky** sky_chunks;
    oskar_Telescope* tel;

    /* Output data and file handles. */
    oskar_VisHeader* header;
    oskar_MeasurementSet* ms;
    oskar_Binary* vis;
    oskar_Mem *temp;
    oskar_Timer* tmr_sim;   /* The total time for the simulation. */
    oskar_Timer* tmr_write; /* The time spent writing vis blocks. */

    /* Array of DeviceData structures, one per compute device. */
    DeviceData* d;
};
#ifndef OSKAR_INTERFEROMETER_TYPEDEF_
#define OSKAR_INTERFEROMETER_TYPEDEF_
typedef struct oskar_Interferometer oskar_Interferometer;
#endif

void oskar_interferometer_check_astrometry(oskar_Interferometer* h, int* status);
void oskar_interferometer_astrometry_prepare(oskar_Interferometer* h,
        DeviceData* d, const oskar_Sky* sky, int* status);
void oskar_interferometer_astrometry_uvw(const oskar_Telescope* tel,
        const oskar_AstrometryState* state, int use_true, int ignore_w,
        int offset, oskar_Mem* u, oskar_Mem* v, oskar_Mem* w, int* status);

#endif /* include guard */
