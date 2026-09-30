/*
 * Copyright (c) 2012-2020, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OSKAR_INTERFEROMETER_H_
#define OSKAR_INTERFEROMETER_H_

/**
 * @file oskar_interferometer.h
 */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Interferometer;
#ifndef OSKAR_INTERFEROMETER_TYPEDEF_
#define OSKAR_INTERFEROMETER_TYPEDEF_
typedef struct oskar_Interferometer oskar_Interferometer;
#endif

#ifdef __cplusplus
}
#endif

#include <oskar_global.h>
#include <interferometer/oskar_interferometer_accessors.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EXPORT
void oskar_interferometer_check_init(oskar_Interferometer* h, int* status);

/* mode = "legacy" or "precision"; frame = "apparent", "icrs", "fk5_j2000".
 * Precision supports CPU/CUDA and uses an explicit offline EOP table.
 * Positive time/frequency widths require at least two quadrature samples;
 * increase the counts until convergence for the intended sky and baselines.
 * Changing this configuration after initialisation is not supported.
 */
OSKAR_EXPORT
void oskar_interferometer_set_astrometry(oskar_Interferometer* h,
        const char* mode, const char* frame, const char* eop_file,
        int time_samples, int frequency_samples, int* status);

OSKAR_EXPORT
oskar_Interferometer* oskar_interferometer_create(int precision, int* status);

OSKAR_EXPORT
oskar_VisBlock* oskar_interferometer_finalise_block(oskar_Interferometer* h,
        int block_index, int* status);

OSKAR_EXPORT
void oskar_interferometer_finalise(oskar_Interferometer* h, int* status);

OSKAR_EXPORT
void oskar_interferometer_free(oskar_Interferometer* h, int* status);

OSKAR_EXPORT
void oskar_interferometer_free_device_data(oskar_Interferometer* h, int* status);

OSKAR_EXPORT
void oskar_interferometer_reset_cache(oskar_Interferometer* h, int* status);

OSKAR_EXPORT
void oskar_interferometer_run_block(oskar_Interferometer* h, int block_index,
        int gpu_id, int* status);

OSKAR_EXPORT
void oskar_interferometer_run(oskar_Interferometer* h, int* status);

OSKAR_EXPORT
void oskar_interferometer_write_block(oskar_Interferometer* h,
        const oskar_VisBlock* block, int block_index, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
