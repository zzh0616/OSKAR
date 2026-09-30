#
# oskar/convert/test/CMakeLists.txt
#

set(name convert_test)
set(${name}_SRC
    main.cpp
    Test_convert_pqr_to_ecef_matrix.cpp
    Test_coordinate_conversions.cpp
    Test_uvw_matrix.cpp
    #Test_sofa.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name}
    #sofa_c
    oskar gtest
    #${CASACORE_LIBRARIES}
)
if (OpenMP_CXX_FOUND)
    target_link_libraries(${name} OpenMP::OpenMP_CXX)
endif()
add_test(NAME ${name} COMMAND ${name})

add_executable(astrometry_test main.cpp Test_astrometry.cpp)
target_link_libraries(astrometry_test oskar gtest)
if (OSKAR_USE_ERFA)
    target_link_libraries(astrometry_test ${ERFA_LIBRARY})
endif()
add_test(NAME astrometry_test COMMAND astrometry_test)

# Optional, fully offline acceptance. Python is a test dependency only.
set(OSKAR_ASTROMETRY_TEST_PYTHON "" CACHE FILEPATH
    "Python with numpy, astropy, pyerfa and h5py for precision beam acceptance")
if (OSKAR_ASTROMETRY_TEST_PYTHON)
    if (NOT OSKAR_USE_ERFA OR NOT HDF5_FOUND)
        message(FATAL_ERROR "Precision beam acceptance requires ERFA and HDF5")
    endif()
    add_test(NAME astrometry_beam_reference
        COMMAND "${OSKAR_ASTROMETRY_TEST_PYTHON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/uvw_beam_reference.py"
        --build "${CMAKE_BINARY_DIR}"
        --output "${CMAKE_CURRENT_BINARY_DIR}/beam_acceptance")
    set_tests_properties(astrometry_beam_reference PROPERTIES
        TIMEOUT 300
        FIXTURES_SETUP astrometry_beam_inputs
        LABELS "astrometry;offline;integration"
        ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
endif()

set(OSKAR_ASTROMETRY_TEST_CUDA_IDS "" CACHE STRING
    "Opt-in offline CPU/CUDA acceptance: CUDA IDs, e.g. 0 or 0,1")
if (OSKAR_ASTROMETRY_TEST_CUDA_IDS)
    if (NOT CUDAToolkit_FOUND OR NOT OSKAR_ASTROMETRY_TEST_PYTHON)
        message(FATAL_ERROR "CUDA astrometry acceptance requires CUDA and the Python reference suite")
    endif()
    string(REGEX REPLACE ",.*" "" astrometry_first_gpu "${OSKAR_ASTROMETRY_TEST_CUDA_IDS}")
    add_test(NAME astrometry_cuda_reference
        COMMAND "${OSKAR_ASTROMETRY_TEST_PYTHON}"
        "${CMAKE_CURRENT_SOURCE_DIR}/uvw_gpu_reference.py"
        --build "${CMAKE_BINARY_DIR}"
        --fixtures "${CMAKE_CURRENT_BINARY_DIR}/beam_acceptance/base"
                   "${CMAKE_CURRENT_BINARY_DIR}/beam_acceptance"
        --output "${CMAKE_CURRENT_BINARY_DIR}/cuda_acceptance"
        --gpu "${astrometry_first_gpu}"
        --multi-gpu "${OSKAR_ASTROMETRY_TEST_CUDA_IDS}")
    set_tests_properties(astrometry_cuda_reference PROPERTIES
        TIMEOUT 300 RUN_SERIAL TRUE FIXTURES_REQUIRED astrometry_beam_inputs
        LABELS "astrometry;offline;integration;cuda"
        ENVIRONMENT "OMP_NUM_THREADS=2;OPENBLAS_NUM_THREADS=1")
endif()
