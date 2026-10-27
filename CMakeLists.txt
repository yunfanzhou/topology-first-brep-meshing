# Builds remeshing_app, the remeshing stage's C++ tool (wildmeshing-toolkit, branch fpir), into <build>/bin:
#
#   cmake -S . -B build
#   cmake --build build --config Release
#
# Needs git, a C++17 compiler and, on Linux and macOS, GMP (apt install libgmp-dev / brew install gmp);
# on Windows wildmeshing-toolkit downloads GMP itself. The forward stage's mmg2d_O3 comes from the mmgpy
# package in requirements.txt and needs no build.
cmake_minimum_required(VERSION 3.20)
project(topology_first_brep_meshing_tools NONE)

include(ExternalProject)

set(WMTK_COMMIT ed57e12061dca446b852449d49a57defbe4fe259 CACHE STRING "wildmeshing-toolkit commit (branch fpir)")

cmake_host_system_information(RESULT jobs QUERY NUMBER_OF_LOGICAL_CORES)
if(DEFINED ENV{CMAKE_BUILD_PARALLEL_LEVEL})
    set(jobs $ENV{CMAKE_BUILD_PARALLEL_LEVEL})
endif()

# With Homebrew, point wildmeshing-toolkit at GMP's own prefix: found through the shared
# /opt/homebrew/include, GMP would also expose other Homebrew headers to the build, such as a newer
# HDF5 that shadows the one wildmeshing-toolkit bundles.
set(gmp_args)
if(APPLE)
    execute_process(COMMAND brew --prefix gmp OUTPUT_VARIABLE gmp_prefix RESULT_VARIABLE brew_result
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(brew_result EQUAL 0 AND EXISTS "${gmp_prefix}/include/gmp.h")
        set(gmp_args -DGMP_INCLUDES=${gmp_prefix}/include -DGMP_LIBRARIES=${gmp_prefix}/lib/libgmp.dylib)
    endif()
endif()

# Eigen 3.4.0 for wildmeshing-toolkit's dependencies (libigl, lagrange): their recipes git-clone it from
# gitlab.com, which refuses clones under load, while the archive download keeps working.
ExternalProject_Add(eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    PREFIX ${CMAKE_BINARY_DIR}/eigen
    SOURCE_DIR ${CMAKE_BINARY_DIR}/eigen-src
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Add(wmtk
    DEPENDS eigen
    # the commit's tarball: a clone would also fetch ~650 MB of history
    URL https://github.com/wildmeshing/wildmeshing-toolkit/archive/${WMTK_COMMIT}.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    # short paths: wildmeshing-toolkit's dependencies nest deep enough to hit MAX_PATH with MSVC
    PREFIX ${CMAKE_BINARY_DIR}/wmtk
    SOURCE_DIR ${CMAKE_BINARY_DIR}/ws
    BINARY_DIR ${CMAKE_BINARY_DIR}/wb
    CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        # some of its dependencies declare a CMake version older than CMake 4 accepts
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DWMTK_APP_UNIFORM_REMESH=ON
        -DWMTK_APP_SEC=ON
        -DWMTK_APP_HARMTET=OFF
        -DWMTK_APP_QSLIM=OFF
        -DWMTK_APP_INTERIOR_TET_OPT=OFF
        -DWMTK_APP_UNIT_TESTS=OFF
        # HDF5 (for paraviewo) would otherwise use any installed libaec, whose include directory can hold
        # the headers of a different HDF5 (e.g. Homebrew's /opt/homebrew/include)
        -DHDF5_ENABLE_SZIP_SUPPORT=OFF
        -DFETCHCONTENT_SOURCE_DIR_EIGEN=${CMAKE_BINARY_DIR}/eigen-src
        -DCPM_eigen_SOURCE=${CMAKE_BINARY_DIR}/eigen-src
        -DCMAKE_PROJECT_WildMeshingToolkit_INCLUDE=${CMAKE_CURRENT_SOURCE_DIR}/cmake/wmtk_project_include.cmake
        ${gmp_args}
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --target remeshing_app --parallel ${jobs}
    INSTALL_COMMAND ${CMAKE_COMMAND} -DSRC=<BINARY_DIR>/app -DDST=${CMAKE_BINARY_DIR}/bin
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/install_remeshing_app.cmake
)
