# Included at the end of wildmeshing-toolkit's project() call (CMAKE_PROJECT_WildMeshingToolkit_INCLUDE).

# Defining wmtk::data here makes its wmtk_data recipe return early instead of cloning the ~200 MB
# test-data repository, which only its unit tests read (through WMTK_DATA_DIR).
add_library(wmtk_data INTERFACE)
add_library(wmtk::data ALIAS wmtk_data)

# Some dependencies declare a CMake version older than CMake 4 accepts. The cache variable passed to this
# build doesn't reach the CMake runs they start themselves (fast-envelope's geogram download on Windows),
# but the environment variable does.
set(ENV{CMAKE_POLICY_VERSION_MINIMUM} 3.5)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # lagrange's bundled mapbox/earcut.h uses uint32_t without including <cstdint>, which GCC 13 and
    # later no longer include transitively.
    string(APPEND CMAKE_CXX_FLAGS " -include cstdint")
    # wildmeshing-toolkit compiles itself with a list of -Werror=... flags, and GCC 15 reports a false
    # positive in Eigen's AVX code under one of them (-Werror=array-bounds). Defining its warnings target
    # here, empty, drops that list. Only with GCC: the list also holds code-generation options, so the
    # other compilers keep it and build exactly what was tested.
    add_library(wmtk_warnings INTERFACE)
    add_library(wmtk::warnings ALIAS wmtk_warnings)
endif()

# Visual Studio 2026 removed <hash_map>, which geogram (built by fast-envelope on Windows) includes.
if(MSVC)
    include(CheckIncludeFileCXX)
    check_include_file_cxx(hash_map WMTK_HAVE_HASH_MAP)
    if(NOT WMTK_HAVE_HASH_MAP)
        include_directories(BEFORE SYSTEM "${CMAKE_CURRENT_LIST_DIR}/msvc_compat")
    endif()
endif()
