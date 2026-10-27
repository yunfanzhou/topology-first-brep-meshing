# cmake -DSRC=<wildmeshing-toolkit build>/app -DDST=<dir> -P install_remeshing_app.cmake
# Copies remeshing_app and the DLLs built next to it (GMP on Windows) to DST. Multi-config generators
# put them in a per-configuration subdirectory of SRC, hence the recursive glob.
file(GLOB_RECURSE files "${SRC}/remeshing_app" "${SRC}/remeshing_app.exe" "${SRC}/*.dll")
if(NOT files)
    message(FATAL_ERROR "remeshing_app not found in ${SRC}")
endif()
file(COPY ${files} DESTINATION "${DST}")
message(STATUS "remeshing_app installed to ${DST}")
