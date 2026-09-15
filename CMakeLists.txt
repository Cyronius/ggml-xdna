cmake_minimum_required(VERSION 3.20)
project(ggml-xdna CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prebuilt llama.cpp release we build against and run with. The headers come
# from the matching source tag; the import libs are generated from the release
# DLLs by tools/fetch-llama.ps1.
set(GGML_DIST "${CMAKE_SOURCE_DIR}/third_party/llama-b10944" CACHE PATH "llama.cpp release binaries")
set(GGML_HEADERS "${CMAKE_SOURCE_DIR}/third_party/ggml-b10944" CACHE PATH "matching ggml headers")
set(GGML_IMPLIB_DIR "${CMAKE_SOURCE_DIR}/third_party" CACHE PATH "generated ggml import libs")

add_library(ggml-xdna SHARED
    src/ggml-xdna.cpp
    src/xdna-ref.cpp
    include/ggml-xdna.h
)

target_include_directories(ggml-xdna PRIVATE
    include
    "${GGML_HEADERS}/include"
    "${GGML_HEADERS}/src"
)

target_compile_definitions(ggml-xdna PRIVATE
    GGML_SHARED            # import ggml core from ggml-base.dll
    GGML_BACKEND_SHARED
    GGML_BACKEND_BUILD     # export ggml_backend_init / ggml_backend_score
    GGML_BACKEND_DL
    _CRT_SECURE_NO_WARNINGS
)

target_link_libraries(ggml-xdna PRIVATE
    "${GGML_IMPLIB_DIR}/ggml-base.lib"
)

set_target_properties(ggml-xdna PROPERTIES PREFIX "")

# Drop the DLL next to the llama.cpp binaries so GGML_BACKEND_PATH is a
# one-liner and ggml-base.dll resolves from the same directory.
add_custom_command(TARGET ggml-xdna POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:ggml-xdna> "${GGML_DIST}"
)

# XRT is vendored: headers under vendor/xrt (Apache-2.0, see vendor/xrt/LICENSE)
# and our extern "C" shim over XRT's C++ API under vendor/xrt-shim. XRT ships no
# import lib on Windows, so tools/gen-xrt-implib.ps1 reconstructs one from the
# export table of the driver's own xrt_coreutil.dll.
set(XRT_INC "${CMAKE_SOURCE_DIR}/vendor/xrt/include")
set(XRT_SHIM "${CMAKE_SOURCE_DIR}/vendor/xrt-shim")
set(XRT_LIB "${CMAKE_SOURCE_DIR}/third_party/xrt_coreutil.lib" CACHE FILEPATH "XRT import lib")

if(EXISTS "${XRT_LIB}")
    add_executable(bench-dispatch
        tools/bench-dispatch.cpp
        "${XRT_SHIM}/xrt_shim.cpp"
    )
    target_include_directories(bench-dispatch PRIVATE
        "${XRT_SHIM}"
        "${XRT_INC}"
    )
    # XRT's headers pick std::any over boost::any only when __cplusplus is
    # reported honestly, which MSVC does not do without this.
    target_compile_options(bench-dispatch PRIVATE /Zc:__cplusplus)
    target_link_libraries(bench-dispatch PRIVATE "${XRT_LIB}")
else()
    message(STATUS "${XRT_LIB} missing - run tools/gen-xrt-implib.ps1; skipping bench-dispatch")
endif()

foreach(t test-mul-mat test-dispatch-gate)
    add_executable(${t} tests/${t}.cpp)
    target_include_directories(${t} PRIVATE "${GGML_HEADERS}/include")
    target_compile_definitions(${t} PRIVATE GGML_SHARED GGML_BACKEND_SHARED)
    target_link_libraries(${t} PRIVATE
        "${GGML_IMPLIB_DIR}/ggml-base.lib"
        "${GGML_IMPLIB_DIR}/ggml.lib"
    )
    add_custom_command(TARGET ${t} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:${t}> "${GGML_DIST}"
    )
endforeach()
