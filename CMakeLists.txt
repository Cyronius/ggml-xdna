cmake_minimum_required(VERSION 3.20)
project(ggml-xdna CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Prebuilt llama.cpp release we build against and run with. The headers come
# from the matching source tag; on Windows the import libs are generated from
# the release DLLs by tools/fetch-llama.ps1.
set(GGML_DIST "${CMAKE_SOURCE_DIR}/third_party/llama-b10944" CACHE PATH "llama.cpp release binaries")
set(GGML_HEADERS "${CMAKE_SOURCE_DIR}/third_party/ggml-b10944" CACHE PATH "matching ggml headers")
set(GGML_IMPLIB_DIR "${CMAKE_SOURCE_DIR}/third_party" CACHE PATH "generated ggml import libs")

# Without the NPU path the backend runs its claimed matmuls on a CPU
# reference only, like GGML_XDNA_HOST_ONLY=1.
option(GGML_XDNA_NPU "build the NPU path" ON)

if(WIN32)
    set(GGML_LIBS "${GGML_IMPLIB_DIR}/ggml-base.lib" "${GGML_IMPLIB_DIR}/ggml.lib")
    set(EXE ".exe")
    set(XDNA_LIB "ggml-xdna.dll")
    set(AVX2 /arch:AVX2)
else()
    find_library(GGML_BASE_LIB NAMES ggml-base PATHS "${GGML_DIST}" NO_DEFAULT_PATH REQUIRED)
    find_library(GGML_LIB NAMES ggml PATHS "${GGML_DIST}" NO_DEFAULT_PATH REQUIRED)
    set(GGML_LIBS "${GGML_BASE_LIB}" "${GGML_LIB}")
    set(EXE "")
    set(XDNA_LIB "libggml-xdna.so")
    set(AVX2 -mavx2 -mfma)
    # everything we build runs next to llama.cpp's libraries and finds them there, as llama.cpp's own programs do
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
    find_package(Threads REQUIRED)
endif()

add_library(ggml-xdna SHARED
    src/ggml-xdna.cpp
    src/xdna-ref.cpp
    src/xdna-platform.cpp
    include/ggml-xdna.h
)

target_include_directories(ggml-xdna PRIVATE
    include
    src
    "${GGML_HEADERS}/include"
    "${GGML_HEADERS}/src"
)

target_compile_definitions(ggml-xdna PRIVATE
    GGML_SHARED            # import ggml core from ggml-base
    GGML_BACKEND_SHARED
    GGML_BACKEND_BUILD     # export ggml_backend_init / ggml_backend_score
    GGML_BACKEND_DL
    _CRT_SECURE_NO_WARNINGS
)

# ggml-base for the core, ggml for the device registry, to find the Vulkan device we work next to
target_link_libraries(ggml-xdna PRIVATE ${GGML_LIBS})

if(WIN32)
    set_target_properties(ggml-xdna PROPERTIES PREFIX "")
else()
    target_link_libraries(ggml-xdna PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
    # only ggml's entry points leave the library, so our stand-ins for the XRT wrapper's functions stay inside
    set_target_properties(ggml-xdna PROPERTIES CXX_VISIBILITY_PRESET hidden)
endif()

# Drop the library next to the llama.cpp binaries so GGML_BACKEND_PATH is a
# one-liner and ggml-base resolves from the same directory. The NPU
# kernel goes next to the library in both places: the backend looks for it there.
set(XDNA_XCLBIN "${CMAKE_SOURCE_DIR}/kernels/bfp16_gemm/prebuilt/bfp16_gemm.xclbin")
add_custom_command(TARGET ggml-xdna POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:ggml-xdna> "${GGML_DIST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${XDNA_XCLBIN}" "${GGML_DIST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${XDNA_XCLBIN}" $<TARGET_FILE_DIR:ggml-xdna>
)

# The NPU path. XRT is vendored: headers under vendor/xrt (Apache-2.0, see
# vendor/xrt/LICENSE) and our extern "C" shim over XRT's C++ API under
# vendor/xrt-shim. The build needs no NPU driver on either system, and the
# backend loads without one and says so.
if(GGML_XDNA_NPU)
    target_sources(ggml-xdna PRIVATE
        src/xdna-npu.cpp
        src/xdna-exec.cpp
        src/npu/npu_bfp16.cpp
        src/npu/bfp16_pack.cpp
        src/npu/bfp16_insts.cpp
    )
    target_include_directories(ggml-xdna PRIVATE src/npu vendor/xrt-shim)
    target_compile_definitions(ggml-xdna PRIVATE XDNA_HAVE_NPU)
    if(WIN32)
        # XRT ships no import lib on Windows, so one is built here from the
        # list of functions the backend imports (vendor/xrt-implib). The
        # driver's DLL is delay-loaded.
        set(XRT_DEF "${CMAKE_SOURCE_DIR}/vendor/xrt-implib/xrt_coreutil.def")
        set(XRT_LIB "${CMAKE_BINARY_DIR}/xrt_coreutil.lib")
        add_custom_command(OUTPUT "${XRT_LIB}"
            COMMAND "${CMAKE_AR}" /nologo "/def:${XRT_DEF}" /machine:x64 "/out:${XRT_LIB}"
            DEPENDS "${XRT_DEF}"
            VERBATIM
        )
        add_custom_target(xrt-implib DEPENDS "${XRT_LIB}")
        add_dependencies(ggml-xdna xrt-implib)
        target_sources(ggml-xdna PRIVATE vendor/xrt-shim/xrt_shim.cpp)
        target_include_directories(ggml-xdna PRIVATE vendor/xrt/include)
        # /Zc:__cplusplus: XRT's headers pick std::any over boost::any only when
        # __cplusplus is reported honestly. /arch:AVX2 /fp:fast: the host's small
        # ops vectorize.
        target_compile_options(ggml-xdna PRIVATE /Zc:__cplusplus /arch:AVX2 /fp:fast)
        target_link_libraries(ggml-xdna PRIVATE "${XRT_LIB}" delayimp)
        target_link_options(ggml-xdna PRIVATE /DELAYLOAD:xrt_coreutil.dll)
    else()
        target_sources(ggml-xdna PRIVATE src/npu/xrt_load.cpp)
        # fast math that still has NaN and infinity: the check on the NPU's results looks for them
        target_compile_options(ggml-xdna PRIVATE ${AVX2} -ffast-math -fno-finite-math-only)
        # The shim as a library of its own, unlinked to XRT: the backend loads XRT, then this (src/npu/xrt_load.cpp).
        add_library(ggml-xdna-xrt SHARED vendor/xrt-shim/xrt_shim.cpp)
        target_include_directories(ggml-xdna-xrt PRIVATE vendor/xrt-shim vendor/xrt/include)
        find_library(UUID_LIB uuid REQUIRED)  # XRT's headers use libuuid on Linux (uuid-dev)
        target_link_libraries(ggml-xdna-xrt PRIVATE "${UUID_LIB}")
        add_dependencies(ggml-xdna ggml-xdna-xrt)
        add_custom_command(TARGET ggml-xdna-xrt POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:ggml-xdna-xrt> "${GGML_DIST}"
        )
    endif()

    # A standalone repro of the NPU's buffers going bad past its memory ceiling: the NPU layer and the kernel only. Not shipped.
    add_executable(npu-ceiling-repro
        tools/npu-ceiling-repro.cpp
        src/xdna-platform.cpp
        src/npu/npu_bfp16.cpp
        src/npu/bfp16_pack.cpp
        src/npu/bfp16_insts.cpp
    )
    target_include_directories(npu-ceiling-repro PRIVATE src src/npu vendor/xrt-shim)
    if(WIN32)
        target_sources(npu-ceiling-repro PRIVATE vendor/xrt-shim/xrt_shim.cpp)
        target_include_directories(npu-ceiling-repro PRIVATE vendor/xrt/include)
        target_compile_options(npu-ceiling-repro PRIVATE /Zc:__cplusplus ${AVX2})
        target_link_libraries(npu-ceiling-repro PRIVATE "${XRT_LIB}" pdh)
        add_dependencies(npu-ceiling-repro xrt-implib)
    else()
        target_sources(npu-ceiling-repro PRIVATE src/npu/xrt_load.cpp)
        target_compile_options(npu-ceiling-repro PRIVATE ${AVX2})
        target_link_libraries(npu-ceiling-repro PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
        add_dependencies(npu-ceiling-repro ggml-xdna-xrt)
    endif()
    # next to the kernel, and on Linux to the XRT wrapper
    add_custom_command(TARGET npu-ceiling-repro POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:npu-ceiling-repro> "${GGML_DIST}"
    )
endif()

# The launcher: `npu llama-server ...` with nothing to set (XDNA-LAUNCHER).
# It goes next to the llama.cpp programs it runs.
add_executable(npu tools/npu.cpp)
if(WIN32)
    target_compile_definitions(npu PRIVATE UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
    target_compile_options(npu PRIVATE /utf-8)
    target_link_libraries(npu PRIVATE shell32)
endif()
add_custom_command(TARGET npu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:npu> "${GGML_DIST}"
)

#
# tests
#

enable_testing()

foreach(t test-mul-mat test-dispatch-gate test-memory-budget test-safe-start test-host-ops)
    add_executable(${t} tests/${t}.cpp)
    target_include_directories(${t} PRIVATE "${GGML_HEADERS}/include")
    target_compile_definitions(${t} PRIVATE GGML_SHARED GGML_BACKEND_SHARED)
    target_link_libraries(${t} PRIVATE ${GGML_LIBS})
    # they run next to the llama.cpp libraries
    add_custom_command(TARGET ${t} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:${t}> "${GGML_DIST}"
    )
endforeach()

# The NPU instruction streams made on the host, against the toolchain's (no
# NPU or XRT needed).
add_executable(test-insts-gen tests/test-insts-gen.cpp src/npu/bfp16_insts.cpp)
target_include_directories(test-insts-gen PRIVATE src/npu)
target_compile_definitions(test-insts-gen PRIVATE XDNA_INSTS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/insts")

# The weight packer against the kernel's verified layout (pure host code).
add_executable(test-bfp16-pack tests/test-bfp16-pack.cpp src/npu/bfp16_pack.cpp)
target_include_directories(test-bfp16-pack PRIVATE src/npu)
target_compile_options(test-bfp16-pack PRIVATE ${AVX2})

# ctest labels: "host" needs no NPU (the backend-facing ones do need a Vulkan
# device), "npu" needs the NPU, "nodriver" expects a machine without the NPU
# driver (CI).
file(TO_NATIVE_PATH "${GGML_DIST}/${XDNA_LIB}" XDNA_PATH)
set(XDNA_ENV "GGML_BACKEND_PATH=${XDNA_PATH}" "GGML_XDNA_KERNELS=")
add_test(NAME insts-gen COMMAND test-insts-gen)
add_test(NAME bfp16-pack COMMAND test-bfp16-pack)
add_test(NAME safe-start COMMAND "${GGML_DIST}/test-safe-start${EXE}")
add_test(NAME safe-start-no-driver COMMAND "${GGML_DIST}/test-safe-start${EXE}" --no-driver)
foreach(t dispatch-gate mul-mat)
    add_test(NAME ${t}-host COMMAND "${GGML_DIST}/test-${t}${EXE}")
    add_test(NAME ${t}-npu COMMAND "${GGML_DIST}/test-${t}${EXE}")
    set_tests_properties(${t}-host PROPERTIES LABELS host ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=1")
    set_tests_properties(${t}-npu PROPERTIES LABELS npu ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=")
endforeach()
# every NPU result comes back as NaN: the backend has to catch it and still
# give the right answer
add_test(NAME mul-mat-npu-nan COMMAND "${GGML_DIST}/test-mul-mat${EXE}")
set_tests_properties(mul-mat-npu-nan PROPERTIES LABELS npu ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=;GGML_XDNA_NAN_AFTER=0")
# every NPU submission fails: the same, through the other way the NPU fails
add_test(NAME mul-mat-npu-fail COMMAND "${GGML_DIST}/test-mul-mat${EXE}")
set_tests_properties(mul-mat-npu-fail PROPERTIES LABELS npu ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=;GGML_XDNA_FAIL_AFTER=0")
add_test(NAME memory-budget-host COMMAND "${GGML_DIST}/test-memory-budget${EXE}")
add_test(NAME memory-budget-npu COMMAND "${GGML_DIST}/test-memory-budget${EXE}")
set(XDNA_BUDGET_ENV "${XDNA_ENV};GGML_XDNA_MAX_COPY_GB=0.05")
set_tests_properties(memory-budget-host PROPERTIES LABELS host ENVIRONMENT "${XDNA_BUDGET_ENV};GGML_XDNA_HOST_ONLY=1")
set_tests_properties(memory-budget-npu PROPERTIES LABELS npu ENVIRONMENT "${XDNA_BUDGET_ENV};GGML_XDNA_HOST_ONLY=")
# the host's small steps between NPU matmuls, against ggml's CPU backend
add_test(NAME host-ops-host COMMAND "${GGML_DIST}/test-host-ops${EXE}")
set_tests_properties(host-ops-host PROPERTIES LABELS host ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=1")
set_tests_properties(insts-gen bfp16-pack PROPERTIES LABELS host)
set_tests_properties(safe-start PROPERTIES LABELS host ENVIRONMENT "${XDNA_ENV}")
set_tests_properties(safe-start-no-driver PROPERTIES LABELS nodriver ENVIRONMENT "${XDNA_ENV}")
