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

# Without the NPU path the backend runs its claimed matmuls on a CPU
# reference only, like GGML_XDNA_HOST_ONLY=1.
option(GGML_XDNA_NPU "build the NPU path" ON)

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
    "${GGML_IMPLIB_DIR}/ggml.lib"      # the device registry, to find the Vulkan device we work next to
)

set_target_properties(ggml-xdna PROPERTIES PREFIX "")

# Drop the DLL next to the llama.cpp binaries so GGML_BACKEND_PATH is a
# one-liner and ggml-base.dll resolves from the same directory. The NPU
# kernel goes next to the DLL in both places: the backend looks for it there.
set(XDNA_XCLBIN "${CMAKE_SOURCE_DIR}/kernels/bfp16_gemm/prebuilt/bfp16_gemm.xclbin")
add_custom_command(TARGET ggml-xdna POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:ggml-xdna> "${GGML_DIST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${XDNA_XCLBIN}" "${GGML_DIST}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${XDNA_XCLBIN}" $<TARGET_FILE_DIR:ggml-xdna>
)

# The NPU path. XRT is vendored: headers under vendor/xrt (Apache-2.0, see
# vendor/xrt/LICENSE) and our extern "C" shim over XRT's C++ API under
# vendor/xrt-shim. XRT ships no import lib on Windows, so one is built here
# from the list of functions the backend imports (vendor/xrt-implib). The
# driver's DLL is delay-loaded: the backend then loads on a machine without
# the driver, and says so instead of failing to load.
if(GGML_XDNA_NPU)
    set(XRT_DEF "${CMAKE_SOURCE_DIR}/vendor/xrt-implib/xrt_coreutil.def")
    set(XRT_LIB "${CMAKE_BINARY_DIR}/xrt_coreutil.lib")
    add_custom_command(OUTPUT "${XRT_LIB}"
        COMMAND "${CMAKE_AR}" /nologo "/def:${XRT_DEF}" /machine:x64 "/out:${XRT_LIB}"
        DEPENDS "${XRT_DEF}"
        VERBATIM
    )
    add_custom_target(xrt-implib DEPENDS "${XRT_LIB}")
    add_dependencies(ggml-xdna xrt-implib)

    target_sources(ggml-xdna PRIVATE
        src/xdna-npu.cpp
        src/xdna-exec.cpp
        src/npu/npu_bfp16.cpp
        src/npu/bfp16_pack.cpp
        src/npu/bfp16_insts.cpp
        vendor/xrt-shim/xrt_shim.cpp
    )
    target_include_directories(ggml-xdna PRIVATE src/npu vendor/xrt-shim vendor/xrt/include)
    target_compile_definitions(ggml-xdna PRIVATE XDNA_HAVE_NPU)
    # /Zc:__cplusplus: XRT's headers pick std::any over boost::any only when
    # __cplusplus is reported honestly. /arch:AVX2 /fp:fast: the host's small
    # ops vectorize.
    target_compile_options(ggml-xdna PRIVATE /Zc:__cplusplus /arch:AVX2 /fp:fast)
    target_link_libraries(ggml-xdna PRIVATE "${XRT_LIB}" delayimp)
    target_link_options(ggml-xdna PRIVATE /DELAYLOAD:xrt_coreutil.dll)

    # A standalone repro, for AMD, of the NPU's buffers going bad past its
    # memory ceiling: the NPU layer and the kernel only, no ggml. Not shipped.
    add_executable(npu-ceiling-repro
        tools/npu-ceiling-repro.cpp
        src/npu/npu_bfp16.cpp
        src/npu/bfp16_pack.cpp
        src/npu/bfp16_insts.cpp
        vendor/xrt-shim/xrt_shim.cpp
    )
    target_include_directories(npu-ceiling-repro PRIVATE src/npu vendor/xrt-shim vendor/xrt/include)
    target_compile_options(npu-ceiling-repro PRIVATE /Zc:__cplusplus /arch:AVX2)
    target_link_libraries(npu-ceiling-repro PRIVATE "${XRT_LIB}" pdh)
    add_dependencies(npu-ceiling-repro xrt-implib)
    add_custom_command(TARGET npu-ceiling-repro POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${XDNA_XCLBIN}" $<TARGET_FILE_DIR:npu-ceiling-repro>
    )
endif()

# The launcher: `npu llama-server ...` with nothing to set (XDNA-LAUNCHER).
# It goes next to the llama.cpp programs it runs.
add_executable(npu tools/npu.cpp)
target_compile_definitions(npu PRIVATE UNICODE _UNICODE _CRT_SECURE_NO_WARNINGS)
target_compile_options(npu PRIVATE /utf-8)
target_link_libraries(npu PRIVATE shell32)
add_custom_command(TARGET npu POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:npu> "${GGML_DIST}"
)

#
# tests
#

enable_testing()

foreach(t test-mul-mat test-dispatch-gate test-memory-budget test-safe-start)
    add_executable(${t} tests/${t}.cpp)
    target_include_directories(${t} PRIVATE "${GGML_HEADERS}/include")
    target_compile_definitions(${t} PRIVATE GGML_SHARED GGML_BACKEND_SHARED)
    target_link_libraries(${t} PRIVATE
        "${GGML_IMPLIB_DIR}/ggml-base.lib"
        "${GGML_IMPLIB_DIR}/ggml.lib"
    )
    # they run next to the llama.cpp DLLs
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
target_compile_options(test-bfp16-pack PRIVATE /arch:AVX2)

# ctest labels: "host" needs no NPU (the backend-facing ones do need a Vulkan
# device), "npu" needs the NPU, "nodriver" expects a machine without the NPU
# driver (CI).
file(TO_NATIVE_PATH "${GGML_DIST}/ggml-xdna.dll" XDNA_DLL)
set(XDNA_ENV "GGML_BACKEND_PATH=${XDNA_DLL}" "GGML_XDNA_KERNELS=")
add_test(NAME insts-gen COMMAND test-insts-gen)
add_test(NAME bfp16-pack COMMAND test-bfp16-pack)
add_test(NAME safe-start COMMAND "${GGML_DIST}/test-safe-start.exe")
add_test(NAME safe-start-no-driver COMMAND "${GGML_DIST}/test-safe-start.exe" --no-driver)
foreach(t dispatch-gate mul-mat)
    add_test(NAME ${t}-host COMMAND "${GGML_DIST}/test-${t}.exe")
    add_test(NAME ${t}-npu COMMAND "${GGML_DIST}/test-${t}.exe")
    set_tests_properties(${t}-host PROPERTIES LABELS host ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=1")
    set_tests_properties(${t}-npu PROPERTIES LABELS npu ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=")
endforeach()
# every NPU result comes back as NaN: the backend has to catch it and still
# give the right answer
add_test(NAME mul-mat-npu-nan COMMAND "${GGML_DIST}/test-mul-mat.exe")
set_tests_properties(mul-mat-npu-nan PROPERTIES LABELS npu ENVIRONMENT "${XDNA_ENV};GGML_XDNA_HOST_ONLY=;GGML_XDNA_NAN_AFTER=0")
add_test(NAME memory-budget-host COMMAND "${GGML_DIST}/test-memory-budget.exe")
add_test(NAME memory-budget-npu COMMAND "${GGML_DIST}/test-memory-budget.exe")
set(XDNA_BUDGET_ENV "${XDNA_ENV};GGML_XDNA_MAX_COPY_GB=0.05")
set_tests_properties(memory-budget-host PROPERTIES LABELS host ENVIRONMENT "${XDNA_BUDGET_ENV};GGML_XDNA_HOST_ONLY=1")
set_tests_properties(memory-budget-npu PROPERTIES LABELS npu ENVIRONMENT "${XDNA_BUDGET_ENV};GGML_XDNA_HOST_ONLY=")
set_tests_properties(insts-gen bfp16-pack PROPERTIES LABELS host)
set_tests_properties(safe-start PROPERTIES LABELS host ENVIRONMENT "${XDNA_ENV}")
set_tests_properties(safe-start-no-driver PROPERTIES LABELS nodriver ENVIRONMENT "${XDNA_ENV}")
