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
    "${GGML_IMPLIB_DIR}/ggml.lib"      # the device registry, to find the Vulkan device we work next to
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

# The backend's NPU path (plan: .claude/plans/backend-size-aware.md, steps 2
# and 3): the prototype's bfp16 dispatch and packing, compiled into the DLL.
# Without the XRT import lib the backend still builds, running its claimed
# matmuls on the host reference.
if(EXISTS "${XRT_LIB}")
    target_sources(ggml-xdna PRIVATE
        src/xdna-npu.cpp
        src/xdna-exec.cpp
        hybrid/npu_bfp16.cpp
        hybrid/bfp16_pack.cpp
        "${XRT_SHIM}/xrt_shim.cpp"
    )
    target_include_directories(ggml-xdna PRIVATE hybrid "${XRT_SHIM}" "${XRT_INC}")
    target_compile_definitions(ggml-xdna PRIVATE XDNA_HAVE_NPU)
    target_compile_options(ggml-xdna PRIVATE /Zc:__cplusplus /arch:AVX2 /fp:fast)  # vectorized host ops, as hybrid/
    target_link_libraries(ggml-xdna PRIVATE "${XRT_LIB}")
endif()

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

    # Real GGUF weights through our packer onto the NPU, against a kernel
    # built in openflowlm-next. Needs the driver, so it sits behind the same
    # guard as bench-dispatch.
    add_executable(test-npu-gemm
        hybrid/test-npu-gemm.cpp
        hybrid/npu_gemm.cpp
        hybrid/q4_pack.cpp
        "${XRT_SHIM}/xrt_shim.cpp"
    )
    target_include_directories(test-npu-gemm PRIVATE
        hybrid "${GGML_HEADERS}/include" "${XRT_SHIM}" "${XRT_INC}"
    )
    target_compile_definitions(test-npu-gemm PRIVATE GGML_SHARED GGML_BACKEND_SHARED _CRT_SECURE_NO_WARNINGS)
    target_compile_options(test-npu-gemm PRIVATE /Zc:__cplusplus)
    target_link_libraries(test-npu-gemm PRIVATE "${XRT_LIB}" "${GGML_IMPLIB_DIR}/ggml-base.lib")
    add_custom_command(TARGET test-npu-gemm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:test-npu-gemm> "${GGML_DIST}"
    )

    # Stage 3b of .claude/plans/npu-prefill-beats-gpu.md: one layer's real
    # weights, packed by our code, through the bfp16 kernel on the NPU.
    add_executable(test-npu-bfp16
        hybrid/test-npu-bfp16.cpp
        hybrid/npu_bfp16.cpp
        hybrid/bfp16_pack.cpp
        "${XRT_SHIM}/xrt_shim.cpp"
    )
    target_include_directories(test-npu-bfp16 PRIVATE
        hybrid "${GGML_HEADERS}/include" "${XRT_SHIM}" "${XRT_INC}"
    )
    target_compile_definitions(test-npu-bfp16 PRIVATE GGML_SHARED GGML_BACKEND_SHARED _CRT_SECURE_NO_WARNINGS)
    target_compile_options(test-npu-bfp16 PRIVATE /Zc:__cplusplus /arch:AVX2)
    target_link_libraries(test-npu-bfp16 PRIVATE "${XRT_LIB}" "${GGML_IMPLIB_DIR}/ggml-base.lib")
    add_custom_command(TARGET test-npu-bfp16 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:test-npu-bfp16> "${GGML_DIST}"
    )
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

# Stage 0 of the hybrid prefill plan (.claude/plans/hybrid-prefill-amd-design.md):
# the prefill-to-decode handoff through llama.cpp's sequence-state API. This
# drives llama.cpp as a library rather than living inside it as a backend, so
# it links llama.lib (tools/fetch-llama.ps1 makes it) and needs llama.h beside
# the ggml headers.
add_executable(kv-handoff
    hybrid/kv-handoff.cpp
    hybrid/kv_state.cpp
)
target_include_directories(kv-handoff PRIVATE hybrid "${GGML_HEADERS}/include")
target_compile_definitions(kv-handoff PRIVATE GGML_SHARED GGML_BACKEND_SHARED LLAMA_SHARED _CRT_SECURE_NO_WARNINGS)
target_link_libraries(kv-handoff PRIVATE
    "${GGML_IMPLIB_DIR}/ggml-base.lib"
    "${GGML_IMPLIB_DIR}/ggml.lib"
    "${GGML_IMPLIB_DIR}/llama.lib"
)
add_custom_command(TARGET kv-handoff POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:kv-handoff> "${GGML_DIST}"
)

# The GPU prefill baseline. Same library linkage as kv-handoff; it exists
# because a prefill timed once in a fresh process is dominated by Vulkan
# shader compilation and cannot be quoted.
add_executable(gpu-prefill
    hybrid/gpu-prefill.cpp
)
target_include_directories(gpu-prefill PRIVATE hybrid "${GGML_HEADERS}/include")
target_compile_definitions(gpu-prefill PRIVATE GGML_SHARED GGML_BACKEND_SHARED LLAMA_SHARED _CRT_SECURE_NO_WARNINGS)
target_link_libraries(gpu-prefill PRIVATE
    "${GGML_IMPLIB_DIR}/ggml-base.lib"
    "${GGML_IMPLIB_DIR}/ggml.lib"
    "${GGML_IMPLIB_DIR}/llama.lib"
)
add_custom_command(TARGET gpu-prefill POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:gpu-prefill> "${GGML_DIST}"
)

# How fast data moves between Vulkan's buffers and CPU memory, by each path
# ggml offers (step 0 of .claude/plans/backend-size-aware.md). A measurement.
add_executable(datapath-probe
    hybrid/datapath-probe.cpp
)
target_include_directories(datapath-probe PRIVATE hybrid "${GGML_HEADERS}/include")
target_compile_definitions(datapath-probe PRIVATE GGML_SHARED GGML_BACKEND_SHARED LLAMA_SHARED _CRT_SECURE_NO_WARNINGS)
target_link_libraries(datapath-probe PRIVATE
    "${GGML_IMPLIB_DIR}/ggml-base.lib"
    "${GGML_IMPLIB_DIR}/ggml.lib"
    "${GGML_IMPLIB_DIR}/llama.lib"
)
add_custom_command(TARGET datapath-probe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:datapath-probe> "${GGML_DIST}"
)

# The host reference prefill for Qwen3 and the test that hands its keys and
# values to a GPU context. The reference is plain loops over float32, so it
# gets the vector instructions and the relaxed float rules; nothing in it
# depends on strict ordering.
add_executable(ref-handoff
    hybrid/ref-handoff.cpp
    hybrid/qwen3_ref.cpp
    hybrid/kv_state.cpp
)
target_include_directories(ref-handoff PRIVATE hybrid "${GGML_HEADERS}/include")
target_compile_definitions(ref-handoff PRIVATE GGML_SHARED GGML_BACKEND_SHARED LLAMA_SHARED _CRT_SECURE_NO_WARNINGS)
target_compile_options(ref-handoff PRIVATE /arch:AVX2 /fp:fast)
target_link_libraries(ref-handoff PRIVATE
    "${GGML_IMPLIB_DIR}/ggml-base.lib"
    "${GGML_IMPLIB_DIR}/ggml.lib"
    "${GGML_IMPLIB_DIR}/llama.lib"
)
add_custom_command(TARGET ref-handoff POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:ref-handoff> "${GGML_DIST}"
)

# Stage 3a of .claude/plans/npu-prefill-beats-gpu.md: the bfp16 packer against
# the host layout the NPU kernel was verified with. Pure host code.
add_executable(test-bfp16-pack
    hybrid/test-bfp16-pack.cpp
    hybrid/bfp16_pack.cpp
)
target_include_directories(test-bfp16-pack PRIVATE hybrid)
target_compile_options(test-bfp16-pack PRIVATE /arch:AVX2)
add_custom_command(TARGET test-bfp16-pack POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:test-bfp16-pack> "${GGML_DIST}"
)

# Stage 1 of .claude/plans/npu-prefill-beats-gpu.md: the reference prefill with
# its matmul inputs rounded the way an NPU datapath would round them, each
# result handed to a GPU context, compared against float32.
add_executable(prefill-numerics
    hybrid/prefill-numerics.cpp
    hybrid/qwen3_ref.cpp
    hybrid/kv_state.cpp
)
target_include_directories(prefill-numerics PRIVATE hybrid "${GGML_HEADERS}/include")
target_compile_definitions(prefill-numerics PRIVATE GGML_SHARED GGML_BACKEND_SHARED LLAMA_SHARED _CRT_SECURE_NO_WARNINGS)
target_compile_options(prefill-numerics PRIVATE /arch:AVX2 /fp:fast)
target_link_libraries(prefill-numerics PRIVATE
    "${GGML_IMPLIB_DIR}/ggml-base.lib"
    "${GGML_IMPLIB_DIR}/ggml.lib"
    "${GGML_IMPLIB_DIR}/llama.lib"
)
add_custom_command(TARGET prefill-numerics POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:prefill-numerics> "${GGML_DIST}"
)
# With the XRT import library present, --npu runs the projections on the NPU
# (stage 3c), and the same comparison then measures the real kernel.
if(EXISTS "${XRT_LIB}")
    target_sources(prefill-numerics PRIVATE
        hybrid/npu_projections.cpp
        hybrid/npu_prefill.cpp
        hybrid/gpu_attention.cpp
        hybrid/npu_bfp16.cpp
        hybrid/bfp16_pack.cpp
        "${XRT_SHIM}/xrt_shim.cpp"
    )
    target_include_directories(prefill-numerics PRIVATE "${XRT_SHIM}" "${XRT_INC}")
    target_compile_definitions(prefill-numerics PRIVATE HAVE_NPU)
    target_compile_options(prefill-numerics PRIVATE /Zc:__cplusplus)
    target_link_libraries(prefill-numerics PRIVATE "${XRT_LIB}")
endif()

# The GGUF-to-NPU-chunk weight packer and its check. Reads GGUF through ggml,
# so it needs no llama.cpp.
add_executable(test-q4-pack
    hybrid/test-q4-pack.cpp
    hybrid/q4_pack.cpp
)
target_include_directories(test-q4-pack PRIVATE hybrid "${GGML_HEADERS}/include")
target_compile_definitions(test-q4-pack PRIVATE GGML_SHARED GGML_BACKEND_SHARED _CRT_SECURE_NO_WARNINGS)
target_link_libraries(test-q4-pack PRIVATE "${GGML_IMPLIB_DIR}/ggml-base.lib")
add_custom_command(TARGET test-q4-pack POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:test-q4-pack> "${GGML_DIST}"
)

# Stage 3e of .claude/plans/npu-prefill-beats-gpu.md: the whole prefill with
# every projection on the NPU, timed against the GPU in one process.
if(EXISTS "${XRT_LIB}")
    add_executable(npu-prefill-bench
        hybrid/npu-prefill-bench.cpp
        hybrid/npu_prefill.cpp
        hybrid/gpu_attention.cpp
        hybrid/npu_bfp16.cpp
        hybrid/bfp16_pack.cpp
        hybrid/qwen3_ref.cpp
        hybrid/kv_state.cpp
        "${XRT_SHIM}/xrt_shim.cpp"
    )
    target_include_directories(npu-prefill-bench PRIVATE hybrid "${GGML_HEADERS}/include" "${XRT_SHIM}" "${XRT_INC}")
    target_compile_definitions(npu-prefill-bench PRIVATE GGML_SHARED GGML_BACKEND_SHARED LLAMA_SHARED _CRT_SECURE_NO_WARNINGS)
    target_compile_options(npu-prefill-bench PRIVATE /Zc:__cplusplus /arch:AVX2 /fp:fast /Zi)
    target_link_options(npu-prefill-bench PRIVATE /DEBUG /OPT:REF /OPT:ICF)  # a PDB, for crash stacks
    target_link_libraries(npu-prefill-bench PRIVATE
        "${XRT_LIB}"
        "${GGML_IMPLIB_DIR}/ggml-base.lib"
        "${GGML_IMPLIB_DIR}/ggml.lib"
        "${GGML_IMPLIB_DIR}/llama.lib"
    )
    add_custom_command(TARGET npu-prefill-bench POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:npu-prefill-bench> "${GGML_DIST}"
    )
endif()
