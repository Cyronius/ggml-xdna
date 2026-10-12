#!/usr/bin/env bash
# Builds ggml-xdna on Linux and runs its tests, the twin of build.cmd: ./build.sh, or ./build.sh notest.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

need=()
for tool in cmake g++ curl; do command -v "$tool" >/dev/null || need+=("$tool"); done
[[ -e /usr/include/uuid/uuid.h ]] || need+=("uuid-dev")
if ((${#need[@]})); then
    echo "missing: ${need[*]} (on Ubuntu: sudo apt install build-essential cmake curl uuid-dev)" >&2
    exit 1
fi

tools/fetch-llama.sh

generator=()
command -v ninja >/dev/null && generator=(-G Ninja)
[[ -f build/CMakeCache.txt ]] || cmake -S . -B build "${generator[@]}" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"

[[ "${1:-}" == notest ]] && exit 0
# with the NPU and its driver: every test but the one expecting no driver; without them, no NPU tests
xrt=no
if ldconfig -p 2>/dev/null | grep -q 'libxrt_coreutil\.so\.2' || [[ -e /opt/xilinx/xrt/lib/libxrt_coreutil.so.2 ]]; then
    xrt=yes
fi
skip=npu
[[ -e /dev/accel/accel0 && $xrt == yes ]] && skip=nodriver
ctest --test-dir build -LE "$skip" --output-on-failure
