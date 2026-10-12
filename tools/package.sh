#!/usr/bin/env bash
# The Linux twin of package.ps1: tools/package.sh v0.2.0, after build.sh; the release workflow runs it.
set -euo pipefail

version=${1:?usage: tools/package.sh <version>}
tag=b10944
archive_sha256=de48f1c0890365d432dea3db1d8e9482686748fe5821f211719779e5724bdc7b
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
build=${BUILD:-$root/build}
out=${OUT:-$root/dist}

name="ggml-xdna-$version-llama-$tag-ubuntu-x64"
stage="$out/$name"
rm -rf "$stage" "$stage.tar.gz" "$stage.tar.gz.sha256"
mkdir -p "$stage"

# llama.cpp's release, as published (fresh, never third_party/, where the build also copies its tests)
dl="$out/llama-$tag-bin-ubuntu-vulkan-x64.tar.gz"
[[ -f "$dl" ]] || curl --fail --location --retry 3 --silent --show-error \
    "https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-bin-ubuntu-vulkan-x64.tar.gz" -o "$dl"
echo "$archive_sha256  $dl" | sha256sum --check --status || { echo "llama.cpp archive checksum mismatch" >&2; exit 1; }
tar -xzf "$dl" -C "$stage" --strip-components=1

# ours
for f in libggml-xdna.so libggml-xdna-xrt.so npu; do
    [[ -f "$build/$f" ]] || { echo "no $f in $build; build first" >&2; exit 1; }
    cp "$build/$f" "$stage/"
done
cp "$root/kernels/bfp16_gemm/prebuilt/bfp16_gemm.xclbin" "$stage/"
cp "$root/tools/bench.ps1" "$stage/"
cp "$root/README.md" "$root/LICENSE" "$root/NOTICE" "$stage/"

# each part's license
mkdir -p "$stage/LICENSES"
curl --fail --location --retry 3 --silent --show-error \
    "https://raw.githubusercontent.com/ggml-org/llama.cpp/$tag/LICENSE" -o "$stage/LICENSES/llama.cpp-LICENSE.txt"
cp "$root/vendor/xrt/LICENSE" "$stage/LICENSES/xrt-LICENSE.txt"
cp "$root/vendor/xrt/NOTICE" "$stage/LICENSES/xrt-NOTICE.txt"
cp "$root/kernels/bfp16_gemm/LICENSE" "$stage/LICENSES/mlir-aie-LICENSE.txt"

# every file's SHA-256, for sha256sum -c
(cd "$stage" && find . -type f ! -name SHA256SUMS.txt | sed 's|^\./||' | LC_ALL=C sort | xargs -d '\n' sha256sum > SHA256SUMS.txt)

tar -czf "$stage.tar.gz" -C "$out" "$name"
(cd "$out" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
echo "$stage.tar.gz  $(du -m "$stage.tar.gz" | cut -f1) MB  sha256 $(cut -d' ' -f1 "$stage.tar.gz.sha256")"
