#!/usr/bin/env bash
# The Linux twin of fetch-llama.ps1, adapted from Nulifyer's port: the pinned release (checksum checked) and its headers.
set -euo pipefail

tag=b10944
archive_sha256=de48f1c0890365d432dea3db1d8e9482686748fe5821f211719779e5724bdc7b
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tp="$root/third_party"
dist="$tp/llama-$tag"
hdr="$tp/ggml-$tag"
mkdir -p "$tp"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

if [[ ! -f "$dist/llama-server" ]]; then
    url="https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-bin-ubuntu-vulkan-x64.tar.gz"
    echo "downloading $url"
    curl --fail --location --retry 3 --silent --show-error "$url" -o "$tmp/llama.tar.gz"
    echo "$archive_sha256  $tmp/llama.tar.gz" | sha256sum --check --status ||
        { echo "the llama.cpp $tag archive's checksum doesn't match" >&2; exit 1; }
    mkdir -p "$tmp/x" "$dist"
    tar -xzf "$tmp/llama.tar.gz" -C "$tmp/x"
    server=$(find "$tmp/x" -type f -name llama-server -print -quit)
    [[ -n "$server" ]] || { echo "no llama-server in the llama.cpp $tag archive" >&2; exit 1; }
    cp -a "$(dirname "$server")/." "$dist/"
fi

if [[ ! -f "$hdr/src/ggml-backend-impl.h" ]]; then
    url="https://github.com/ggml-org/llama.cpp/archive/refs/tags/$tag.tar.gz"
    echo "downloading $url"
    curl --fail --location --retry 3 --silent --show-error "$url" -o "$tmp/src.tar.gz"
    tar -xzf "$tmp/src.tar.gz" -C "$tmp"
    src="$tmp/llama.cpp-$tag/ggml"
    mkdir -p "$hdr/include" "$hdr/src"
    cp "$src"/include/*.h "$hdr/include/"
    for f in ggml-backend-impl.h ggml-impl.h ggml-common.h; do cp "$src/src/$f" "$hdr/src/"; done
fi

echo "third_party ready at $tp"
