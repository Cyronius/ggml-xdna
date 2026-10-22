#!/usr/bin/env bash
# The on-hardware Linux check (issue #19), run from the repository on Linux booted on a Ryzen AI machine.
set -uo pipefail

usage() {
    cat <<'EOF'
tools/linux-npu-check.sh [--install] [--quick] [--models DIR]

  --install   install what's missing with apt (Ubuntu), asking for your password
  --quick     skip the slow parts: the second model, the failure test and the memory ceiling
  --models    where to keep the downloaded models (default ~/.cache/zest-check)

Everything it prints also goes to npu-check-<date>.log in the repository: send that file back.
EOF
}

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
install=0 quick=0 models=${XDG_CACHE_HOME:-$HOME/.cache}/zest-check
while (($#)); do
    case $1 in
        --install) install=1 ;;
        --quick) quick=1 ;;
        --models) models=$2; shift ;;
        -h | --help) usage; exit 0 ;;
        *) usage; exit 2 ;;
    esac
    shift
done

log="$root/npu-check-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$log") 2>&1
dist="$root/third_party/llama-b10944"
addon="$dist/libggml-zest.so"
summary=()
pass() { summary+=("PASS  $*"); echo ">>> PASS: $*"; }
fail() { summary+=("FAIL  $*"); echo ">>> FAIL: $*"; }
note() { summary+=("NOTE  $*"); echo ">>> NOTE: $*"; }
section() { printf '\n===== %s (%s)\n' "$*" "$(date +%T)"; }
# the add-on's own lines from a llama.cpp run's output
xdna_lines() { grep -E '(xdna|zest): ' "$1" | sed -E 's/^[0-9.]+ [A-Z] //' | sort | uniq -c | sed 's/^ *//' | head -20; }

finish() {
    section "summary"
    printf '%s\n' "${summary[@]}"
    echo
    echo "The whole log: $log"
    echo "Please send that file back."
    exit 0
}

#
# the machine
#

section "the machine"
date
uname -a
grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
grep -m1 'model name' /proc/cpuinfo
free -g | head -2
echo "locked memory: soft $(ulimit -S -l), hard $(ulimit -H -l) (KB)"
echo "amdxdna module: $(modinfo -F filename amdxdna 2>/dev/null || echo none) version $(modinfo -F version amdxdna 2>/dev/null)"
lsmod | grep -E '^amdxdna' || echo "amdxdna not loaded"
ls -l /dev/accel 2>&1
for f in /sys/class/accel/accel0/device/fw_version /sys/class/accel/accel0/device/vbnv; do
    [[ -r $f ]] && echo "$f: $(cat "$f")"
done
(journalctl -k -b --no-pager 2>/dev/null || dmesg 2>/dev/null) | grep -i amdxdna | tail -15
command -v xrt-smi >/dev/null && xrt-smi examine 2>&1 | head -40
command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>/dev/null | grep -E 'deviceName|driverName|apiVersion' | head -8

#
# what it needs
#

section "prerequisites"
missing=()
for tool in cmake g++ curl python3; do command -v "$tool" >/dev/null || missing+=("$tool"); done
[[ -e /usr/include/uuid/uuid.h ]] || missing+=(uuid-dev)
have_xrt=0
if ldconfig -p 2>/dev/null | grep -q 'libxrt_coreutil\.so\.2' || [[ -e /opt/xilinx/xrt/lib/libxrt_coreutil.so.2 ]]; then
    have_xrt=1
fi
((have_xrt)) || missing+=(xrt)
if ((${#missing[@]})) && ((install)); then
    echo "installing for: ${missing[*]}"
    if ((have_xrt == 0)) && ! apt-cache show libxrt-npu2 >/dev/null 2>&1; then
        sudo add-apt-repository -y ppa:lemonade-team/stable
    fi
    sudo apt-get update
    sudo apt-get install -y build-essential cmake ninja-build curl uuid-dev python3 libvulkan1 mesa-vulkan-drivers \
        libxrt-npu2
    missing=()
    for tool in cmake g++ curl python3; do command -v "$tool" >/dev/null || missing+=("$tool"); done
    if ldconfig -p 2>/dev/null | grep -q 'libxrt_coreutil\.so\.2' || [[ -e /opt/xilinx/xrt/lib/libxrt_coreutil.so.2 ]]; then
        have_xrt=1
    fi
    ((have_xrt)) || missing+=(xrt)
fi
if ((${#missing[@]})); then
    fail "missing: ${missing[*]}. Rerun with --install, or: sudo apt install build-essential cmake curl uuid-dev python3 libxrt-npu2 (24.04 and 25.10 need ppa:lemonade-team/stable for libxrt-npu2)"
    finish
fi
pass "prerequisites"
if [[ ! -e /dev/accel/accel0 ]]; then
    fail "no /dev/accel/accel0: the kernel's NPU driver (amdxdna) isn't running. A newer kernel, or amdxdna-dkms, is needed"
    finish
fi

#
# build and tests
#

section "build"
if (cd "$root" && ./build.sh notest); then pass "build"; else fail "build"; finish; fi

section "the add-on at this session's locked-memory limit ($(ulimit -S -l) KB)"
GGML_BACKEND_PATH="$addon" "$dist/llama-server" --list-devices > "$root/build/devices.txt" 2>&1
cat "$root/build/devices.txt"
if grep -q '^  XDNA0:' "$root/build/devices.txt"; then
    pass "XDNA0 offered at the session's own limit"
else
    note "XDNA0 not offered at the session's own limit: $(grep -m1 'not offering' "$root/build/devices.txt" | sed -E 's/^[0-9.]+ [A-Z] //')"
fi

section "the add-on with locked memory capped at 8 MB"
at8=$(ulimit -S -l 8192 && GGML_BACKEND_PATH="$addon" "$dist/llama-server" --list-devices 2>&1 | grep -E 'xdna|XDNA')
echo "$at8"
note "at 8 MB of locked memory: $(grep -m1 -E 'xdna: |XDNA0:' <<< "$at8" | sed -E 's/^[0-9.]+ [A-Z] //')"

section "lifting the locked-memory cap"
if [[ $(ulimit -H -l) == unlimited ]]; then
    ulimit -S -l unlimited
else
    echo "raising this script's own cap, which needs your password once"
    sudo prlimit --pid $$ --memlock=unlimited:unlimited
fi
echo "locked memory now: soft $(ulimit -S -l), hard $(ulimit -H -l)"

section "tests"
if ctest --test-dir "$root/build" -L 'host|npu' --output-on-failure; then pass "ctest host and npu"; else fail "ctest host and npu"; fi

#
# models
#

mkdir -p "$models"
fetch() {  # repository, file
    local dst="$models/$2"
    if [[ ! -s $dst ]]; then
        echo "downloading $2"
        curl --fail --location --retry 3 --silent --show-error "https://huggingface.co/$1/resolve/main/$2" -o "$dst.part" &&
            mv "$dst.part" "$dst"
    fi
    [[ -s $dst ]]
}
small="$models/Qwen3-1.7B-Q4_0.gguf"
big="$models/Qwen3-4B-Q4_K_M.gguf"
section "models"
fetch unsloth/Qwen3-1.7B-GGUF Qwen3-1.7B-Q4_0.gguf || { fail "couldn't download Qwen3-1.7B"; finish; }
((quick)) || fetch unsloth/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf || note "couldn't download Qwen3-4B; skipping it"
ls -l "$models"

work="$root/build/check"
mkdir -p "$work"
head -c 500 "$root/tests/prompts/prose.txt" > "$work/short.txt"
head -c 12000 "$root/README.md" > "$work/long.txt"
cat "$root/README.md" "$root/specs/xdna-backend/spec.md" > "$work/text.txt"

section "the same answer as the GPU (XDNA-BLOCK-AGREES)"
agree=(-m "$small" -f "$work/short.txt" -n 24 --temp 0 -no-cnv -fa on -b 2048 -ub 2048)
GGML_BACKEND_PATH="$addon" "$dist/llama-completion" "${agree[@]}" -dev Vulkan0 > "$work/gpu.txt" 2> "$work/gpu.err"
GGML_BACKEND_PATH="$addon" GGML_XDNA_MIN_BATCH=32 "$dist/llama-completion" "${agree[@]}" -dev XDNA0,Vulkan0 -ts 0,1 \
    > "$work/npu.txt" 2> "$work/npu.err"
echo "GPU:  $(tail -c 300 "$work/gpu.txt")"
echo "NPU:  $(tail -c 300 "$work/npu.txt")"
xdna_lines "$work/npu.err"
if [[ -s $work/gpu.txt ]] && cmp -s "$work/gpu.txt" "$work/npu.txt"; then
    pass "24 greedy tokens identical to the GPU's"
else
    fail "the NPU's 24 tokens differ from the GPU's (or a run failed; see above)"
fi

section "npu on a 3,000-token prompt, with the trace on"
start=$(date +%s)
GGML_XDNA_TRACE=1 "$dist/zest" llama-completion -m "$small" -f "$work/long.txt" -n 32 --temp 0 -no-cnv -c 8192 \
    > "$work/launch.txt" 2> "$work/launch.err"
code=$?
echo "exit code $code, $(($(date +%s) - start)) s"
xdna_lines "$work/launch.err"
grep -E '^xdna (times|host passes|weight copies)' "$work/launch.err" | tail -3
if ((code == 0)) && grep -q '^npu: prompts on the NPU' "$work/launch.err" && grep -q '^xdna times' "$work/launch.err" &&
    ! grep -q 'the NPU failed' "$work/launch.err"; then
    pass "npu llama-completion ran on the NPU"
else
    fail "npu llama-completion (exit $code; see above)"
fi

section "how far the answers drift from the GPU's (KL divergence)"
kl=(-m "$small" -f "$work/text.txt" -c 512 -b 2048 -ub 2048 --chunks 8 --kl-divergence-base "$work/base.kld")
GGML_BACKEND_PATH="$addon" "$dist/llama-perplexity" "${kl[@]}" -dev Vulkan0 > "$work/kl-gpu.txt" 2>&1
GGML_BACKEND_PATH="$addon" "$dist/llama-perplexity" "${kl[@]}" -dev XDNA0,Vulkan0 -ts 0,1 --kl-divergence \
    > "$work/kl-npu.txt" 2>&1
grep -E 'Mean +KLD|Same top p' "$work/kl-npu.txt"
kld=$(grep -oP 'Mean\s+KLD:\s+\K[-0-9.]+' "$work/kl-npu.txt" | head -1)
rm -f "$work/base.kld"
if [[ -n $kld ]] && python3 -c "import sys; sys.exit(0 if 0 < float('$kld') < 0.01 else 1)"; then
    pass "mean KL divergence $kld (under 0.01)"
else
    fail "mean KL divergence '${kld:-none}' (want above 0 and under 0.01)"
fi

section "speed against the GPU alone"
speed() {  # model, rounds
    local m=$1 rounds=$2 r
    for ((r = 1; r <= rounds; r++)); do
        "$dist/zest" llama-bench -m "$m" -dev Vulkan0,XDNA0/Vulkan0 -p 2048 -n 0 -ub 2048 -b 2048 -r 3 -o json \
            > "$work/bench-$r.json" 2> "$work/bench-$r.err"
    done
    python3 - "$work" "$rounds" "$(basename "$m")" <<'EOF'
import json, re, statistics, sys
work, rounds, name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ratios, gpus = [], []
for r in range(1, rounds + 1):
    try:
        text = open(f"{work}/bench-{r}.json").read()
        tests = json.loads(re.search(r"\[.*\]", text, re.S).group(0))
        gpu = next(t["avg_ts"] for t in tests if t["devices"] == "Vulkan0")
        npu = next(t["avg_ts"] for t in tests if t["devices"] == "XDNA0/Vulkan0")
    except Exception as e:
        print(f"  round {r}: no result ({e})")
        continue
    ratios.append(npu / gpu)
    gpus.append(gpu)
    print(f"  round {r}: GPU alone {gpu:8.1f}  with the NPU {npu:8.1f} tokens/s  {npu / gpu:.2f}x")
if ratios:
    print(f"RESULT {name}: GPU alone {statistics.median(gpus):.0f} tokens/s, with the NPU "
          f"{statistics.median(ratios):.2f}x (rounds {min(ratios):.2f}-{max(ratios):.2f}, {len(ratios)} of {rounds})")
else:
    print(f"RESULT {name}: no result")
EOF
}
speed "$small" 5 | tee "$work/speed-small.txt"
note "$(grep '^RESULT' "$work/speed-small.txt")"
if ((!quick)) && [[ -s $big ]]; then
    speed "$big" 3 | tee "$work/speed-big.txt"
    note "$(grep '^RESULT' "$work/speed-big.txt")"
fi

#
# the slow parts
#

if ((!quick)); then
    section "an NPU failure mid-prompt (GGML_XDNA_FAIL_AFTER=30)"
    start=$(date +%s)
    GGML_BACKEND_PATH="$addon" GGML_XDNA_FAIL_AFTER=30 "$dist/llama-completion" -m "$small" -f "$work/long.txt" -n 24 \
        --temp 0 -no-cnv -c 8192 -b 2048 -ub 2048 -dev XDNA0,Vulkan0 -ts 0,1 > "$work/fail.txt" 2> "$work/fail.err"
    code=$?
    took=$(($(date +%s) - start))
    GGML_BACKEND_PATH="$addon" "$dist/llama-completion" -m "$small" -f "$work/long.txt" -n 24 --temp 0 -no-cnv -c 8192 \
        -b 2048 -ub 2048 -dev Vulkan0 > "$work/fail-gpu.txt" 2> /dev/null
    xdna_lines "$work/fail.err"
    if ((code == 0)) && [[ $(grep -c 'the NPU failed' "$work/fail.err") == 1 ]]; then
        pass "a failing NPU: the run finished, with one 'the NPU failed' line, in $took s"
    else
        fail "a failing NPU: exit $code, $(grep -c 'the NPU failed' "$work/fail.err") 'the NPU failed' lines, $took s"
    fi
    cmp -s "$work/fail.txt" "$work/fail-gpu.txt" && note "after the failure the reply matched the GPU's" ||
        note "after the failure the reply differed from the GPU's"

    section "the NPU's memory ceiling (26 GiB on Windows)"
    if [[ -x $dist/npu-ceiling-repro ]]; then
        "$dist/npu-ceiling-repro" --start-gib 16 --step-gib 2 --max-gib 40 --min-free-gib 8
        case $? in
            0) note "memory ceiling: nothing damaged up to the largest step that fit (see above)" ;;
            1) note "memory ceiling: buffers came back damaged with no error (see above for where)" ;;
            3) note "memory ceiling: the driver refused or failed first, nothing damaged (see above)" ;;
            *) fail "memory ceiling: the test couldn't start" ;;
        esac
    else
        fail "memory ceiling: no npu-ceiling-repro in $dist"
    fi
fi

finish
