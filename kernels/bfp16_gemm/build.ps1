# Build one whole-array bfp16 GEMM design into build\<design>[_<variant>]\<shape>\{final.xclbin,insts.bin}.
#   .\build.ps1 -Design whole_array_bfp -M 512 -K 2048 -N 4096 -Tm 64 -Tk 128 -Tn 64 [-Cols 8] [-Rne]
# Designs are copies of mlir-aie v1.4.2's block_datatypes examples (see designs\).
# Peano only: IRON compiles the kernels with llvm-aie clang++ -O2.
# -Rne builds the kernel with -DROUND_CONV_EVEN (round-to-nearest-even for the
# core's accum->bf16/bfp16 conversions; the stock kernel leaves the rounding
# register at floor).
# -Paired builds mm_bfp.cc's pair-interleaved kernel (bfp design only; the host
# must lay A and B out with bench_bfp16.exe layout=paired).
# -Tag/-KFlags add a named variant with extra kernel defines, e.g.
#   -Paired -Tag u2 -KFlags '-DPAIRED_UNROLL=2'.
# -OutMode 1|2 builds whole_array_bfp_rtp's 16-bit output modes (into <shape>_m<mode>).
# -DesignArgs passes extra design CLI flags, e.g. for whole_array_bfp_acc:
#   -Tag f32d1 -DesignArgs '--acc 32 --c-depth 1'.
param(
    [Parameter(Mandatory)][string]$Design,
    [Parameter(Mandatory)][int]$M, [Parameter(Mandatory)][int]$K, [Parameter(Mandatory)][int]$N,
    [int]$Tm = 64, [int]$Tk = 64, [int]$Tn = 64, [int]$Cols = 8, [switch]$Rne, [switch]$Paired,
    [string]$Tag = '', [string]$KFlags = '', [string]$DesignArgs = '', [int]$OutMode = 0
)
$ErrorActionPreference = 'Stop'
. C:\dev\mlir-aie\iron_env.ps1
$here = $PSScriptRoot
$shape = "${M}x${K}x${N}_${Tm}x${Tk}x${Tn}_c${Cols}"
# -OutMode 1|2 (whole_array_bfp_rtp): the build's output mode, in the directory name as _m<mode>
if ($OutMode) { $shape += "_m$OutMode"; $DesignArgs += " --out-mode $OutMode" }
$variant = $Design; $flags = @()
if ($Paired) { $variant += '_paired'; $flags += '-DPAIRED' }
if ($Rne) { $variant += '_rne'; $flags += '-DROUND_CONV_EVEN' }
if ($Tag) { $variant += "_$Tag"; $flags += $KFlags.Split(' ', [StringSplitOptions]::RemoveEmptyEntries) }
$env:BFP_KFLAGS = $flags -join ' '
$out = Join-Path $here "build\$variant\$shape"
New-Item -ItemType Directory -Force -Path $out | Out-Null
# aiecc keeps kernel objects in <xclbin>.prj and skips recompiling them; start clean.
Remove-Item -Recurse -Force (Join-Path $out 'final.prj') -ErrorAction SilentlyContinue
$sw = [Diagnostics.Stopwatch]::StartNew()
python (Join-Path $here "designs\$Design.py") -d npu2 --n-aie-cols $Cols `
    -M $M -K $K -N $N -m $Tm -k $Tk -n $Tn `
    --xclbin-path (Join-Path $out 'final.xclbin') --insts-path (Join-Path $out 'insts.bin') `
    @($DesignArgs.Split(' ', [StringSplitOptions]::RemoveEmptyEntries))
if ($LASTEXITCODE -ne 0) { Write-Host "BUILD_FAIL $variant $shape"; exit 1 }
$sw.Stop()
Write-Host ("BUILD_OK {0} {1} {2:N1}s -> {3}" -f $variant, $shape, $sw.Elapsed.TotalSeconds, $out)
