# Provenance — vendor/

Third-party and shared source carried in-tree so this repo builds with no other
checkout present. Nothing here is generated at build time; nothing here is
edited to fix bugs (bump the upstream instead, and re-record it below).

## vendor/xrt/include/xrt/

XRT's public C++ API headers (`xrt::device`, `xclbin`, `hw_context`, `bo`,
`kernel`, `run`, `runlist`, plus the `experimental/` and `detail/` headers they
pull in). Apache-2.0 — see `xrt/LICENSE` and `xrt/NOTICE`.

Version **2.20**, as declared by `xrt/detail/version-slim.h`.

Pruned to what the shim actually compiles against: the set was taken from
`cl /showIncludes` on `vendor/xrt-shim/xrt_shim.cpp`, which reaches 28 files,
all of them under `xrt/`. The whole `xrt/` subtree is kept (52 files) rather
than exactly those 28, so the boundary is one coherent upstream directory
instead of a hand-picked list. XRT's top-level headers (`ert.h`, `xclbin.h`,
`xclhal2.h`, the top-level `experimental/`, `deprecated/`, `xdp/`, …) are
reachable from nothing here and are not carried.

**One file is not upstream:** `xrt/detail/version-slim.h`. Upstream generates it
at build time and does not track it in git, so it is a minimal stand-in
declaring the version macros that `xrt/detail/abi.h` consumes. Those macros
affect the compile-time ABI tag, so keep them matching the XRT the driver
actually installs. Re-create it after any XRT bump.

## vendor/xrt-implib/

XRT ships no `.lib` on Windows. `xrt_coreutil.def` lists, by decorated name,
the functions of the driver's `xrt_coreutil.dll` that the shim calls, taken
from the export table of the driver named in its header. CMake builds the
import lib from it, so the build needs no driver. The backend delay-loads the
DLL and binds every listed function when it starts, by name, so it runs
against whatever driver is installed and turns itself off, with a reason, if
one is missing. `tools/check-xrt-driver.ps1` checks an installed driver
against the list.

## vendor/xrt-shim/

`xrt_shim.h` / `xrt_shim.cpp` — a 96-line `extern "C"` surface over XRT's C++
API (device, hw context, xclbin, kernel, bo, run, runlist), so callers that are
not C++17-with-XRT's-headers can drive the NPU. Author: Cyrus Attoun. Shared
with the kernel-side runtime work; it is the same file in both places, so
changes should land in both rather than diverging here.
