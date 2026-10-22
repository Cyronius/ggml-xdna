# Moving from ggml-xdna to Zest

The project is now [Cyronius/zest](https://github.com/Cyronius/zest), with its
website at [cyronius.github.io/zest](https://cyronius.github.io/zest/).
[Download Zest v0.2.0-rc2](https://github.com/Cyronius/zest/releases/tag/v0.2.0-rc2)
and extract it into a new folder, keeping its files together.

| Component | Earlier name | Zest name |
|---|---|---|
| Windows package | `ggml-xdna-<version>-llama-b10944-win-x64.zip` | `zest-<version>-llama-b10944-win-x64.zip` |
| Linux package | `ggml-xdna-<version>-llama-b10944-ubuntu-x64.tar.gz` | `zest-<version>-llama-b10944-ubuntu-x64.tar.gz` |
| Command-line launcher | `npu.exe` / `npu` | `zest.exe` / `zest` |
| Windows backend | `ggml-xdna.dll` | `ggml-zest.dll` |
| Linux backend | `libggml-xdna.so` | `libggml-zest.so` |
| Linux XRT shim | `libggml-xdna-xrt.so` | `libggml-zest-xrt.so` |

For Lemonade, use `Start-Lemonade.cmd` from the new package. It selects
`ggml-zest.dll` and still accepts an existing launcher-managed configuration.
For direct llama.cpp use:

```text
zest llama-server -m model.gguf
zest --memory-gb 20 llama-cli -m model.gguf
```

The package also includes `npu` as a compatibility copy of the Zest launcher.
Both commands use the new backend. Zest saves model choices under
`%LOCALAPPDATA%\zest` on Windows or `$XDG_STATE_HOME/zest` (normally
`~/.local/state/zest`) on Linux. It reads the old `ggml-xdna/last-model.txt`
preference when no Zest preference has been saved yet.

If you set `GGML_BACKEND_PATH` yourself, update its filename to the new DLL or
shared library. The device remains **`XDNA0`**, and the **`GGML_XDNA_*`** settings
and C entry point retain their hardware names. Zest still requires the pinned
llama.cpp **b10944** runtime; the rename doesn't change that ABI requirement.

Historical releases and benchmark records retain their original filenames and
hashes. The published model table was measured before the binary rename; its
`ggml-xdna.dll` receipts describe the actual tested artifact.

## Verification

A fresh Windows build in a separate runtime directory passed all **12 applicable
CTest tests**, including actual NPU matmuls, NaN rejection, submission failure,
memory limits, and host operations. The no-driver test was excluded because this
machine has the driver. [CTest output](results/zest-rename-ctest.txt).
Two additional launcher tests verify that both `zest` and its `npu` alias discover
the backend themselves and start llama-server successfully.
[Launcher test output](results/zest-rename-cli-ctest.txt).

The release verification receipt records the Windows package hash, file-manifest
check, actual Lemonade answer checks, and positive NPU-call trace. It is attached
to the [Zest preview release](https://github.com/Cyronius/zest/releases/tag/v0.2.0-rc2).
Linux builds and packaging are checked in CI; Linux NPU hardware remains untested
by this project.
