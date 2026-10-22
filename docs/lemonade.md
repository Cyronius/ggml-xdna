# Use ggml-xdna with Lemonade

Verified on Windows on 2026-10-21 with Lemonade **2026.43.3**, the unmodified
llama.cpp **b10944 Vulkan** release, and this repository's `ggml-xdna.dll`.
Lemonade forwards chat requests to llama-server; the add-on sends eligible
prompt work to the NPU while the Vulkan GPU handles attention and generation.
Neither Lemonade nor llama.cpp needs a source patch for this setup.

This uses Lemonade's custom backend binary setting. Its default Vulkan
backend is b11393 in this release, which is **not** the version this add-on
was built against. Do not drop the b10944 DLL into an arbitrary newer build.
This check does not establish compatibility with that default backend.

See the [clean-package and desktop verification report](lemonade-validation.md)
for the build, model-quality, UI and fallback evidence.

[Desktop walkthrough](lemonade-desktop-walkthrough.mp4) (captured desktop
screenshots; not a real-time latency recording).

## Start from the Windows ZIP

1. Download the [Windows preview](https://github.com/Cyronius/ggml-xdna/releases/tag/lemonade-preview-20261021-r2)
   and extract it. Keep all the files together.
2. Quit the existing Lemonade tray server. The desktop chat window can stay open.
3. Double-click `Start-Lemonade.cmd`. It looks for an installed Lemonade server;
   if it can't find one, choose `LemonadeServer.exe` in your Lemonade `bin`
   folder. The portable `lemond.exe` works too. Choose your local `.gguf` file
   in the next dialog. Cancel either dialog to stop without starting a server.
4. Leave the launcher window open. In Lemonade App, set **Settings > Base URL**
   to `http://127.0.0.1:13315` and click **Save**. Pick the imported model and
   send a long prompt. The first request can take longer while weights are
   converted for the NPU.

This preview was tested with [Lemonade 2026.43.3](https://github.com/lemonade-sdk/lemonade/releases/tag/v2026.43.3).
It includes the matching backend, but no Lemonade installer, AMD driver, or
model weights. Other Lemonade versions haven't been verified here.

To go back to your regular Lemonade setup, stop this server with Ctrl+C,
restart the normal tray server, and restore your previous Base URL
(normally `http://localhost:13305`). The launcher's configuration is separate.

### If something goes wrong

| What you see | What to do |
|---|---|
| LemonadeServer is already running | Quit the tray server first. Closing the chat window alone doesn't stop it. |
| A file picker asks for the server | Choose `LemonadeServer.exe` or portable `lemond.exe`, rather than `lemonade-app.exe`. |
| Port 13315 is in use | Stop the previous isolated server, or use `-Port` from a terminal. |
| The app says disconnected | Check the launcher's printed Base URL, save it in the app, and keep the server running. |
| Your model isn't listed | Pick its GGUF file again and make sure the app points at this server. |
| NPU unavailable; using the GPU | Read `third_party/lemonade-instance/device-probe.log` in the extracted folder for the reason. |
| No NPU calls on a short message | Try a longer passage. Chunks under 1,024 tokens normally stay on the GPU. |

The launcher prints the path to `server.log`. NPU call counts in that log
are the check that matters; Lemonade may still label this custom recipe
as GPU-only. Avoid another NPU backend running at the same time.

## Start from a terminal

For the Windows desktop app, use the included launcher from an unpacked
ggml-xdna release (or the repository):

Quit any existing Lemonade tray server first: `LemonadeServer.exe` enforces a
single running instance. The desktop chat window can remain open. Portable
`lemond.exe` can instead run alongside an installed server on a different
port.

```powershell
powershell -ExecutionPolicy Bypass -File tools\start-lemonade.ps1 `
    -LemonadeExe 'C:\path\to\Lemonade\bin\LemonadeServer.exe' `
    -Model 'C:\models\Qwen3-4B-Q4_K_M.gguf'
```

For a repository build, run `build.cmd` first. The launcher finds the pinned
llama.cpp next to the release's `tools` directory, or in the repository's
`third_party/llama-b10944`; `-LlamaDir` overrides that location. It probes the
NPU before choosing `XDNA0,Vulkan0`, and selects `Vulkan0` when the NPU is
unavailable. `-GpuOnly` explicitly chooses the GPU. Configuration, probe
output and server logs live under a separate `-StateDir` (default:
`third_party/lemonade-instance`). Keep this server running during the test.

Open **Lemonade App**, set **Settings > Base URL** to
`http://127.0.0.1:13315`, and click **Save**. Select the imported model in the
chat model picker and send a long prompt. The launcher saves that model's
device and batch settings, so a load initiated by the app uses them too.
It also accepts the portable `lemond.exe` in `-LemonadeExe` when the desktop
app is not needed. A missing NPU can fall back at startup; runtime failure
recovery has the limitations described in the [technical guide](technical-guide.md#limitations).

The launcher saves `merge_args=true` so Lemonade inherits the saved batch
and tensor-split arguments when its desktop app loads the model. Use
`merge_args=false` only when supplying all custom arguments explicitly in
that same API request, as the benchmark script does.

### Manual setup with portable Lemonade

First build this repository with `build.cmd`, or unpack its b10944 release.
Download and unpack
[`lemonade-embeddable-2026.43.3-windows-x64.zip`](https://github.com/lemonade-sdk/lemonade/releases/tag/v2026.43.3).
No system Lemonade installation is necessary.

From the repository root in PowerShell, edit the first two paths below.
For a release ZIP, set `$llama` to the unpacked release directory instead.
Use a model already downloaded to disk; the example benchmark uses
`Qwen3-4B-Q4_K_M.gguf`.

```powershell
$lemond = 'C:\path\to\lemonade-embeddable-2026.43.3-windows-x64\lemond.exe'
$models = 'C:\path\to\models'
$llama = (Resolve-Path third_party\llama-b10944).Path
$demo = Join-Path (Get-Location) 'third_party\lemonade-demo'
New-Item -ItemType Directory -Force "$demo\cache", "$demo\config" | Out-Null
@{
    config_version = 2
    host = '127.0.0.1'
    port = 13315
    broadcast = $false
    offline = $true
    no_fetch_executables = $true
    log_level = 'debug'
    models_dir = "$demo\models"
    extra_models_dir = $models
    llamacpp = @{
        backend = 'vulkan'
        vulkan_bin = "$llama\llama-server.exe"
    }
} | ConvertTo-Json -Depth 4 | Set-Content "$demo\config\config.json"
$env:GGML_BACKEND_PATH = "$llama\ggml-xdna.dll"
$env:GGML_XDNA_TRACE = '1'
& $lemond "$demo\cache" "$demo\config" --port 13315 --host 127.0.0.1 --no-broadcast
```

Both positional directories matter: the first holds runtime/cache data,
the second persistent configuration. In the tested release's offline path,
`vulkan_bin` must name **the executable**, despite the documentation showing
a directory. A directory produced Windows error 5 when Lemonade tried to
execute it. The full executable path worked.

Use a second terminal for requests:

```powershell
$api = 'http://127.0.0.1:13315/api/v1'
(Invoke-RestMethod "$api/models").data |
    Where-Object source -eq 'extra_models_dir' | Select-Object id
$load = @{
    model_name = 'extra.Qwen3-4B-Q4_K_M'
    ctx_size = 8192
    llamacpp_backend = 'vulkan'
    llamacpp_device = 'XDNA0,Vulkan0'
    llamacpp_args = '-ts 0,1 -ub 2048 -b 2048 --flash-attn on --parallel 1'
    merge_args = $false
} | ConvertTo-Json
Invoke-RestMethod "$api/load" -Method Post -ContentType application/json -Body $load
```

Use `llamacpp_device` for device selection: Lemonade reserves `--device` in
`llamacpp_args`. `-ts 0,1` keeps layer storage on the GPU, and `-ub 2048 -b
2048` gives the NPU sufficiently large chunks. `--parallel 1` makes this a
single-request check. The model's bare ID also works when unambiguous;
`extra.` explicitly selects a model imported from `extra_models_dir`.

Clients can now send ordinary OpenAI-compatible chat requests to
`http://127.0.0.1:13315/api/v1`, naming that model. Prompts under 1,024 tokens
stay on the GPU by default. Use a long prompt to demonstrate the NPU.

## Measure your model

On this private test instance, run:

```powershell
powershell -ExecutionPolicy Bypass -File tools\bench-lemonade.ps1 `
    -Model extra.Qwen3-4B-Q4_K_M `
    -Rounds 3 -OutDir third_party\lemonade-results-1
```

The script unloads and reloads that model for each configuration, warms it
with one full request, then measures another with prompt caching disabled.
It alternates which device configuration runs first between rounds. GPU
and NPU configurations use the same model, binary, context, prompt, chunk
sizes and generation settings. Results include complete measured responses,
a CSV, and the request JSON. The script unloads its model when finished.

These are **server-reported prefill and decode timings**, excluding model
loading and warmup. They are not measured client streaming time to first
token. The script rejects cached prompt work and prompts below 2,048 tokens.
It does not check output quality or prove hardware execution on its own.

For separate streamed correctness checks (requires Python 3):

```powershell
python tools\check-lemonade.py --model extra.Qwen3-4B-Q4_K_M `
    --output third_party\lemonade-quality.json
```

This reloads the named model on GPU and NPU. It reports answer correctness,
formatting, and client time to first content. `--mode saved` instead uses
the launcher's saved settings; `--strict-format` makes formatting mismatches
fatal. Use the private test instance, since these checks reload its model.

For evidence that the NPU actually ran, retain the server output and find
`xdna times` with a positive NPU call count. The initial successful request
reported:

```text
xdna times, 199 matmuls in 30 pieces: all 1397.4 ms | read inputs 75.6, encode 30.7, NPU 664.5 (342 calls), decode 0.0, write results 100.1, weight copies 5972.5 ms
```

The 2,781-token prompt returned a 44-token answer through Lemonade, with
fingerprint `b10944-b6b003d2c` and zero cached tokens. That initial smoke
request measured 555.5 prompt tokens/s; it is not a paired benchmark result.

Lemonade reported `NPU hardware: No` in this sandbox, even while the add-on
successfully ran NPU calls. Its native hardware discovery and GPU recipe
label do not establish whether this custom backend used the NPU. Avoid
running another NPU backend concurrently: this configuration is represented
as a GPU recipe, so Lemonade cannot be assumed to reserve the NPU for it.

Stop the isolated server with Ctrl+C when finished.

## Paired result: 2026-10-21

Ryzen AI 9 HX **PRO** 370 with Radeon 890M, Windows, Qwen3-4B Q4_K_M,
2,781 prompt tokens. Three pairs using the script above:

| Round | GPU prefill, ms | NPU-assisted prefill, ms | Speed ratio |
|---|---:|---:|---:|
| 1 | 6,445 | 5,113 | 1.261x |
| 2 | 6,623 | 5,227 | 1.267x |
| 3 | 7,656 | 5,334 | 1.435x |

The median paired prefill speed ratio was **1.267x** (about 21% less prefill
time). This is one workload in one sitting, not a general speed claim.
GPU-only performance moved from 431 to 363 tokens/s across the sitting;
background load and thermals were not controlled. Decode ranged from
15.6 to 20.8 tokens/s, so these runs establish no decode advantage.

[The measurement record](results/lemonade-2026-10-21.json) contains all six
measured responses, binary and model SHA-256 hashes, settings and caveats.
[NPU trace excerpts](results/lemonade-2026-10-21-npu-trace.txt) show positive
hardware call counts in warmup and measured passes. All measured responses
reported the same 2,781 prompt tokens, b10944 fingerprint and zero cache use.
Warmup matters: each NPU run's first full request took about 12 seconds of
prefill while weight copies finished, compared with 5.1-5.3 seconds in the
measured requests. The warm result is not cold-start latency.

## References

- [Lemonade's backend binary selection](https://github.com/lemonade-sdk/lemonade/blob/v2026.43.3/docs/guide/configuration/README.md#backend-binary-selection)
- [Lemonade's pinned backend versions](https://github.com/lemonade-sdk/lemonade/blob/v2026.43.3/src/cpp/resources/backend_versions.json)
- [Lemonade's llama.cpp subprocess setup](https://github.com/lemonade-sdk/lemonade/blob/v2026.43.3/src/cpp/server/backends/llamacpp/llamacpp_server.cpp)
- [Embeddable Lemonade runtime](https://github.com/lemonade-sdk/lemonade/blob/v2026.43.3/docs/embeddable/runtime.md)
