"""Check an extracted release, including both public launcher commands."""
import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile


def check(package: Path) -> None:
    package = package.resolve()
    windows = os.name == "nt"
    suffix = ".exe" if windows else ""
    backend = "ggml-zest.dll" if windows else "libggml-zest.so"
    required = [backend, "bfp16_gemm.xclbin", "llama-server" + suffix,
                "zest" + suffix, "npu" + suffix]
    required += ["Start-Lemonade.cmd", "tools/start-lemonade.ps1"] if windows else ["libggml-zest-xrt.so"]
    for name in required:
        if not (package / name).is_file():
            raise RuntimeError(f"Missing package file: {name}")

    manifest = {}
    for line in (package / "SHA256SUMS.txt").read_text(encoding="utf-8-sig").splitlines():
        digest, name = line.split("  ", 1)
        path = (package / name).resolve()
        if not path.is_relative_to(package):
            raise RuntimeError(f"Manifest path outside package: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Checksum mismatch: {name}")
        manifest[name] = digest
    files = set()
    for path in package.rglob("*"):
        if path.is_symlink():
            # Upstream Linux libraries use SONAME links; package.sh hashes
            # regular files, so validate each link against its hashed target.
            target = path.resolve()
            if (not target.is_relative_to(package) or not target.is_file()
                    or target.relative_to(package).as_posix() not in manifest):
                raise RuntimeError(f"Invalid or unhashed symlink target: {path.name}")
        elif path.is_file() and path.name != "SHA256SUMS.txt":
            files.add(path.relative_to(package).as_posix())
    if files != set(manifest):
        raise RuntimeError(f"Manifest coverage differs: {files ^ set(manifest)}")
    if (package / ("zest" + suffix)).read_bytes() != (package / ("npu" + suffix)).read_bytes():
        raise RuntimeError("zest and npu must contain the same launcher")

    environment = os.environ.copy()
    for name in ("GGML_BACKEND_PATH", "GGML_XDNA_KERNELS"):
        environment.pop(name, None)
    # A different working directory prevents accidental reliance on the checkout.
    with tempfile.TemporaryDirectory(prefix="zest-launcher-") as working:
        for command in ("zest", "npu"):
            launcher = package / (command + suffix)
            if not windows and not os.access(launcher, os.X_OK):
                raise RuntimeError(f"Launcher isn't executable: {command}")
            result = subprocess.run([str(launcher), "llama-server", "--list-devices"],
                                    cwd=working, env=environment, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    errors="replace", timeout=60)
            print(result.stdout, end="")
            if result.returncode or not re.search(r"^\s*Vulkan0:", result.stdout, re.MULTILINE):
                raise RuntimeError(f"{command} failed to enumerate Vulkan devices")
    print(f"Verified {len(manifest)} checksums and both packaged commands.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    check(parser.parse_args().package)
