"""Build the GitHub Pages site using only Python's standard library."""
from pathlib import Path
import argparse
import shutil

ROOT = Path(__file__).resolve().parents[1]


def build(destination: Path) -> None:
    destination = destination.resolve()
    # Only replace the dedicated, ignored output directory in this checkout.
    if destination != ROOT / "dist" / "site":
        raise SystemExit("Site output must be this checkout's dist/site directory")
    if destination.is_symlink() or (ROOT / "dist").is_symlink():
        raise SystemExit("Site output cannot be a symlink")
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(ROOT / "website", destination)
    assets = destination / "assets"
    assets.mkdir(exist_ok=True)
    for name in ("lemonade-logo-light.png", "lemonade-logo-dark.png", "lemonade-LICENSE.txt",
                 "lemonade-results.svg", "jakarta-latin.woff2", "jakarta-OFL.txt",
                 "lemonade-speed-comparison.mp4", "lemonade-speed-comparison.png"):
        shutil.copy2(ROOT / "docs" / "assets" / name, assets / name)
    shutil.copy2(ROOT / "docs/results/lemonade-desktop-answer.png", assets / "lemonade-desktop.png")
    shutil.copy2(ROOT / "docs/results/lemonade-2026-10-21.json", assets / "lemonade-measurements.json")
    shutil.copy2(ROOT / "docs/results/lemonade-speed-comparison-2026-10-22.json", assets / "lemonade-speed-comparison.json")
    shutil.copytree(ROOT / "docs/results/headline-models-2026-10-21", assets / "model-results")
    (destination / ".nojekyll").touch()
    print(f"Built {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "dist/site")
    build(parser.parse_args().out)
