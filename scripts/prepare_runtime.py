"""Assemble the ASVO runtime from the Git submodule and experiment overrides."""
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream/ASVO"
RUNTIME = ROOT / ".asvo-runtime"
OVERRIDES = ROOT / "runtime_overrides"


def main():
    expected = subprocess.check_output(
        ["git", "ls-files", "--stage", "upstream/ASVO"], cwd=ROOT, text=True
    ).split()[1]
    if not (UPSTREAM / ".git").exists():
        raise SystemExit("Initialize ASVO with git submodule update --init.")
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip()
    if actual != expected or subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--"], cwd=UPSTREAM
    ).returncode:
        raise SystemExit("ASVO must match the pinned submodule commit.")
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "examples", "concordia"], cwd=UPSTREAM
    ).decode().split("\0")
    for filename in files:
        if not filename.endswith(".py"):
            continue
        if filename.startswith(("concordia/prefabs/", "concordia/environment/engines/")):
            continue
        target = RUNTIME / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        override = OVERRIDES / filename
        shutil.copyfile(override if override.exists() else UPSTREAM / filename, target)
    print(f"Prepared {RUNTIME.name} from ASVO {actual[:8]}.")


if __name__ == "__main__":
    main()
