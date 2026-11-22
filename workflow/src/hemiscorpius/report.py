"""Simple text report that is printed to the screen and saved to a file."""
import sys
from pathlib import Path


class Report:
    """Use as a context manager: everything printed inside is also saved."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        self._file = open(self.path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = self
        return self

    def write(self, text):
        self._stdout.write(text)
        self._file.write(text)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def __exit__(self, *exc):
        sys.stdout = self._stdout
        self._file.close()
        print(f"\nReport saved to: {self.path}")
        return False


def section(title: str):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def ok(msg):
    print(f"  [OK]    {msg}")


def warn(msg):
    print(f"  [WARN]  {msg}")


def fail(msg):
    print(f"  [FAIL]  {msg}")


def write_run_info(path, cfg_file):
    """Record when and with which software versions a step was run (reproducibility)."""
    import hashlib
    import json
    import platform
    import sys as _sys
    from datetime import datetime
    from importlib import metadata
    pkgs = ["numpy", "pandas", "xarray", "rasterio", "geopandas", "netCDF4", "matplotlib",
            "scipy", "scikit-learn", "statsmodels"]
    versions = {}
    for p in pkgs:
        try:
            versions[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versions[p] = "not installed"
    from datetime import timezone
    root = Path(__file__).resolve().parents[2]
    code_files = [p for folder in ("scripts", "src") for p in (root / folder).rglob("*.py")
                  if "__pycache__" not in p.parts]
    info = dict(run_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                python=_sys.version.split()[0], platform=platform.platform(),
                config_file=str(Path(cfg_file).resolve()),
                config_sha256=hashlib.sha256(Path(cfg_file).read_bytes()).hexdigest(),
                source_sha256={p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(code_files)},
                packages=versions)
    Path(path).write_text(json.dumps(info, indent=2), encoding="utf-8")
