from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path


def _candidate_openmp_paths() -> list[Path]:
    candidates = [
        os.environ.get("LIBOMP_PATH"),
        "/opt/homebrew/opt/libomp/lib/libomp.dylib",
        "/usr/local/opt/libomp/lib/libomp.dylib",
        "/Library/Frameworks/R.framework/Resources/lib/libomp.dylib",
    ]

    r_framework = Path("/Library/Frameworks/R.framework/Versions")
    if r_framework.exists():
        candidates.extend(
            str(path)
            for path in sorted(r_framework.glob("*/Resources/lib/libomp.dylib"), reverse=True)
        )

    return [Path(path) for path in candidates if path]


def ensure_openmp_runtime() -> str | None:
    """Load libomp on macOS before importing XGBoost when Homebrew is unavailable."""
    if platform.system() != "Darwin":
        return None

    for candidate in _candidate_openmp_paths():
        if not candidate.exists():
            continue
        try:
            existing_library_path = os.environ.get("DYLD_LIBRARY_PATH", "")
            existing_fallback_path = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
            library_dir = str(candidate.parent)
            if library_dir not in existing_library_path.split(":"):
                os.environ["DYLD_LIBRARY_PATH"] = ":".join(
                    [part for part in [library_dir, existing_library_path] if part]
                )
            if library_dir not in existing_fallback_path.split(":"):
                os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
                    [part for part in [library_dir, existing_fallback_path] if part]
                )
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return str(candidate)
        except OSError:
            continue
    return None
