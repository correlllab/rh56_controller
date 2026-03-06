"""
mink_ctrl_table.py — Offline mink IK ctrl lookup table for the Inspire RH56 hand.

After running compare_grasp_methods.py, the sweep results are cached as .npz files
under mink_vs_analytical/cache/.  This module loads those caches and provides fast
(microsecond) ctrl array lookup via linear interpolation, eliminating the need to run
the iterative mink IK solver at grasp-execution time.

Usage:
    from rh56_controller.mink_ctrl_table import MinkCtrlTable

    table = MinkCtrlTable()               # auto-detects cache dir relative to this file
    ctrl  = table.lookup("line", 0.045)   # → np.ndarray shape (6,)
    ctrl  = table.lookup("plane", 0.080, n_fingers=4)
    ctrl  = table.lookup("cylinder", 0.065)

Ctrl array layout:
    [pinky=0, ring=1, middle=2, index=3, thumb_bend=4, thumb_yaw=5]
    (matches MinkGraspResult.ctrl and inspire_right.xml actuator order)

Cache key convention:
    "line"       → cache/line.npz
    "plane"      → cache/plane_{n_fingers}f.npz   (n_fingers ∈ {3, 4, 5})
    "cylinder"   → cache/cylinder.npz

Notes:
    - Width / diameter in metres.
    - Values outside the cached range are clamped to the nearest endpoint.
    - Converged-only flag: by default, samples where mink did not converge are
      excluded from the interpolation.  Set converged_only=False to use all samples.
    - If the cache file does not exist, MinkCtrlTable raises FileNotFoundError with
      a helpful message pointing to compare_grasp_methods.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Ctrl channel layout  (must match mink_grasp_planner._ACT_TO_JOINT order)
# ---------------------------------------------------------------------------
CTRL_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_yaw"]
N_CTRL = 6

_HERE      = Path(__file__).parent.parent
_CACHE_DIR = _HERE / "mink_vs_analytical" / "cache"


class MinkCtrlTable:
    """
    Offline lookup table for mink grasp ctrl arrays.

    Loads pre-computed sweep results and provides fast interpolated ctrl lookups.

    Args:
        cache_dir:      Directory containing .npz cache files.  Defaults to the
                        standard mink_vs_analytical/cache/ path.
        converged_only: If True (default), only samples where mink converged are
                        used for interpolation.  Unconverged samples may produce
                        unreliable ctrl values.
        kind:           scipy interpolation kind — "linear" (default) or "cubic".
    """

    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        converged_only: bool = True,
        kind: str = "linear",
    ) -> None:
        self._cache_dir     = Path(cache_dir) if cache_dir else _CACHE_DIR
        self._converged_only = converged_only
        self._kind           = kind
        self._tables: Dict[str, interp1d] = {}   # key → interpolator (widths_m → ctrl N×6)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def lookup(
        self,
        mode: str,
        width_m: float,
        n_fingers: int = 4,
    ) -> np.ndarray:
        """
        Return interpolated ctrl array (shape (6,)) for the given mode and width.

        Args:
            mode:      "line", "plane", or "cylinder".
            width_m:   Target width / diameter in metres.
            n_fingers: Number of non-thumb fingers (3, 4, or 5).  Only used for
                       mode="plane".

        Returns:
            ctrl: np.ndarray shape (6,), [pinky, ring, middle, index, thumb_bend, thumb_yaw]

        Raises:
            FileNotFoundError: If the cache file for this mode has not been built yet.
            ValueError:        If mode is unrecognised.
        """
        cache_key = self._cache_key(mode, n_fingers)
        if cache_key not in self._tables:
            self._load(cache_key)
        return self._tables[cache_key](width_m)

    def lookup_finger(
        self,
        mode: str,
        width_m: float,
        finger: str,
        n_fingers: int = 4,
    ) -> float:
        """
        Return a single finger's ctrl value (radians).

        Args:
            finger: One of "pinky", "ring", "middle", "index", "thumb_bend", "thumb_yaw".
        """
        idx = CTRL_NAMES.index(finger)
        return float(self.lookup(mode, width_m, n_fingers)[idx])

    def loaded_modes(self) -> list[str]:
        """Return list of already-loaded cache keys."""
        return list(self._tables.keys())

    def preload_all(self) -> None:
        """Load all available cache files up front."""
        for path in self._cache_dir.glob("*.npz"):
            key = path.stem   # e.g. "line", "plane_4f", "cylinder"
            if key not in self._tables:
                try:
                    self._load(key)
                except Exception as e:
                    print(f"[MinkCtrlTable] warning: could not load {path.name}: {e}")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _cache_key(mode: str, n_fingers: int) -> str:
        if mode == "line":
            return "line"
        elif mode == "plane":
            return f"plane_{n_fingers}f"
        elif mode == "cylinder":
            return "cylinder"
        raise ValueError(f"Unknown grasp mode: {mode!r}.  Use 'line', 'plane', or 'cylinder'.")

    def _load(self, cache_key: str) -> None:
        path = self._cache_dir / f"{cache_key}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"Mink ctrl cache not found: {path}\n"
                "Run compare_grasp_methods.py first to generate the cache:\n"
                "  uv run python -m rh56_controller.compare_grasp_methods --no-show"
            )

        data = np.load(str(path), allow_pickle=True)
        widths_mm  = data["widths_mm"]           # (N,)
        ctrl_arr   = data["mink_ctrl_arr"]        # (N, 6)
        conv       = data["mink_conv"].astype(bool)  # (N,)

        if self._converged_only and conv.any():
            widths_mm = widths_mm[conv]
            ctrl_arr  = ctrl_arr[conv]

        if len(widths_mm) < 2:
            raise ValueError(
                f"Cache {cache_key}: fewer than 2 converged samples — "
                "cannot build interpolator.  Re-run with more widths or "
                "set converged_only=False."
            )

        widths_m = widths_mm / 1000.0   # mm → m
        # Build one interpolator per ctrl channel, then wrap in a single function
        interps = [
            interp1d(widths_m, ctrl_arr[:, i],
                     kind=self._kind,
                     bounds_error=False,
                     fill_value=(ctrl_arr[0, i], ctrl_arr[-1, i]))
            for i in range(N_CTRL)
        ]

        def _lookup(w_m: float) -> np.ndarray:
            return np.array([float(f(w_m)) for f in interps])

        self._tables[cache_key] = _lookup
        print(f"[MinkCtrlTable] loaded {cache_key} ({len(widths_mm)} samples, "
              f"range [{widths_m[0]*1000:.0f}–{widths_m[-1]*1000:.0f} mm])")


# ---------------------------------------------------------------------------
# Quick demo / validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    table = MinkCtrlTable()
    table.preload_all()

    print("\nLookup examples:")
    for mode, w, nf in [
        ("line",     0.045, 2),
        ("plane",    0.060, 3),
        ("plane",    0.080, 4),
        ("plane",    0.100, 5),
        ("cylinder", 0.065, 5),
    ]:
        try:
            ctrl = table.lookup(mode, w, n_fingers=nf)
            thumb = np.degrees(ctrl[4])
            idx   = np.degrees(ctrl[3])
            print(f"  {mode:8s} {nf}f  w={w*1000:.0f}mm  "
                  f"thumb_bend={thumb:.1f}°  index={idx:.1f}°  "
                  f"ctrl={np.degrees(ctrl).round(1)}")
        except FileNotFoundError as e:
            print(f"  {mode}: {e}")
