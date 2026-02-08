# famy_trajectories.py
"""
FUNCTION 3 — Warming / trajectory plot from saved FAMYs folder.

Purpose
-------
- Reads EPW files directly from disk (no dependency on epwclass.epw).
- Auto-detects scenarios from filenames in a location's /famy folder.
- Builds a tidy trajectory table (annual + JJA mean dry-bulb temperature).
- Plots smoothed scenario trajectories with a red gradient (darkest = highest emissions).

Location handling
-----------------
- If your notebook already defines `location` (e.g., from Step 1), use:
    plot_famy_from_location(location, ...)
- If you want *auto-detection* from the local repo folders, use:
    plot_famy_auto(base_dir="/content/wxbuild/epwdata", ...)

Expected FAMY filename pattern (example)
---------------------------------------
Boston__MA__USA_famy_2043_ssp370_CanESM5.epw

This file does NOT auto-run plots on import.
"""

from __future__ import annotations

import os
import re
import glob
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Path + parsing helpers
# ---------------------------

_FAMY_RE = re.compile(r"_famy_(\d{4})_(ssp\d{3})_", re.IGNORECASE)


def get_famy_dir(location: str, base_dir: str = "/content/wxbuild/epwdata") -> str:
    """Return the expected famy directory for a location."""
    return os.path.join(base_dir, location, "famy")


def auto_detect_location(base_dir: str = "/content/wxbuild/epwdata") -> str:
    """Auto-detect a location folder under base_dir.

    Preference order:
    1) folder that contains a 'famy' subfolder with at least 1 *.epw
    2) otherwise, first folder under base_dir
    """
    if not os.path.isdir(base_dir):
        raise RuntimeError(f"Base EPW directory not found: {base_dir}")

    candidates = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith(".")
    )
    if not candidates:
        raise RuntimeError(f"No location folders found under {base_dir}")

    for loc in candidates:
        famy_dir = get_famy_dir(loc, base_dir=base_dir)
        if os.path.isdir(famy_dir) and glob.glob(os.path.join(famy_dir, "*.epw")):
            return loc

    return candidates[0]


def parse_famy_filename(path: str) -> Tuple[Optional[int], Optional[str]]:
    """Extract (year, scenario) from a FAMY EPW filename."""
    base = os.path.basename(path)
    m = _FAMY_RE.search(base)
    if not m:
        return None, None
    return int(m.group(1)), m.group(2).lower()


def epw_annual_and_jja_means(epw_path: str, jja_months=(6, 7, 8)) -> Tuple[float, float]:
    """Compute annual and JJA mean dry-bulb temperature from an EPW file.

    EPW has 8 header lines, then hourly rows. We read:
      - Month (col 1)
      - DryBulb (col 6)
    """
    df = pd.read_csv(
        epw_path,
        skiprows=8,
        header=None,
        usecols=[1, 6],
        names=["Month", "DryBulb_C"],
        encoding_errors="ignore",
    )
    month = pd.to_numeric(df["Month"], errors="coerce")
    dbt = pd.to_numeric(df["DryBulb_C"], errors="coerce")

    t_annual = float(np.nanmean(dbt.values))
    t_jja = float(np.nanmean(dbt[month.isin(jja_months)].values))
    return t_annual, t_jja


# ---------------------------
# Data build
# ---------------------------

def build_famy_trajectory_table(famy_dir: str, location_key: Optional[str] = None) -> pd.DataFrame:
    """Read all *.epw in famy_dir and return a tidy trajectory dataframe."""
    if not os.path.isdir(famy_dir):
        raise RuntimeError(f"FAMY folder not found: {famy_dir}")

    epw_files = sorted(glob.glob(os.path.join(famy_dir, "*.epw")))
    if not epw_files:
        raise RuntimeError(f"No EPW files found in {famy_dir}")

    rows = []
    for f in epw_files:
        year, scenario = parse_famy_filename(f)
        if year is None or scenario is None:
            continue
        t_annual, t_jja = epw_annual_and_jja_means(f)
        rows.append(
            {
                "location": location_key or os.path.basename(os.path.dirname(famy_dir)),
                "year": year,
                "scenario": scenario,
                "tmean_annual": t_annual,
                "tmean_jja": t_jja,
                "file": os.path.basename(f),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "Parsed zero FAMY files. Check filename pattern includes '_famy_<year>_<ssp>_'."
        )

    return df.sort_values(["scenario", "year"]).reset_index(drop=True)


# ---------------------------
# Plotting
# ---------------------------

def plot_famy_smoothed_clean(
    df: pd.DataFrame,
    *,
    location_label: Optional[str] = None,
    metric: str = "tmean_jja",
    window: int = 5,
    figsize: Tuple[float, float] = (14, 5),
    scenario_colors: Optional[Dict[str, str]] = None,
):
    """Plot smoothed trajectories per scenario."""
    if metric not in df.columns:
        raise ValueError(f"metric '{metric}' not found in df columns: {list(df.columns)}")

    if scenario_colors is None:
        scenario_colors = {
            "ssp126": "#f3b4b4",  # light
            "ssp245": "#e77b7b",
            "ssp370": "#c94b4b",
            "ssp585": "#7a0c0c",  # darkest
        }

    scenarios = sorted(df["scenario"].dropna().unique().tolist())

    plt.figure(figsize=figsize)
    ax = plt.gca()

    for s in scenarios:
        sub = df[df["scenario"] == s].sort_values("year").copy()
        y = pd.to_numeric(sub[metric], errors="coerce").astype(float)
        y_smooth = (
            y.rolling(window=window, center=True)
            .mean()
            .interpolate(limit_direction="both")
        )

        ax.plot(
            sub["year"].values,
            y_smooth.values,
            linewidth=3,
            label=s,
            color=scenario_colors.get(s, None),
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Dry-Bulb Temperature (°C)")

    loc_txt = location_label or (df["location"].iloc[0] if "location" in df.columns and len(df) else "")
    ax.set_title(f"{loc_txt} — {metric} (rolling mean, window={window})")

    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    plt.show()


def plot_famy_from_location(
    location: str,
    *,
    base_dir: str = "/content/wxbuild/epwdata",
    metric: str = "tmean_jja",
    window: int = 5,
    figsize: Tuple[float, float] = (14, 5),
    scenario_colors: Optional[Dict[str, str]] = None,
    return_df: bool = True,
):
    """Convenience wrapper: build df from location and plot."""
    famy_dir = get_famy_dir(location, base_dir=base_dir)
    df = build_famy_trajectory_table(famy_dir, location_key=location)

    scenarios = sorted(df["scenario"].unique().tolist())
    years = (int(df["year"].min()), int(df["year"].max()))
    print("✅ Scenarios detected:", scenarios)
    print("✅ Years range:", years[0], "to", years[1])
    print("✅ Files parsed:", len(df))

    plot_famy_smoothed_clean(
        df,
        location_label=location,
        metric=metric,
        window=window,
        figsize=figsize,
        scenario_colors=scenario_colors,
    )

    return df if return_df else None


def plot_famy_auto(
    *,
    base_dir: str = "/content/wxbuild/epwdata",
    location: Optional[str] = None,
    metric: str = "tmean_jja",
    window: int = 5,
    figsize: Tuple[float, float] = (14, 5),
    scenario_colors: Optional[Dict[str, str]] = None,
    return_df: bool = True,
):
    """Auto-detect location (if not provided), then plot."""
    loc = location or auto_detect_location(base_dir=base_dir)
    print(f"📍 Using location: {loc}")
    return plot_famy_from_location(
        loc,
        base_dir=base_dir,
        metric=metric,
        window=window,
        figsize=figsize,
        scenario_colors=scenario_colors,
        return_df=return_df,
    )
