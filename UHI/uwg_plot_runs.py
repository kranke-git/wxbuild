
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------
# Helpers
# ---------------------------

def read_epw_safe(path: str | Path) -> pd.DataFrame:
    """Read an EPW (skip 8 header rows) and return columns needed for comparisons."""
    path = Path(path)
    df = pd.read_csv(path, skiprows=8, header=None).copy()

    # EPW standard positions
    df["year"]  = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    df["month"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    df["day"]   = pd.to_numeric(df.iloc[:, 2], errors="coerce")
    df["hour"]  = pd.to_numeric(df.iloc[:, 3], errors="coerce")
    df["dbt"]   = pd.to_numeric(df.iloc[:, 6], errors="coerce")

    df["dpt"]   = pd.to_numeric(df.iloc[:, 7], errors="coerce") if df.shape[1] > 7 else np.nan
    df["rh"]    = pd.to_numeric(df.iloc[:, 8], errors="coerce") if df.shape[1] > 8 else np.nan

    df["hoy"]   = np.arange(len(df))
    return df[["year", "month", "day", "hour", "hoy", "dbt", "dpt", "rh"]]


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="both", direction="out")
    ax.grid(True, alpha=0.25)


def _detect_city_from_epwdata(wxbuild_root: str | Path) -> str:
    """Detect city folders under epwdata.

    Filters out non-city folders (e.g., 'cmip6') by requiring a 'rmy' subfolder
    containing at least one EPW.
    """
    wxbuild_root = Path(wxbuild_root)
    epwdata = wxbuild_root / "epwdata"
    if not epwdata.exists():
        raise FileNotFoundError(f"Missing epwdata folder: {epwdata}")

    city_dirs = []
    for p in sorted(epwdata.iterdir()):
        if not p.is_dir():
            continue
        rmy_dir = p / "rmy"
        if rmy_dir.exists() and any(rmy_dir.glob("*.epw")):
            city_dirs.append(p.name)

    if len(city_dirs) == 1:
        return city_dirs[0]

    raise RuntimeError(
        f"Could not uniquely detect a city under {epwdata}. "
        f"Found candidate cities with rmy EPWs: {city_dirs}. "
        f"Please pass city explicitly."
    )


def find_base_rmy_epw(wxbuild_root: str | Path, city: str) -> Path:
    """Pick the base EPW from <city>/rmy (prefer *RMY*.epw)."""
    rmy_dir = Path(wxbuild_root) / "epwdata" / city / "rmy"
    if not rmy_dir.exists():
        raise FileNotFoundError(f"Missing rmy folder: {rmy_dir}")
    epws = sorted(rmy_dir.glob("*.epw"))
    if not epws:
        raise FileNotFoundError(f"No EPWs in: {rmy_dir}")
    rmy_named = [p for p in epws if "rmy" in p.stem.lower()]
    return rmy_named[0] if rmy_named else epws[0]


def list_uwg_runs(wxbuild_root: str | Path, city: str) -> List[Path]:
    """Return UWG EPWs in <city>/uwg matching City_UWG_RunX.epw, sorted by X."""
    uwg_dir = Path(wxbuild_root) / "epwdata" / city / "uwg"
    if not uwg_dir.exists():
        return []
    pat = re.compile(rf"^{re.escape(city)}_UWG_Run(\d+)\.epw$", re.IGNORECASE)
    items: List[Tuple[int, Path]] = []
    for p in uwg_dir.glob(f"{city}_UWG_Run*.epw"):
        m = pat.match(p.name)
        if m:
            items.append((int(m.group(1)), p))
    items.sort(key=lambda x: x[0])
    return [p for _, p in items]


def _palette_reds(n: int) -> List[str]:
    """Generate n shades of deep-to-light red (no external deps)."""
    if n <= 0:
        return []
    # two endpoints (deep red -> lighter red)
    deep = np.array([0x7A, 0x00, 0x00]) / 255.0   # #7A0000
    light = np.array([0xD0, 0x4A, 0x4A]) / 255.0  # #D04A4A
    cols = []
    for i in range(n):
        t = 0 if n == 1 else i / (n - 1)
        rgb = (1 - t) * deep + t * light
        cols.append("#%02X%02X%02X" % tuple((rgb * 255).round().astype(int)))
    return cols


# ---------------------------
# Main plotting entry point
# ---------------------------

def plot_uwg_runs_vs_base(
    wxbuild_root: str | Path = "/content/wxbuild",
    city: str | None = None,
    save_dir: str | Path | None = None,
    show: bool = True,
) -> Dict:
    """
    Auto-detect UWG runs under:
      <wxbuild_root>/epwdata/<city>/uwg/City_UWG_RunX.epw

    Plot each run against the base RMY EPW in:
      <wxbuild_root>/epwdata/<city>/rmy/

    Base is always dark grey. UWG runs use a red palette.
    Optionally saves PNGs.
    """
    wxbuild_root = Path(wxbuild_root)

    if city is None:
        city = _detect_city_from_epwdata(wxbuild_root)

    base_path = find_base_rmy_epw(wxbuild_root, city)
    runs = list_uwg_runs(wxbuild_root, city)

    if not runs:
        raise FileNotFoundError(
            f"No UWG runs found in {wxbuild_root / 'epwdata' / city / 'uwg'}\n"
            f"Expected files like: {city}_UWG_Run1.epw, {city}_UWG_Run2.epw, ..."
        )

    base = read_epw_safe(base_path)
    base = base.reset_index(drop=True)

    run_dfs = []
    for p in runs:
        df = read_epw_safe(p).reset_index(drop=True)
        n = min(len(base), len(df))
        run_dfs.append((p, df.iloc[:n].copy()))
    base = base.iloc[: min(len(base), min(len(df) for _, df in run_dfs))].copy()

    # Colors
    COL_BASE = "0.25"  # dark grey
    uwg_cols = _palette_reds(len(run_dfs))

    # Output folder for plots
    if save_dir is None:
        save_dir = wxbuild_root / "epwdata" / city / "uwg"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1) Overlay full year ----
    plt.figure(figsize=(12, 4))
    plt.plot(base["hoy"], base["dbt"], label=f"Base ({base_path.name})", linewidth=2.0, color=COL_BASE)

    for (p, df), c in zip(run_dfs, uwg_cols):
        plt.plot(df["hoy"], df["dbt"], label=p.stem, linewidth=2.2, color=c)

    plt.title(f"{city} — Dry-bulb temperature: Base vs UWG runs")
    plt.xlabel("Hour of year")
    plt.ylabel("Dry-bulb Temperature (°C)")
    ax = plt.gca()
    style_axes(ax)
    plt.legend(frameon=False, ncol=1, fontsize=9)
    fig1_path = save_dir / f"{city}_UWG_overlay_full_year.png"
    plt.tight_layout()
    plt.savefig(fig1_path, dpi=200)
    if show:
        plt.show()
    # ---- 2) ΔT time series per run ----
    stats_rows = []
    for (p, df), c in zip(run_dfs, uwg_cols):
        dt = df["dbt"].to_numpy() - base["dbt"].to_numpy()

        plt.figure(figsize=(12, 4))
        plt.plot(base["hoy"], dt, linewidth=1.8, color=c)
        plt.title(f"{city} — Δ Dry-bulb (UWG − Base) | {p.name}")
        plt.xlabel("Hour of year")
        plt.ylabel("Δ Dry-bulb (°C)")
        ax = plt.gca()
        style_axes(ax)
        plt.tight_layout()
        fig2_path = save_dir / f"{p.stem}_delta_dbt.png"
        plt.savefig(fig2_path, dpi=200)
        if show:
            plt.show()
        stats_rows.append({
            "city": city,
            "run_file": p.name,
            "mean": float(np.nanmean(dt)),
            "min": float(np.nanmin(dt)),
            "max": float(np.nanmax(dt)),
            "p05": float(np.nanpercentile(dt, 5)),
            "p50": float(np.nanpercentile(dt, 50)),
            "p95": float(np.nanpercentile(dt, 95)),
        })

    stats_df = pd.DataFrame(stats_rows)
    stats_csv = save_dir / f"{city}_UWG_delta_stats.csv"
    stats_df.to_csv(stats_csv, index=False)

    # ---- 3) Monthly means + monthly Δ (multi-run) ----
    monthly_base = base.groupby("month")["dbt"].mean()

    plt.figure(figsize=(10, 4))
    plt.plot(monthly_base.index, monthly_base.values, marker="o", linewidth=2.4, color=COL_BASE, label="Base")
    for (p, df), c in zip(run_dfs, uwg_cols):
        monthly_uwg = df.groupby("month")["dbt"].mean()
        plt.plot(monthly_uwg.index, monthly_uwg.values, marker="o", linewidth=2.2, color=c, label=p.stem)
    plt.title(f"{city} — Monthly mean dry-bulb temperature")
    plt.xlabel("Month")
    plt.ylabel("Mean Dry-bulb Temperature (°C)")
    plt.xticks(range(1, 13))
    ax = plt.gca()
    style_axes(ax)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    fig3_path = save_dir / f"{city}_UWG_monthly_means.png"
    plt.savefig(fig3_path, dpi=200)
    if show:
        plt.show()
    # Monthly delta plot per run
    plt.figure(figsize=(10, 4))
    for (p, df), c in zip(run_dfs, uwg_cols):
        monthly_uwg = df.groupby("month")["dbt"].mean()
        monthly_dlt = monthly_uwg - monthly_base
        plt.plot(monthly_dlt.index, monthly_dlt.values, marker="o", linewidth=2.2, color=c, label=p.stem)
    plt.title(f"{city} — Monthly mean Δ dry-bulb (UWG − Base)")
    plt.xlabel("Month")
    plt.ylabel("Mean Δ Dry-bulb (°C)")
    plt.xticks(range(1, 13))
    ax = plt.gca()
    style_axes(ax)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    fig4_path = save_dir / f"{city}_UWG_monthly_delta.png"
    plt.savefig(fig4_path, dpi=200)
    if show:
        plt.show()
    return {
        "city": city,
        "base_epw": str(base_path),
        "uwg_runs": [str(p) for p in runs],
        "plot_files": [str(fig1_path), str(fig3_path), str(fig4_path)],
        "delta_stats_csv": str(stats_csv),
        "plots_dir": str(save_dir),
    }


if __name__ == "__main__":
    # Colab tip: run this file with `%run uwg_plot_runs_SHOWSAVE2.py` (not `!python ...`) to display figures inline.
    res = plot_uwg_runs_vs_base(show=True)
    print("\nSaved outputs to:", res["plots_dir"])
    print("Delta stats CSV:", res["delta_stats_csv"])
