# plots/extremes_timeline.py
"""
Extreme Event Explorer — timeline plots for RMY and FRMY event CSVs.

Updates (per your request)
--------------------------
1) Copy the repo extremes folder into the *city folder under epwdata* in Colab:

   /content/wxbuild/epwdata/<EPW_LOCATION>/extremes

   (EPW_LOCATION typically comes from your Step 1 picker, e.g. Boston__MA__USA)

2) FRMY plot "5-year increments" now means:
   2025, 2030, 2035, ... , 2100  (i.e., start_year=2025, step=5)

Folder conventions
------------------
Repo source:
  /content/wxbuild/data/<REPO_LOCATION>/extremes/*.csv
Destination in Colab:
  /content/wxbuild/epwdata/<EPW_LOCATION>/extremes/*.csv

Note: REPO_LOCATION and EPW_LOCATION may differ (single vs double underscores).
This module resolves REPO_LOCATION automatically.
"""

from __future__ import annotations

import os
import re
import shutil
import difflib
from typing import Optional, Tuple, List

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

# -------------------------
# SSP detection from chosen scenario
# -------------------------

def _detect_ssp_from_chosen_scenario_epw(
    *,
    location: str,
    epwdata_root: str = "/content/wxbuild/epwdata",
) -> Optional[str]:
    """
    Detect SSP code (e.g., 'SSP126', 'SSP245', 'SSP585') from an EPW file stored in:
        <epwdata_root>/<location>/Chosen_scenario/*.epw

    Preference order:
      1) chosen_scenario.txt (if exists) -> 'copied_to:' or 'source_epw:' line
      2) most recently modified .epw in Chosen_scenario
    Returns 'SSP###' or None if not detectable.
    """
    chosen_dir = os.path.join(epwdata_root, location, "Chosen_scenario")
    if not os.path.isdir(chosen_dir):
        return None

    meta = os.path.join(chosen_dir, "chosen_scenario.txt")
    candidate_paths = []

    if os.path.isfile(meta):
        try:
            txt = Path(meta).read_text()
            m = re.search(r"copied_to:\s*(.+\.epw)", txt, flags=re.IGNORECASE)
            if not m:
                m = re.search(r"source_epw:\s*(.+\.epw)", txt, flags=re.IGNORECASE)
            if m:
                candidate_paths.append(m.group(1).strip())
        except Exception:
            pass

    epws = [os.path.join(chosen_dir, f) for f in os.listdir(chosen_dir) if f.lower().endswith(".epw")]
    epws = sorted(epws, key=lambda p: os.path.getmtime(p), reverse=True)
    candidate_paths.extend(epws)

    for p in candidate_paths:
        fn = os.path.basename(p)
        m = re.search(r"(SSP\d{3})", fn, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

        m2 = re.search(r"ssp\s*([1-5])\s*[-_]?\s*([0-9]\.?[0-9])", fn, flags=re.IGNORECASE)
        if m2:
            ssp_num = int(m2.group(1))
            forcing = m2.group(2).replace(".", "")
            return f"SSP{ssp_num}{forcing}".upper()

    return None


# -------------------------
# Location resolution
# -------------------------

def _list_location_folders(root: str) -> List[str]:
    return sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    )


def _normalize_loc_token(s: str) -> str:
    return re.sub(r"[_\-\s]+", "", s.strip().lower())


def resolve_repo_location_folder(
    location: str,
    *,
    repo_root: str = "/content/wxbuild",
    require_extremes: bool = True,
) -> str:
    """
    Resolve a notebook `location` (EPW location string) to an actual folder under <repo_root>/data/.

    Tries:
    - exact match
    - swap "__" <-> "_"
    - collapse multiple underscores
    - normalized-token match
    - fuzzy match
    - fallback to first folder with extremes
    """
    data_root = os.path.join(repo_root, "data")
    if not os.path.isdir(data_root):
        raise RuntimeError(f"Could not find repo data folder: {data_root}")

    candidates = _list_location_folders(data_root)
    if not candidates:
        raise RuntimeError(f"No location folders found under {data_root}")

    def has_extremes(loc_folder: str) -> bool:
        ex_dir = os.path.join(data_root, loc_folder, "extremes")
        return os.path.isdir(ex_dir) and any(f.lower().endswith(".csv") for f in os.listdir(ex_dir))

    variants = []
    variants.append(location)
    variants.append(location.replace("__", "_"))
    variants.append(location.replace("_", "__"))
    variants.append(re.sub(r"_+", "_", location))
    variants.append(re.sub(r"_+", "__", location))
    # de-dupe
    seen = set()
    variants = [v for v in variants if not (v in seen or seen.add(v))]

    for v in variants:
        if v in candidates and (not require_extremes or has_extremes(v)):
            return v

    norm_map = {_normalize_loc_token(c): c for c in candidates}
    for v in variants:
        nv = _normalize_loc_token(v)
        if nv in norm_map:
            c = norm_map[nv]
            if not require_extremes or has_extremes(c):
                return c

    target = _normalize_loc_token(location)
    close = difflib.get_close_matches(target, list(norm_map.keys()), n=1, cutoff=0.6)
    if close:
        c = norm_map[close[0]]
        if not require_extremes or has_extremes(c):
            return c

    if require_extremes:
        for c in candidates:
            if has_extremes(c):
                return c

    return candidates[0]


def _auto_detect_location_from_repo(repo_root: str = "/content/wxbuild") -> str:
    data_root = os.path.join(repo_root, "data")
    if not os.path.isdir(data_root):
        raise RuntimeError(f"Could not find repo data folder: {data_root}")
    candidates = _list_location_folders(data_root)
    if not candidates:
        raise RuntimeError(f"No location folders found under {data_root}")
    for loc in candidates:
        ex_dir = os.path.join(data_root, loc, "extremes")
        if os.path.isdir(ex_dir) and any(f.lower().endswith(".csv") for f in os.listdir(ex_dir)):
            return loc
    return candidates[0]


# -------------------------
# Copy extremes into epwdata/<location>/extremes
# -------------------------

def copy_extremes_to_epwdata_city(
    *,
    location: Optional[str] = None,
    repo_root: str = "/content/wxbuild",
    epwdata_root: str = "/content/wxbuild/epwdata",
    overwrite: bool = True,
) -> str:
    """
    Copy: <repo_root>/data/<repo_location>/extremes/*.csv
      to: <epwdata_root>/<location>/extremes/*.csv

    Returns destination directory.
    """
    if location is None:
        location = globals().get("location")
    if not location:
        raise RuntimeError("No EPW location provided and no global 'location' found. Run Step 1 first.")

    repo_loc = resolve_repo_location_folder(location, repo_root=repo_root, require_extremes=True)
    src_dir = os.path.join(repo_root, "data", repo_loc, "extremes")
    if not os.path.isdir(src_dir):
        raise RuntimeError(
            f"Repo extremes not found. EPW location='{location}' resolved to repo folder='{repo_loc}', "
            f"missing: {src_dir}"
        )

    city_dir = os.path.join(epwdata_root, location)
    if not os.path.isdir(city_dir):
        raise RuntimeError(f"EPW city folder not found (expected from Step 1 downloads): {city_dir}")

    dest_dir = os.path.join(city_dir, "extremes")
    if os.path.isdir(dest_dir) and overwrite:
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    copied = 0
    for fn in os.listdir(src_dir):
        if fn.lower().endswith(".csv"):
            shutil.copy2(os.path.join(src_dir, fn), os.path.join(dest_dir, fn))
            copied += 1
    if copied == 0:
        raise RuntimeError(f"No CSVs copied from {src_dir}.")
    return dest_dir


def _get_extremes_csv_paths(
    *,
    location: Optional[str],
    repo_root: str,
    epwdata_root: str,
    which: str,  # "rmy" or "frmy"
    ssp: Optional[str] = None,  # e.g. "SSP585"
) -> Tuple[str, str, str]:
    """
    Ensure extremes copied to epwdata and return (dest_dir, heatwave_csv, coldspell_csv).

    For FRMY, if `ssp` is provided (e.g. "SSP585"), will look for:
      frmy_SSP585_heatwave_events.csv
      frmy_SSP585_coldspells_events.csv
    Otherwise falls back to:
      frmy_heatwave_events.csv
      frmy_coldspells_events.csv
    """
    dest_dir = copy_extremes_to_epwdata_city(
        location=location,
        repo_root=repo_root,
        epwdata_root=epwdata_root,
        overwrite=False,
    )

    if which not in ("rmy", "frmy"):
        raise ValueError(f"which must be 'rmy' or 'frmy', got: {which}")

    if which == "frmy" and ssp:
        ssp_norm = ssp.upper()
        heatwave_csv = os.path.join(dest_dir, f"frmy_{ssp_norm}_heatwave_events.csv")
        coldspell_csv = os.path.join(dest_dir, f"frmy_{ssp_norm}_coldspells_events.csv")

        if not os.path.isfile(heatwave_csv) or not os.path.isfile(coldspell_csv):
            heatwave_csv = os.path.join(dest_dir, "frmy_heatwave_events.csv")
            coldspell_csv = os.path.join(dest_dir, "frmy_coldspells_events.csv")
    else:
        heatwave_csv = os.path.join(dest_dir, f"{which}_heatwave_events.csv")
        coldspell_csv = os.path.join(dest_dir, f"{which}_coldspells_events.csv")

    if not os.path.isfile(heatwave_csv):
        raise RuntimeError(f"Missing file: {heatwave_csv}")
    if not os.path.isfile(coldspell_csv):
        raise RuntimeError(f"Missing file: {coldspell_csv}")

    return dest_dir, heatwave_csv, coldspell_csv
def create_custom_colormap():
    heatwave_colors = ['#F7DDE1', '#EBAFB9', '#DC7284', '#CF3952', '#B22C42']
    coldspell_colors = ['#b3e7f2', '#80d2e6', '#4dbeda', '#1aaacb', '#0086b3']
    heatwave_cmap = mcolors.LinearSegmentedColormap.from_list('heatwave_cmap', heatwave_colors)
    coldspell_cmap = mcolors.LinearSegmentedColormap.from_list('coldspell_cmap', coldspell_colors)
    return heatwave_cmap, coldspell_cmap


def visualize_extreme_events(
    heatwave_csv: str,
    coldspell_csv: str,
    *,
    title: str = "Extreme Event Explorer: Heatwaves and Cold Spells Timeline",
    keep_years: Optional[List[int]] = None,
    circle_size_factor: float = 58.0,
    figsize=(20, 15),
):
    heatwave_df = pd.read_csv(heatwave_csv)
    coldspell_df = pd.read_csv(coldspell_csv)

    heatwave_df['begin_date'] = pd.to_datetime(heatwave_df['begin_date'], format='%d/%m/%Y', errors='coerce')
    coldspell_df['begin_date'] = pd.to_datetime(coldspell_df['begin_date'], format='%d/%m/%Y', errors='coerce')
    heatwave_df = heatwave_df.dropna(subset=["begin_date"]).copy()
    coldspell_df = coldspell_df.dropna(subset=["begin_date"]).copy()

    for df in (heatwave_df, coldspell_df):
        df["year"] = df["begin_date"].dt.year
        df["month"] = df["begin_date"].dt.month
        df["day"] = df["begin_date"].dt.day

    if keep_years is not None:
        keep = set(int(y) for y in keep_years)
        heatwave_df = heatwave_df[heatwave_df["year"].isin(keep)].copy()
        coldspell_df = coldspell_df[coldspell_df["year"].isin(keep)].copy()

    heatwave_cmap, coldspell_cmap = create_custom_colormap()
    fig, ax = plt.subplots(figsize=figsize)

    unique_years = sorted(set(heatwave_df['year'].unique()) | set(coldspell_df['year'].unique()))
    if not unique_years:
        raise RuntimeError("No events to plot after filtering (check inputs/years).")

    for i, _year in enumerate(unique_years):
        ax.hlines(y=i, xmin=1, xmax=13, color='gray', alpha=0.5, linestyle='--')

    for month in range(1, 14):
        ax.axvline(x=month, color='lightgray', linestyle='--', linewidth=0.7)

    heatwave_norm = mcolors.Normalize(vmin=85, vmax=110)
    if "avg_heat_index" not in heatwave_df.columns:
        raise RuntimeError("Heatwave CSV must include column 'avg_heat_index'.")
    heatwave_df['clipped_index'] = pd.to_numeric(heatwave_df['avg_heat_index'], errors='coerce').clip(85, 110)

    if 'avg_wind_chill' in coldspell_df.columns:
        wind_col = 'avg_wind_chill'
        cold_vals = pd.to_numeric(coldspell_df[wind_col], errors='coerce')
        coldspell_norm = mcolors.Normalize(vmin=float(cold_vals.min()), vmax=float(cold_vals.max()))
        cold_label = 'Cold Spell Wind Chill (°C)'
    elif 'avg_wind_speed' in coldspell_df.columns:
        wind_col = 'avg_wind_speed'
        cold_vals = pd.to_numeric(coldspell_df[wind_col], errors='coerce')
        coldspell_norm = mcolors.Normalize(vmin=float(cold_vals.min()), vmax=float(cold_vals.max()))
        cold_label = 'Cold Spell Wind Speed (m/s)'
    else:
        raise RuntimeError("Coldspell CSV must include 'avg_wind_chill' or 'avg_wind_speed'.")

    for _, row in heatwave_df.iterrows():
        year_index = unique_years.index(int(row['year']))
        month_day = float(row['month']) + float(row['day']) / 30.0
        dur = float(row.get("duration", 1))
        ax.scatter(
            month_day, year_index,
            s=dur * circle_size_factor,
            color=heatwave_cmap(heatwave_norm(float(row['clipped_index']))),
            alpha=0.7, edgecolors='black', linewidth=0.5
        )

    for _, row in coldspell_df.iterrows():
        year_index = unique_years.index(int(row['year']))
        month_day = float(row['month']) + float(row['day']) / 30.0
        dur = float(row.get("duration", 1))
        val = float(pd.to_numeric(row[wind_col], errors='coerce'))
        ax.scatter(
            month_day, year_index,
            s=dur * circle_size_factor,
            color=coldspell_cmap(coldspell_norm(val)),
            alpha=0.7, edgecolors='black', linewidth=0.5
        )

    heatwave_sm = plt.cm.ScalarMappable(cmap=heatwave_cmap, norm=heatwave_norm)
    coldspell_sm = plt.cm.ScalarMappable(cmap=coldspell_cmap, norm=coldspell_norm)
    heatwave_sm.set_array([])
    coldspell_sm.set_array([])

    cbar_heatwave = fig.colorbar(heatwave_sm, ax=ax, orientation='horizontal', pad=0.0001, shrink=0.8, aspect=40)
    cbar_coldspell = fig.colorbar(coldspell_sm, ax=ax, orientation='horizontal', pad=0.08, shrink=0.8, aspect=40)
    cbar_heatwave.set_label('Heatwave Average Heat Index (°C)', fontsize=14)
    cbar_coldspell.set_label(cold_label, fontsize=14)

    sizes = [1, 3, 5, 7]
    labels = [f'{size} days' for size in sizes]
    handles = [
        ax.scatter([], [], s=size * circle_size_factor, color='gray', alpha=0.5, edgecolors='black', label=label)
        for size, label in zip(sizes, labels)
    ]
    legend = ax.legend(
        handles=handles,
        scatterpoints=1,
        frameon=False,
        labelspacing=1,
        title='Event Duration',
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        fontsize=14
    )
    legend.get_title().set_fontsize(16)

    ax.set_xlim(0.8, 13.2)
    ax.set_ylim(-1, len(unique_years))
    ax.set_yticks(ticks=range(len(unique_years)))
    ax.set_yticklabels(unique_years, fontsize=16)
    ax.set_xticks(ticks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=16)

    ax.set_xlabel('Month', fontsize=16)
    ax.set_ylabel('Year', fontsize=16)
    ax.set_title(title)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()


# -------------------------
# Public API
# -------------------------

def plot_rmy_timeline(
    *,
    location: Optional[str] = None,
    repo_root: str = "/content/wxbuild",
    epwdata_root: str = "/content/wxbuild/epwdata",
    title: Optional[str] = None,
):
    dest_dir, hw, cs = _get_extremes_csv_paths(
        location=location,
        repo_root=repo_root,
        epwdata_root=epwdata_root,
        which="rmy",
    )
    loc = location or globals().get("location") or "Selected city"
    visualize_extreme_events(hw, cs, title=title or f"{loc} — RMY extremes (observed/historic)")


def plot_frmy_timeline_5yr_from_2025(
    *,
    location: Optional[str] = None,
    repo_root: str = "/content/wxbuild",
    epwdata_root: str = "/content/wxbuild/epwdata",
    start_year: int = 2025,
    end_year: int = 2100,
    step: int = 5,
    title: Optional[str] = None,
):
    """
    Plot FRMY extremes only for years: 2025, 2030, ..., 2100 (configurable),
    and automatically detect the SSP from the EPW saved in:

        epwdata/<location>/Chosen_scenario/*.epw

    Expected repo CSV naming:
        frmy_SSP126_heatwave_events.csv, frmy_SSP126_coldspells_events.csv, etc.

    If SSP-specific files are not found, falls back to generic frmy_* files.
    """
    loc = location or globals().get("location")
    if not loc:
        raise RuntimeError("No location provided and no global 'location' found. Run Step 1 first.")

    ssp = _detect_ssp_from_chosen_scenario_epw(location=loc, epwdata_root=epwdata_root)

    dest_dir, hw, cs = _get_extremes_csv_paths(
        location=loc,
        repo_root=repo_root,
        epwdata_root=epwdata_root,
        which="frmy",
        ssp=ssp,
    )

    keep_years = list(range(int(start_year), int(end_year) + 1, int(step)))

    ssp_txt = f" ({ssp})" if ssp else ""
    visualize_extreme_events(
        hw, cs,
        title=title or f"{loc} — FRMY extremes{ssp_txt} ({start_year}-{end_year}, every {step} years)",
        keep_years=keep_years,
    )
