# plots/frmy_picker.py
"""
FRMY Picker + Plotter (Colab widget)

What it does
------------
1) Copies FRMY EPWs from the repo:
      <repo_root>/data/<REPO_LOCATION>/frmys/*.epw
   into the Colab EPW data city folder:
      <epwdata_root>/<EPW_LOCATION>/frmys/*.epw

2) Displays buttons for each FRMY file found (auto-labeled like "SSP126-2050" or "SSP585-2100").

3) When a button is clicked, plots the selected FRMY dry-bulb temperature (hourly) against
   the baseline TMY dry-bulb temperature from:
      <epwdata_root>/<EPW_LOCATION>/tmy/*.epw

Notes
-----
- EPW_LOCATION comes from your Step 1 picker (global `location`), e.g. "Boston__MA__USA".
- REPO_LOCATION may differ (e.g. "Boston_MA_USA"). We auto-resolve it.
- No dependency on epwclass/epw_collection; reads EPW directly from disk.
"""

from __future__ import annotations

import os
import re
import shutil
import difflib
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML


# -------------------------
# Location resolution
# -------------------------

def _list_folders(root: str) -> List[str]:
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
    require_frmys: bool = True,
) -> str:
    """
    Resolve the repo city folder name under <repo_root>/data/<CITY>/ that corresponds to the selected
    epwdata location (e.g., "Phoenix_AZ_USA").

    Repo may store FRMY EPWs under either subfolder name:
        - frmy/
        - frmys/

    Matching order (strict, deterministic, no cross-city fallback):
      1) exact normalized match
      2) substring normalized match
      3) city-token prefix match (first token before underscore)

    If require_frmys=True, the resolved folder must contain frmy(s)/*.epw.
    """
    data_root = os.path.join(repo_root, "data")
    if not os.path.isdir(data_root):
        raise RuntimeError(f"Repo data folder not found: {data_root}")

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", s.lower())

    target = norm(location)

    folders = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

    def has_frmys(folder: str) -> bool:
        for subname in ("frmys", "frmy"):
            fr = os.path.join(data_root, folder, subname)
            if not os.path.isdir(fr):
                continue
            if any(fn.lower().endswith(".epw") for fn in os.listdir(fr)):
                return True
        return False

    if require_frmys:
        folders = [f for f in folders if has_frmys(f)]

    if not folders:
        raise RuntimeError(f"No candidate city folders found under {data_root} (require_frmys={require_frmys}).")

    # 1) exact normalized match
    exact = [f for f in folders if norm(f) == target]
    if exact:
        return sorted(exact, key=lambda x: (len(x), x))[0]

    # 2) substring normalized match
    sub = [f for f in folders if target in norm(f) or norm(f) in target]
    if sub:
        return sorted(sub, key=lambda x: (abs(len(norm(x)) - len(target)), len(x), x))[0]

    # 3) city-token prefix match (e.g. "Phoenix")
    city_token = (location.split("_")[0] if "_" in location else location).lower()
    token_matches = [f for f in folders if f.lower().startswith(city_token + "_") or f.lower() == city_token]
    if token_matches:
        return sorted(token_matches, key=lambda x: (len(x), x))[0]

    raise RuntimeError(
        "Could not match the selected location to a repo city folder with FRMYs.\n"
        f"Selected location: {location}\n"
        f"Looked under: {data_root}\n"
        "Tip: ensure the repo has data/<CITY>/frmy/*.epw (or frmys/*.epw) for this city."
    )
def _read_epw_dbt(epw_path: str) -> np.ndarray:
    """Read EPW dry-bulb temperature (°C) as a numpy array."""
    df = pd.read_csv(
        epw_path,
        skiprows=8,
        header=None,
        usecols=[6],  # DryBulb_C
        names=["DryBulb_C"],
        encoding_errors="ignore",
    )
    return df["DryBulb_C"].astype(float).to_numpy()

def _find_first_epw(folder: str) -> str:
    if not os.path.isdir(folder):
        raise RuntimeError(f"Folder not found: {folder}")
    epws = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".epw")])
    if not epws:
        raise RuntimeError(f"No EPW files found in {folder}")
    return epws[0]


# -------------------------
# FRMY parsing / labels
# -------------------------

# Map "SSP1/2/3/5" in filenames to the commonly used EPW naming
_SSP_MAP = {
    "SSP1": "ssp126",
    "SSP2": "ssp245",
    "SSP3": "ssp370",
    "SSP5": "ssp585",
}

def _parse_frmy_filename(fn: str):
    """
    Expected examples:
      Boston_FRMY_SSP1_2036-2050.epw
      Boston_FRMY_SSP5_2085-2100.epw

    Returns: (scenario_code like 'ssp126', start_year, end_year)
    """
    base = os.path.basename(fn)
    m = re.search(r"_FRMY_(SSP\d)_(\d{4})-(\d{4})\.epw$", base, flags=re.IGNORECASE)
    if not m:
        return None, None, None
    ssp = m.group(1).upper()
    start_y = int(m.group(2))
    end_y = int(m.group(3))
    scen = _SSP_MAP.get(ssp, ssp.lower())
    return scen, start_y, end_y

def _label_for_button(scenario: str, end_year: int) -> str:
    # You requested labels like "SSP126-2050" (uppercase scenario code + end year)
    return f"{scenario.upper()}-{end_year}"



def _ssp_key_from_any_name(name: str) -> Optional[str]:
    """Return normalized SSP key ('ssp126','ssp245','ssp585','ssp370') if detectable from a filename/label."""
    s = os.path.basename(str(name)).lower()

    # common explicit forms in filenames
    if "ssp126" in s or "ssp1-26" in s or "ssp_1-26" in s or "ssp1_26" in s:
        return "ssp126"
    if "ssp245" in s or "ssp2-45" in s or "ssp_2-45" in s or "ssp2_45" in s:
        return "ssp245"
    if "ssp585" in s or "ssp5-85" in s or "ssp_5-85" in s or "ssp5_85" in s:
        return "ssp585"
    if "ssp370" in s or "ssp3-70" in s or "ssp_3-70" in s or "ssp3_70" in s:
        return "ssp370"

    # legacy pattern used in this notebook: ..._FRMY_SSP1_2036-2050.epw
    m = re.search(r"_frmy_(ssp\d)_", s)
    if m:
        return _SSP_MAP.get(m.group(1).upper(), m.group(1))

    # fallback: match SSP<digit> anywhere
    m = re.search(r"ssp(\d)\b", s)
    if m:
        return _SSP_MAP.get(f"SSP{m.group(1)}", f"ssp{m.group(1)}")

    return None

def _ssp_heading(ssp_key: str) -> str:
    return {
        "ssp126": "SSP 1-26",
        "ssp245": "SSP 2-45",
        "ssp585": "SSP 5-85",
        "ssp370": "SSP 3-70",
    }.get(ssp_key, ssp_key.upper())

# -------------------------
# Copy FRMYs into epwdata city
# -------------------------

def copy_frmys_to_epwdata_city(
    *,
    location: Optional[str] = None,
    repo_root: str = "/content/wxbuild",
    epwdata_root: str = "/content/wxbuild/epwdata",
    overwrite: bool = True,
) -> str:
    """
    Sync FRMY EPWs from the repo city folder into the epwdata city folder.

      Source: <repo_root>/data/<CITY>/(frmy|frmys)/*.epw
      Dest:   <epwdata_root>/<location>/frmys/*.epw

    This ensures the FRMY picker uses the currently selected city's FRMYs from the GitHub checkout.
    """
    if location is None:
        location = globals().get("location")
    if not location:
        raise RuntimeError("No EPW location provided and no global 'location' found. Run Step 1 first.")

    city_dir = os.path.join(epwdata_root, location)
    if not os.path.isdir(city_dir):
        raise RuntimeError(f"EPW city folder not found under epwdata: {city_dir}")

    dest_frmys = os.path.join(city_dir, "frmys")
    os.makedirs(dest_frmys, exist_ok=True)

    if overwrite:
        for f in os.listdir(dest_frmys):
            if f.lower().endswith(".epw"):
                os.remove(os.path.join(dest_frmys, f))

    repo_loc = resolve_repo_location_folder(location, repo_root=repo_root, require_frmys=True)

    src_frmys = os.path.join(repo_root, "data", repo_loc, "frmys")
    if not os.path.isdir(src_frmys):
        src_frmys = os.path.join(repo_root, "data", repo_loc, "frmy")
    if not os.path.isdir(src_frmys):
        raise RuntimeError(f"FRMY folder not found in repo for '{repo_loc}': tried frmys/ and frmy/")

    copied = 0
    for fn in os.listdir(src_frmys):
        if fn.lower().endswith(".epw"):
            shutil.copy2(os.path.join(src_frmys, fn), os.path.join(dest_frmys, fn))
            copied += 1

    if copied == 0:
        raise RuntimeError(f"No FRMY EPWs found to copy in {src_frmys}")

    return dest_frmys
def show_frmy_picker(
    *,
    location: Optional[str] = None,
    repo_root: str = "/content/wxbuild",
    epwdata_root: str = "/content/wxbuild/epwdata",
    figsize: Tuple[int, int] = (14, 4),
    title_prefix: str = "FRMY vs RMY",
    y_label: str = "Dry-Bulb Temperature (°C)",
):
    """
    FRMY picker UI.

    - FRMY EPWs are synced from the GitHub checkout:
        <repo_root>/data/<CITY>/(frmy|frmys)/*.epw
      into:
        <epwdata_root>/<location>/frmys/*.epw

    - BASE EPW is ALWAYS the first EPW found under:
        <epwdata_root>/<location>/rmy/*.epw

    - Clicking a scenario button:
        plots FRMY vs BASE (RMY)

    - "Select Scenario EPW" copies the currently selected FRMY EPW into:
        <epwdata_root>/<location>/Chosen_scenario/
      (folder is created if missing)
    """
    if location is None:
        location = globals().get("location")
    if not location:
        raise RuntimeError("No location provided and no global 'location' set. Run Step 1 first.")

    # Make sure FRMY EPWs exist under epwdata/<location>/frmys
    frmy_dir = copy_frmys_to_epwdata_city(
        location=location,
        repo_root=repo_root,
        epwdata_root=epwdata_root,
        overwrite=True,
    )

    # Baseline RMY EPW (from rmy folder)
    rmy_dir = os.path.join(epwdata_root, location, "rmy")
    rmy_path = _find_first_epw(rmy_dir)
    rmy_dbt = _read_epw_dbt(rmy_path)

    # FRMY files
    frmy_files = sorted([os.path.join(frmy_dir, f) for f in os.listdir(frmy_dir) if f.lower().endswith(".epw")])
    if not frmy_files:
        raise RuntimeError(f"No FRMY EPWs found in {frmy_dir}")

    # Build (label, path, ssp_key) entries
    entries = []
    for p in frmy_files:
        scen, start_y, end_y = _parse_frmy_filename(p)

        # If parsing fails (common when filenames look like City_Country_SSP585_2100_FRMY.epw),
        # fall back to a readable label (basename without extension).
        if scen is None:
            label = os.path.splitext(os.path.basename(p))[0]
        else:
            label = _label_for_button(scen, end_y)

        ssp_key = _ssp_key_from_any_name(p) or scen
        entries.append((label, p, ssp_key))

    # Sort by scenario then end-year if parseable

    def sort_key(item):
        lab, p = item
        scen, start_y, end_y = _parse_frmy_filename(p)
        return (scen or "zzz", end_y or 9999, lab)
    entries = sorted(entries, key=sort_key)

    out = widgets.Output()

    # Track current selection
    state = {"label": None, "path": None}

    select_btn = widgets.Button(
        description="Select Scenario EPW",
        button_style="success",
        layout=widgets.Layout(width="220px", margin="8px 0 0 0"),
        tooltip="Copy the currently selected FRMY EPW into a Chosen_scenario folder under this city."
    )
    select_btn.disabled = True

    def _persist_selection(_):
        if not state["path"]:
            return
        chosen_dir = os.path.join(epwdata_root, location, "Chosen_scenario")
        os.makedirs(chosen_dir, exist_ok=True)

        src_epw = state["path"]
        dst_epw = os.path.join(chosen_dir, os.path.basename(src_epw))
        shutil.copy2(src_epw, dst_epw)

        # Small trace file for reproducibility
        meta_path = os.path.join(chosen_dir, "chosen_scenario.txt")
        with open(meta_path, "w") as f:
            f.write(f"location: {location}\n")
            f.write(f"label: {state['label']}\n")
            f.write(f"source_epw: {src_epw}\n")
            f.write(f"copied_to: {dst_epw}\n")

        with out:
            print(f"\n✅ Selected scenario saved to: {dst_epw}")
            print(f"   Metadata: {meta_path}\n")

    select_btn.on_click(_persist_selection)

    def make_onclick(label: str, path: str):
        def _onclick(_):
            state["label"] = label
            state["path"] = path
            select_btn.disabled = False

            with out:
                clear_output(wait=True)
                frmy_dbt = _read_epw_dbt(path)

                plt.figure(figsize=figsize)
                ax = plt.gca()
                ax.plot(rmy_dbt, label="Original RMY", linewidth=2, color="#3a3a3a")
                ax.plot(frmy_dbt, label=label, linewidth=2, color="#b82e45")

                ax.set_xlabel("Hour of Year")
                ax.set_ylabel(y_label)
                ax.set_title(f"{title_prefix}: {label}")
                ax.grid(True, alpha=0.25)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.legend(frameon=False)
                plt.show()

                # Button under plot
                display(select_btn)
        return _onclick

    # Prevent filename truncation in buttons and widen to show full names
    BUTTON_MIN_WIDTH = "360px"
    GRID_COLS = 2

    display(HTML(
        "<style>"
        ".frmy-scenario-btn button{white-space:nowrap !important; overflow:visible !important; text-overflow:clip !important;}"
        "</style>"
    ))

    buttons = []
    def _set_active(active_btn):
        for bb in buttons:
            bb.button_style = ""
            bb.style.button_color = None
            try:
                bb.remove_class("wxbuild-active")
            except Exception:
                pass

        # Active button: slightly darker than non-selected
        active_btn.button_style = ""
        active_btn.style.button_color = "#C7C7C7"  # light/medium grey
        try:
            active_btn.add_class("wxbuild-active")
        except Exception:
            pass

    # Group FRMY buttons by SSP and add subheadings
    groups = {"ssp126": [], "ssp245": [], "ssp585": []}
    other_entries = []

    for label, path, ssp_key in entries:
        if ssp_key in groups:
            groups[ssp_key].append((label, path))
        else:
            other_entries.append((label, path))

    sections = []

    def _build_group_buttons(group_items):
        group_buttons = []
        for label, path in group_items:
            b = widgets.Button(
                description=label,
                layout=widgets.Layout(min_width=BUTTON_MIN_WIDTH, width="auto")
            )
            try:
                b.add_class("frmy-scenario-btn")
            except Exception:
                pass

            def _wrap_click(btn, lab=label, pth=path):
                def _(_ev):
                    _set_active(btn)
                    make_onclick(lab, pth)(btn)
                return _
            b.on_click(_wrap_click(b))

            buttons.append(b)
            group_buttons.append(b)

        return widgets.GridBox(
            group_buttons,
            layout=widgets.Layout(
                grid_template_columns=f"repeat({GRID_COLS}, {BUTTON_MIN_WIDTH})",
                grid_gap="8px 10px"
            )
        )

    for key in ("ssp126", "ssp245", "ssp585"):
        if groups[key]:
            heading = widgets.HTML(f"<b>{_ssp_heading(key)}</b>")
            grid = _build_group_buttons(groups[key])
            sections.append(widgets.VBox([heading, grid]))

    # If there are any non-matching files, append them (no extra heading requested)
    if other_entries:
        sections.append(_build_group_buttons(other_entries))

    # White text when active
    display(HTML("<style>\n.wxbuild-active button{color:white !important;}\n</style>"))
    display(widgets.VBox([header] + sections + [out]))

    # auto-plot first option
    buttons[0].click()

