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
    title_prefix: str = "FRMY vs Base",
    y_label: str = "Dry-Bulb Temperature (°C)",
):
    """
    Display buttons for FRMY files and plot selected FRMY vs a user-chosen BASE EPW.

    - FRMYs are synced from the GitHub checkout:
        <repo_root>/data/<CITY>/(frmy|frmys)/*.epw
      into:
        <epwdata_root>/<location>/frmys/*.epw

    - BASE EPW is loaded from:
        <epwdata_root>/<location>/Chosen_scenario/*.epw
      If multiple EPWs exist there, a dropdown appears.

    - The selected scenario button is highlighted (light grey with white text).
    - A "Select Scenario EPW" button copies the currently selected FRMY EPW into Chosen_scenario/.
    """
    if location is None:
        location = globals().get("location")
    if not location:
        raise RuntimeError("No location provided and no global 'location' set. Run Step 1 first.")

    # Sync FRMYs for this city from repo -> epwdata/<city>/frmys
    frmy_dir = copy_frmys_to_epwdata_city(location=location, repo_root=repo_root, epwdata_root=epwdata_root, overwrite=True)

    # ---- Base EPW from Chosen_scenario ----
    chosen_dir = os.path.join(epwdata_root, location, "Chosen_scenario")
    if not os.path.isdir(chosen_dir):
        raise RuntimeError(
            f"No 'Chosen_scenario' folder found for {location}.\n"
            f"Expected: {chosen_dir}\n"
            f"Create it by selecting a scenario (button below plot) or place a base EPW there."
        )

    def _list_base_eps():
        return sorted([f for f in os.listdir(chosen_dir) if f.lower().endswith(".epw")])

    base_eps = _list_base_eps()
    if not base_eps:
        raise RuntimeError(
            f"No EPW files found in: {chosen_dir}\n"
            f"Please copy a base EPW into this folder first (or use 'Select Scenario EPW' after choosing a scenario)."
        )

    base_dd = widgets.Dropdown(
        options=base_eps,
        value=base_eps[0],
        description="Base EPW",
        layout=widgets.Layout(width="520px"),
        style={"description_width": "100px"},
    )
    if len(base_eps) == 1:
        base_dd.layout.display = "none"

    def _base_path() -> str:
        return os.path.join(chosen_dir, base_dd.value)

    # ---- FRMY files (in epwdata now) ----
    frmy_files = sorted([os.path.join(frmy_dir, f) for f in os.listdir(frmy_dir) if f.lower().endswith(".epw")])
    if not frmy_files:
        raise RuntimeError(f"No FRMY EPWs found in {frmy_dir}")

    entries = []
    for p in frmy_files:
        scen, start_y, end_y = _parse_frmy_filename(p)
        if scen is None:
            label = os.path.splitext(os.path.basename(p))[0]
        else:
            label = _label_for_button(scen, end_y)
        entries.append((label, p))

    def sort_key(item):
        lab, p = item
        scen, start_y, end_y = _parse_frmy_filename(p)
        return (scen or "zzz", end_y or 9999, lab)
    entries = sorted(entries, key=sort_key)

    out = widgets.Output()
    state = {"label": None, "path": None}

    select_btn = widgets.Button(
        description="Select Scenario EPW",
        button_style="success",
        layout=widgets.Layout(width="220px", margin="8px 0 0 0"),
        tooltip="Copy the currently selected FRMY EPW into Chosen_scenario/ for this city."
    )
    select_btn.disabled = True

    def _persist_selection(_):
        if not state["path"]:
            return
        os.makedirs(chosen_dir, exist_ok=True)

        src_epw = state["path"]
        dst_epw = os.path.join(chosen_dir, os.path.basename(src_epw))
        shutil.copy2(src_epw, dst_epw)

        meta_path = os.path.join(chosen_dir, "chosen_scenario.txt")
        with open(meta_path, "w") as f:
            f.write(f"location: {location}\n")
            f.write(f"label: {state['label']}\n")
            f.write(f"source_epw: {src_epw}\n")
            f.write(f"copied_to: {dst_epw}\n")

        # refresh dropdown options
        opts = _list_base_eps()
        base_dd.options = opts
        if base_dd.value not in opts and opts:
            base_dd.value = opts[0]
        if len(opts) == 1:
            base_dd.layout.display = "none"
        else:
            base_dd.layout.display = None

        with out:
            print(f"\n✅ Selected scenario saved to: {dst_epw}")
            print(f"   Metadata: {meta_path}\n")

    select_btn.on_click(_persist_selection)

    def _plot_current():
        if not state["path"]:
            return
        with out:
            clear_output(wait=True)

            base_path = _base_path()
            base_dbt = _read_epw_dbt(base_path)
            frmy_dbt = _read_epw_dbt(state["path"])

            plt.figure(figsize=figsize)
            ax = plt.gca()
            ax.plot(base_dbt, label=f"Base: {os.path.basename(base_path)}", linewidth=2, color="#3a3a3a")
            ax.plot(frmy_dbt, label=state["label"], linewidth=2, color="#b82e45")

            ax.set_xlabel("Hour of Year")
            ax.set_ylabel(y_label)
            ax.set_title(f"{title_prefix}: {state['label']}")
            ax.grid(True, alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False)
            plt.show()

            display(select_btn)

    # Active-button styling
    buttons = []

    def _set_active(active_btn):
        for bb in buttons:
            bb.button_style = ""
            bb.style.button_color = None
            try:
                bb.remove_class("wxbuild-active")
            except Exception:
                pass
        active_btn.button_style = ""
        active_btn.style.button_color = "#B0B0B0"  # light grey
        try:
            active_btn.add_class("wxbuild-active")
        except Exception:
            pass

    def make_onclick(label: str, path: str):
        def _onclick(_):
            state["label"] = label
            state["path"] = path
            select_btn.disabled = False
            _plot_current()
        return _onclick

    for label, path in entries:
        b = widgets.Button(description=label, layout=widgets.Layout(width="180px"))
        def _wrap_click(btn, lab=label, pth=path):
            def _(_ev):
                _set_active(btn)
                make_onclick(lab, pth)(btn)
            return _
        b.on_click(_wrap_click(b))
        buttons.append(b)

    header = widgets.HTML(
        value=f"<b>FRMY files detected:</b> {len(entries)} &nbsp; | &nbsp; "
              f"<b>City:</b> {location} &nbsp; | &nbsp; "
              f"<b>Base folder:</b> Chosen_scenario"
    )

    grid = widgets.GridBox(
        buttons,
        layout=widgets.Layout(
            grid_template_columns="repeat(4, 180px)",
            grid_gap="8px 10px"
        )
    )

    # White text on active button
    display(HTML("<style>\n.wxbuild-active button{color:white !important;}\n</style>"))

    display(widgets.VBox([header, base_dd, grid, out]))

    # Replot when base changes
    def _on_base_change(_):
        _plot_current()
    base_dd.observe(_on_base_change, names="value")

    # auto-plot first option
    buttons[0].click()
