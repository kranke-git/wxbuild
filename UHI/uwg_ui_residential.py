from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv
def _next_run_index(out_dir: str | Path, city: str) -> int:
    """Return next integer x for City_UWG_Runx.epw based on existing files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(city)}_UWG_Run(\d+)\.epw$", re.IGNORECASE)
    max_i = 0
    for p in out_dir.glob(f"{city}_UWG_Run*.epw"):
        m = pattern.match(p.name)
        if m:
            try:
                max_i = max(max_i, int(m.group(1)))
            except Exception:
                pass
    return max_i + 1


def _append_run_log(out_dir: str | Path, city: str, row: dict):
    """Append a run-parameter row to <out_dir>/<city>_UWG_runs.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{city}_UWG_runs.csv"
    # Ensure stable column order
    fieldnames = list(row.keys())
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


import ipywidgets as widgets
from IPython.display import display, clear_output

def _ensure_uwg_legacy_pickle_shims():
    """Create shim modules expected by older UWG DOERefBuildings pickles (namespace-safe)."""
    import importlib
    import importlib.util
    import pkgutil
    import re as _re
    from pathlib import Path as _Path
    import uwg as _uwg

    # UWG may be a namespace package in editable installs; __file__ can be None.
    uwg_root = _Path(list(_uwg.__path__)[0])  # e.g., /content/uwg
    pkg_dir_candidates = []
    # Prefer the directory that actually contains the uwg python modules
    if (uwg_root / "uwg").exists():
        pkg_dir_candidates.append(uwg_root / "uwg")
    pkg_dir_candidates.append(uwg_root)

    def _find_target_module(legacy_name: str) -> str | None:
        # If already importable, nothing to do.
        try:
            importlib.import_module(legacy_name)
            return legacy_name
        except Exception:
            pass

        # Heuristic: if legacy is uwg.X, try uwg.uwg.X
        if legacy_name.startswith("uwg."):
            tail = legacy_name.split(".", 1)[1]
            for guess in [f"uwg.uwg.{tail}", f"uwg.{tail}"]:
                try:
                    importlib.import_module(guess)
                    return guess
                except Exception:
                    pass

        return None

    def _scan_for_class_or_file(tail: str) -> str | None:
        # tail like "BEMDef" or "building"; also try mapped alternatives
        alt_tails = [tail]
        if tail.lower() in LEGACY_TAIL_MAP:
            alt_tails = LEGACY_TAIL_MAP[tail.lower()] + alt_tails

        for pkg_dir in pkg_dir_candidates:
            # 1) scan for a python file matching any tail (case-insensitive)
            for t in alt_tails:
                for py in pkg_dir.rglob("*.py"):
                    if py.stem.lower() == t.lower():
                        rel = py.relative_to(pkg_dir).with_suffix("")
                        mod = "uwg." + ".".join(rel.parts)
                        try:
                            importlib.import_module(mod)
                            return mod
                        except Exception:
                            continue

            # 2) scan for class definition
            for t in alt_tails:
                class_pat = _re.compile(rf"^\s*class\s+{_re.escape(t)}\b", flags=_re.MULTILINE)
                for py in pkg_dir.rglob("*.py"):
                    try:
                        file_txt = py.read_text(errors="ignore")
                    except Exception:
                        continue
                    if class_pat.search(file_txt):
                        rel = py.relative_to(pkg_dir).with_suffix("")
                        mod = "uwg." + ".".join(rel.parts)
                        try:
                            importlib.import_module(mod)
                            return mod
                        except Exception:
                            continue
        return None

    def _write_shim(legacy_mod: str, target_mod: str):
        # legacy_mod like "uwg.BEMDef" -> write <uwg_root>/BEMDef.py
        tail = legacy_mod.split(".", 1)[1]
        shim_path = uwg_root / f"{tail}.py"
        if shim_path.exists():
            return
        shim_path.write_text(
            "# Auto-generated shim for legacy DOERefBuildings pickle compatibility\n"
            f"# Maps '{legacy_mod}' -> '{target_mod}'\n"
            f"from {target_mod} import *\n"
        )

    # Shims we’ve observed in the errors
    # Some legacy names map to different module/file names in newer UWG versions
    LEGACY_TAIL_MAP = {
        "schedule": ["schdef", "sch", "schedules"],
    }

    needed = ["uwg.BEMDef", "uwg.building", "uwg.element", "uwg.material", "uwg.utilities", "uwg.schdef", "uwg.schedule"]

    for legacy in needed:
        target = _find_target_module(legacy)
        if target is None:
            tail = legacy.split(".", 1)[1]
            target = _scan_for_class_or_file(tail)

        if target is None:
            print(f"⚠️ Could not resolve target module for legacy import: {legacy}")
            continue

        if legacy != target:
            _write_shim(legacy, target)

    # invalidate caches so subsequent imports see the new shims
    importlib.invalidate_caches()


# Robust UWG import across versions/exports
try:
    from uwg import UWG
except Exception:
    from uwg.uwg import UWG

# Local helper module (same folder)
from uwg_setup_utils import (
    detect_cities,
    find_rmy_epw,
    ensure_city_uwg_dir,
    patch_readDOE_mass_wall_roof,
    build_initialize_residential,
    set_key_line_in_file,
)


def _find_base_epw_safe(wxbuild_root: Path, city: str, chosen_filename: str | None = None) -> Path:
    """
    Pick the BASE EPW from <wxbuild_root>/epwdata/<city>/Chosen_scenario/.

    - If chosen_filename is provided, that exact file is used (must exist).
    - Otherwise, prefer a filename containing the city key; else fall back to the first EPW found.
    """
    chosen_dir = wxbuild_root / "epwdata" / city / "Chosen_scenario"
    eps = sorted(chosen_dir.glob("*.epw"))
    if not eps:
        raise FileNotFoundError(f"No EPW files found in: {chosen_dir}")

    if chosen_filename:
        p = chosen_dir / chosen_filename
        if not p.exists():
            raise FileNotFoundError(f"Chosen base EPW not found: {p}")
        return p

    city_key = city.lower()
    for ep in eps:
        if city_key in ep.name.lower():
            return ep
    return eps[0]


def launch_ui(wxbuild_root: str = "/content/wxbuild", uwg_root: str = "/content/uwg"):
    """
    Student-friendly UWG UI:
    - Auto-detect cities under <wxbuild_root>/epwdata/
    - Uses the BASE EPW found in <city>/Chosen_scenario/
    - Writes outputs under <city>/uwg/
    """
    wxbuild_root = Path(wxbuild_root).resolve()
    uwg_root = Path(uwg_root).resolve()

    cities = detect_cities(wxbuild_root)
    if not cities:
        raise RuntimeError(f"No city folders found under: {wxbuild_root / 'epwdata'}")

    city_dd = widgets.Dropdown(
        options=cities,
        value=cities[0],
        description="City",
        style={"description_width": "140px"},
        layout=widgets.Layout(width="520px"),
    )

    # Hide City selector from the UI (still used internally)
    city_dd.layout.display = "none"

    # --- Base EPW selector (reads from epwdata/<city>/Chosen_scenario/) ---
    def _list_chosen_epws(city: str):
        chosen_dir = wxbuild_root / "epwdata" / city / "Chosen_scenario"
        if not chosen_dir.exists():
            return []
        return sorted([p.name for p in chosen_dir.glob("*.epw")])

    def _refresh_base_dropdown(*_):
        opts = _list_chosen_epws(city_dd.value)
        base_dd.options = opts
        if opts:
            if base_dd.value not in opts:
                base_dd.value = opts[0]
        # Show dropdown only if multiple base EPWs exist
        if len(opts) <= 1:
            base_dd.layout.display = "none"
        else:
            base_dd.layout.display = None

    base_dd = widgets.Dropdown(
        options=[],
        value=None,
        description="Base EPW",
        style={"description_width": "140px"},
        layout=widgets.Layout(width="520px"),
    )

    city_dd.observe(lambda ch: _refresh_base_dropdown(), names="value")
    _refresh_base_dropdown()

    # ---- layout knobs ----
    DESC_W      = "360px"
    SLIDER_W    = "640px"
    COL_W       = "860px"
    GAP_PX      = 4

    style  = {"description_width": DESC_W}
    slider_layout = widgets.Layout(width=SLIDER_W)
    col_layout    = widgets.Layout(width=COL_W)

    # --- sliders ---
    bldheight   = widgets.FloatSlider(value=12.0, min=3.0,  max=60.0, step=1.0,
                                      description="Building Height (m)",
                                      style=style, layout=slider_layout, continuous_update=False)
    blddensity  = widgets.FloatSlider(value=0.25, min=0.05, max=0.80, step=0.01,
                                      description="Building Footprint Density (0–1)",
                                      style=style, layout=slider_layout, continuous_update=False)
    vertohor    = widgets.FloatSlider(value=0.40, min=0.10, max=2.00, step=0.05,
                                      description="Canyon Aspect Ratio H/W (–)",
                                      style=style, layout=slider_layout, continuous_update=False)
    albroad     = widgets.FloatSlider(value=0.12, min=0.05, max=0.50, step=0.01,
                                      description="Road Albedo (–)",
                                      style=style, layout=slider_layout, continuous_update=False)
    grasscover  = widgets.FloatSlider(value=0.30, min=0.0,  max=0.90, step=0.01,
                                      description="Green Space Cover (–)",
                                      style=style, layout=slider_layout, continuous_update=False)
    treecover   = widgets.FloatSlider(value=0.25, min=0.0,  max=0.90, step=0.01,
                                      description="Tree Canopy Cover (–)",
                                      style=style, layout=slider_layout, continuous_update=False)
    sensanth    = widgets.FloatSlider(value=1.0,  min=0.0,  max=30.0, step=0.5,
                                      description="Traffic / Anthropogenic Heat (W/m²)",
                                      style=style, layout=slider_layout, continuous_update=False)

    glzr  = widgets.FloatSlider(value=0.30, min=0.05, max=0.80, step=0.01,
                                description="Glazing Ratio (Window-to-Wall, –)",
                                style=style, layout=slider_layout, continuous_update=False)
    shgc  = widgets.FloatSlider(value=0.35, min=0.10, max=0.80, step=0.01,
                                description="Window SHGC (–)",
                                style=style, layout=slider_layout, continuous_update=False)
    albroof = widgets.FloatSlider(value=0.60, min=0.05, max=0.90, step=0.01,
                                  description="Roof Albedo (–)",
                                  style=style, layout=slider_layout, continuous_update=False)

    run_btn = widgets.Button(description="Generate UWG EPW (Full Year)", button_style="success")
    out = widgets.Output()

    def on_run(_):
        with out:
            clear_output()
            print("✅ Clicked Generate")

            try:
                city = city_dd.value

                # Select EPW + output directory
                EPW_PATH = str(_find_base_epw_safe(wxbuild_root, city, chosen_filename=base_dd.value))
                OUT_DIR  = str(ensure_city_uwg_dir(wxbuild_root, city))

                print(f"City: {city}")
                print(f"Original EPW: {EPW_PATH}")
                print(f"UWG output folder: {OUT_DIR}")

                # Patch UWG (legacy behavior)
                patch_readDOE_mass_wall_roof(uwg_root)

                PARAM_UWG = str(Path(OUT_DIR) / "initialize_residential.uwg")
                if not Path(PARAM_UWG).exists():
                    build_initialize_residential(uwg_root, PARAM_UWG, default_zone="5A")

                # Write parameters
                set_key_line_in_file(PARAM_UWG, "bldHeight,",   f"bldHeight,{bldheight.value},")
                set_key_line_in_file(PARAM_UWG, "bldDensity,",  f"bldDensity,{blddensity.value},")
                set_key_line_in_file(PARAM_UWG, "verToHor,",    f"verToHor,{vertohor.value},")
                set_key_line_in_file(PARAM_UWG, "albRoad,",     f"albRoad,{albroad.value},")
                set_key_line_in_file(PARAM_UWG, "grasscover,",  f"grasscover,{grasscover.value},")
                set_key_line_in_file(PARAM_UWG, "treeCover,",   f"treeCover,{treecover.value},")
                set_key_line_in_file(PARAM_UWG, "sensAnth,",    f"sensAnth,{sensanth.value},")
                set_key_line_in_file(PARAM_UWG, "zone,",        "zone,5A,")
                set_key_line_in_file(PARAM_UWG, "nDay,",        "nDay,365,")
                set_key_line_in_file(PARAM_UWG, "dtSim,",       "dtSim,15,")

                set_key_line_in_file(PARAM_UWG, "glzr,",        f"glzr,{glzr.value},")
                set_key_line_in_file(PARAM_UWG, "shgc,",        f"shgc,{shgc.value},")
                set_key_line_in_file(PARAM_UWG, "albRoof,",     f"albRoof,{albroof.value},")

                run_i = _next_run_index(OUT_DIR, city)
                out_name = f"{city}_UWG_Run{run_i}.epw"

                run_row = {
                    "timestamp_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
                    "city": city,
                    "run_index": run_i,
                    "epw_source": EPW_PATH,
                    "param_file": PARAM_UWG,
                    "default_zone": "5A",
                    "bldheight_m": bldheight.value,
                    "blddensity": blddensity.value,
                    "vertohor_h_w": vertohor.value,
                    "albroad": albroad.value,
                    "grasscover": grasscover.value,
                    "treecover": treecover.value,
                    "sensanth_w_m2": sensanth.value,
                    "glzr": glzr.value,
                    "shgc": shgc.value,
                    "albroof": albroof.value,
                }
                _append_run_log(OUT_DIR, city, run_row)

                print(f"Param file: {PARAM_UWG}")

                # Run UWG
                m = UWG.from_param_file(PARAM_UWG, EPW_PATH, OUT_DIR, out_name)

                _ensure_uwg_legacy_pickle_shims()

                m.generate()
                m.simulate()
                m.write_epw()

                new_epw = Path(m.new_epw_path)
                print(f"✅ Wrote: {new_epw}")

                # --- Plot ONLY base vs new dry-bulb (like your example) ---
                base_df = pd.read_csv(EPW_PATH, skiprows=8, header=None)
                new_df  = pd.read_csv(new_epw,  skiprows=8, header=None)

                # EPW dry-bulb is column 6 (0-indexed) for standard EPW (DBT)
                # Keep robust: if fewer cols, error out.
                if base_df.shape[1] < 7 or new_df.shape[1] < 7:
                    raise ValueError("EPW data does not have expected number of columns (need at least 7).")

                base_t = base_df.iloc[:, 6].astype(float).to_numpy()
                new_t  = new_df.iloc[:, 6].astype(float).to_numpy()

                fig, ax = plt.subplots(figsize=(15.5, 4.2))
                ax.plot(base_t, label="Base EPW", color="0.25", linewidth=2)
                ax.plot(new_t,  label=f"UWG_Run{run_i}", color="#B11F3A", linewidth=2)

                ax.set_title("UWG vs Base: Dry-Bulb Temperature")
                ax.set_xlabel("Hour of Year")
                ax.set_ylabel("Dry-Bulb Temperature (°C)")
                ax.legend(frameon=False, loc="upper right")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                plt.show()

            except Exception as e:
                import traceback
                print("\n❌ UWG run failed. Full traceback:\n")
                print(traceback.format_exc())

    run_btn.on_click(on_run)

    
    # --- Land Coverage helpers ---
    landcov_title = widgets.HTML("<h4 style='margin:10px 0 4px 0;'>Land Coverage</h4>")
    landcov_note  = widgets.HTML("<i>Note: In UWG, Building Footprint Density + Green Space Cover + Tree Canopy Cover should approximately sum to 1.0.</i>")

    # Low / Medium / High aligned under the slider track (start / middle / end)
    # Track width is empirically set to match the visible slider groove in Colab.
    VALUE_W = "80px"
    TRACK_W = "280px"
    _lmh = ("<div style='width:100%; display:flex; font-size:11px; color:#666;'>""<span style='flex:1; text-align:left;'>Low</span>""<span style='flex:1; text-align:center;'>Medium</span>""<span style='flex:1; text-align:right;'>High</span></div>")
    _spacer = widgets.HTML("", layout=widgets.Layout(width=DESC_W))
    def _lmh_row():
        # Right spacer matches the slider's value box so labels align under the track only
        _val_spacer = widgets.HTML("", layout=widgets.Layout(width=VALUE_W))
        return widgets.HBox(
            [_spacer,
             widgets.HTML(_lmh, layout=widgets.Layout(width=TRACK_W, margin="-6px 0 0 0")),
             _val_spacer],
            layout=widgets.Layout(width=SLIDER_W, margin="0 0 0 0"),
        )

    land_row_m = widgets.Layout(margin="0 0 6px 0")
    land_tree  = widgets.VBox([treecover,  _lmh_row()], layout=land_row_m)
    land_grass = widgets.VBox([grasscover, _lmh_row()], layout=land_row_m)
    land_dens  = widgets.VBox([blddensity, _lmh_row()], layout=land_row_m)

    urban_box = widgets.VBox([
        widgets.HTML("<h3>Urban Design</h3>"),

        landcov_title,
        land_tree,
        land_grass,
        land_dens,
        landcov_note,

        bldheight,
        vertohor,
        albroad,
        sensanth,
    ], layout=col_layout)

    bldg_box = widgets.VBox([
        widgets.HTML("<h3>Building Design</h3>"),
        glzr, shgc, albroof
    ], layout=col_layout)

    display(widgets.VBox([
        city_dd,
        base_dd,
        widgets.HBox([urban_box, widgets.HTML(f"<div style='width:{GAP_PX}px;'></div>"), bldg_box]),
        run_btn,
        out
    ]))


launch_ui()
