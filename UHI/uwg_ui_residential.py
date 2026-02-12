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
    detect_ashrae_zone_from_epw,
    patch_uwg_param_file,
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

tsph_dd = widgets.Dropdown(
    options=[24, 48, 60],
    value=60,
    description="Timesteps/hour",
    style={"description_width": "140px"},
    layout=widgets.Layout(width="260px")
)
                                  description="Roof Albedo (–)",
                                  style=style, layout=slider_layout, continuous_update=False)

    run_btn = widgets.Button(description="Generate UWG EPW (Full Year)", button_style="success")
    out = widgets.Output()

    def on_run(_):
    with out:
        clear_output(wait=True)
        try:
            print("✅ Clicked Generate")

            city = city_dd.value
            city_dir = os.path.join(EPW_ROOT, city)

            # EPW selection
            EPW_PATH = _find_base_epw_safe(
                city_dir=city_dir,
                chosen_filename=(base_dd.value if ("base_dd" in globals()) else None)
            )
            if not EPW_PATH:
                raise FileNotFoundError("No EPW found for this city and selection.")
            print("City:", city)
            print("Original EPW:", EPW_PATH)

            # Param template
            TEMPLATE_PATH = _find_param_template(city_dir)
            if not TEMPLATE_PATH:
                raise FileNotFoundError(f"No .uwg template found under: {os.path.join(city_dir,'uwg')}")
            print("UWG template:", TEMPLATE_PATH)

            # Auto-detect ASHRAE zone (best-effort)
            detected_zone = detect_ashrae_zone_from_epw(EPW_PATH)
            if detected_zone:
                print("Detected ASHRAE zone from EPW:", detected_zone)
            else:
                detected_zone = "5A"
                print("⚠️ Could not detect zone from EPW. Using fallback:", detected_zone)

            # Supported kwargs for THIS uwg version
            import inspect
            supported = set(inspect.signature(UWG.from_param_args).parameters.keys())

            # Timesteps per hour (stability) → dtsim/dtweather in seconds
            tsph = int(tsph_dd.value) if "tsph_dd" in globals() else 60
            dt_sec = int(3600 / tsph)

            # Output EPW
            os.makedirs(os.path.join(city_dir, "uwg"), exist_ok=True)
            out_epw_name = f"{city}_UWG_{os.path.splitext(os.path.basename(EPW_PATH))[0]}.epw"
            out_epw_path = os.path.join(city_dir, "uwg", out_epw_name)

            # Patch template params (no readDOE)
            updates = {
                "bldHeight": float(bldheight.value),
                "bldDensity": float(blddensity.value),
                "verToHor": float(vertohor.value),
                "albRoad": float(albroad.value),
                "grasscover": float(grasscover.value),
                "treeCover": float(treecover.value),
                "sensAnth": float(sensanth.value),
                "zone": str(detected_zone),
            }

            # Optional (only if present in template AND supported by UWG)
            # Note: Some UWG builds don't accept these via from_param_args; we pass them only if supported.
            if "glzr" in supported:
                updates["glzr"] = float(glzr.value)
            if "shgc" in supported:
                updates["shgc"] = float(shgc.value)
            if "albroof" in supported:
                updates["albroof"] = float(albroof.value)

            # Write patched uwg file next to template
            patched_path = os.path.join(city_dir, "uwg", "initialize_residential_patched.uwg")
            patch_uwg_param_file(
                template_path=TEMPLATE_PATH,
                out_path=patched_path,
                updates=updates,
                schtraffic=None  # keep schedule from template (can add widget later)
            )
            print("Patched UWG file:", patched_path)

            # Run UWG using from_param_file (robust) if available, else from_param_args
            # The Ladybug Tools UWG exposes from_param_file in newer releases.
            if hasattr(UWG, "from_param_file"):
                model = UWG.from_param_file(
                    param_path=patched_path,
                    epw_path=EPW_PATH,
                    new_epw_dir=os.path.dirname(out_epw_path),
                    new_epw_name=os.path.basename(out_epw_path),
                    month=1, day=1, nday=365,
                    dtsim=dt_sec, dtweather=dt_sec
                )
            else:
                # Fall back to from_param_args
                params = dict(
                    epw_path=EPW_PATH,
                    new_epw_dir=os.path.dirname(out_epw_path),
                    new_epw_name=os.path.basename(out_epw_path),
                    month=1, day=1, nday=365,
                    dtsim=dt_sec, dtweather=dt_sec,
                    bldheight=float(bldheight.value),
                    blddensity=float(blddensity.value),
                    vertohor=float(vertohor.value),
                    albroad=float(albroad.value),
                    grasscover=float(grasscover.value),
                    treecover=float(treecover.value),
                    sensanth=float(sensanth.value),
                    zone=str(detected_zone),
                )
                # Optional
                if "glzr" in supported: params["glzr"] = float(glzr.value)
                if "shgc" in supported: params["shgc"] = float(shgc.value)
                if "albroof" in supported: params["albroof"] = float(albroof.value)

                # Drop any unsupported keys (extra safe)
                params = {k:v for k,v in params.items() if k in supported}
                model = UWG.from_param_args(**params)

            print(f"Running UWG… (nday=365, timesteps/hour={tsph})")
            model.generate()
            model.simulate()
            model.write_epw()

            out_path = getattr(model, "new_epw_path", out_epw_path)
            print("✅ UWG EPW written:", out_path)

            # Quick sanity plot
            try:
                base_df = read_epw(EPW_PATH)
                uwg_df  = read_epw(out_path)
                _plot_compare(base_df, uwg_df, title="Baseline vs UWG (Dry-bulb)")
            except Exception as e:
                print("⚠️ Plot skipped:", str(e))

        except Exception:
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
        tsph_dd,
        run_btn,
        out
    ]))


launch_ui()
