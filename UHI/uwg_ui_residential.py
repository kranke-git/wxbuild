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

    # -------------------------
    # Building design
    # -------------------------
    glzr_sl = widgets.FloatSlider(
        value=0.30, min=0.05, max=0.80, step=0.05,
        description="Glazing Ratio (WWR)", style=STYLE, layout=SLIDER_LAYOUT
    )
    shgc_sl = widgets.FloatSlider(
        value=0.35, min=0.10, max=0.80, step=0.05,
        description="Window SHGC (-)", style=STYLE, layout=SLIDER_LAYOUT
    )
    albroof_sl = widgets.FloatSlider(
        value=0.60, min=0.05, max=0.90, step=0.05,
        description="Roof Albedo (-)", style=STYLE, layout=SLIDER_LAYOUT
    )

    # -------------------------
    # Stability / runtime
    # -------------------------
    tsph_dd = widgets.Dropdown(
        options=[24, 48, 60],
        value=60,
        description="Timesteps/hour",
        style=STYLE,
        layout=widgets.Layout(width="420px")
    )

    # Additional vegetation params exposed in template
    albveg_sl = widgets.FloatSlider(
        value=0.25, min=0.05, max=0.50, step=0.01,
        description="Vegetation Albedo (-)", style=STYLE, layout=SLIDER_LAYOUT
    )
    latgrss_sl = widgets.FloatSlider(
        value=0.40, min=0.0, max=1.0, step=0.05,
        description="Grass Latent Fraction (0–1)", style=STYLE, layout=SLIDER_LAYOUT
    )
    lattree_sl = widgets.FloatSlider(
        value=0.60, min=0.0, max=1.0, step=0.05,
        description="Tree Latent Fraction (0–1)", style=STYLE, layout=SLIDER_LAYOUT
    )
    rurveg_sl = widgets.FloatSlider(
        value=0.90, min=0.0, max=1.0, step=0.05,
        description="Rural Veg Cover (0–1)", style=STYLE, layout=SLIDER_LAYOUT
    )

    # Traffic schedule scaling
    traffic_scale_sl = widgets.FloatSlider(
        value=1.0, min=0.0, max=2.0, step=0.1,
        description="Traffic Intensity Multiplier (-)", style=STYLE, layout=SLIDER_LAYOUT
    )

    # Zone override (auto-detect preferred)
    zone_dd = widgets.Dropdown(
        options=["1A","2A","2B","3A","3B","3C","4A","4B","4C","5A","5B","6A","6B","7","8"],
        value="2B",
        description="ASHRAE Zone (override)",
        style=STYLE,
        layout=widgets.Layout(width="420px")
    )

    use_auto_zone_cb = widgets.Checkbox(value=True, description="Auto-detect zone from EPW header")

    nday_sl = widgets.IntSlider(
        value=365, min=7, max=365, step=1,
        description="Simulation Length (days)", style=STYLE, layout=SLIDER_LAYOUT
    )

    # Plot window
    plot_month_dd = widgets.Dropdown(options=list(range(1,13)), value=7, description="Plot month", style=STYLE, layout=widgets.Layout(width="260px"))
    plot_day_dd   = widgets.Dropdown(options=list(range(1,29)), value=1, description="Plot day", style=STYLE, layout=widgets.Layout(width="220px"))
    plot_days_dd  = widgets.Dropdown(options=[7,10,14], value=7, description="Plot span", style=STYLE, layout=widgets.Layout(width="220px"))

    run_btn = widgets.Button(description="Generate UWG EPW", button_style="primary")

    out = widgets.Output()

    ui = widgets.VBox([
        widgets.HTML("<h3 style='margin:6px 0'>Select inputs</h3>"),
        widgets.HBox([city_dd, epw_dd]),
        widgets.HBox([use_auto_zone_cb, zone_dd, tsph_dd]),
        nday_sl,
        widgets.HTML("<h3 style='margin:10px 0 6px 0'>Urban Design</h3>"),
        bldheight_sl, blddensity_sl, vertohor_sl, charlength_sl, albroad_sl, sensanth_sl,
        grasscover_sl, treecover_sl, vegstart_sl, vegend_sl,
        albveg_sl, latgrss_sl, lattree_sl, rurveg_sl,
        traffic_scale_sl,
        widgets.HTML("<h3 style='margin:10px 0 6px 0'>Building Design</h3>"),
        glzr_sl, shgc_sl, albroof_sl,
        widgets.HTML("<h3 style='margin:10px 0 6px 0'>Plotting</h3>"),
        widgets.HBox([plot_month_dd, plot_day_dd, plot_days_dd]),
        run_btn,
        out,
    ])

    display(ui)

    def _scale_schtraffic(mult: float):
        # SchTraffic is 3x24 in the template; keep in [0,1]
        s = []
        for row in DEFAULT_SCHTRAFFIC:
            arr = np.clip(np.array(row, dtype=float) * float(mult), 0.0, 1.0)
            s.append(arr.tolist())
        return s

    def on_run(_):
        with out:
            clear_output(wait=True)
            try:
                city = city_dd.value
                epw_sel = epw_dd.value
                if not epw_sel:
                    raise RuntimeError("No EPW selected.")

                # ensure /uwg folder + template
                uwg_dir = ensure_city_uwg_dir(WXBUILD_ROOT, city)
                template_path = build_initialize_residential(uwg_dir)

                detected = detect_ashrae_zone_from_epw(epw_sel)
                zone_val = detected if (use_auto_zone_cb.value and detected) else zone_dd.value

                tsph = int(tsph_dd.value)
                dt_sec = int(round(3600 / tsph))

                updates = {
                    "bldHeight": float(bldheight_sl.value),
                    "bldDensity": float(blddensity_sl.value),
                    "verToHor": float(vertohor_sl.value),
                    "h_mix": 1.0,  # keep template default
                    "charLength": float(charlength_sl.value),
                    "albRoad": float(albroad_sl.value),
                    "sensAnth": float(sensanth_sl.value),
                    "zone": str(zone_val),
                    "grasscover": float(grasscover_sl.value),
                    "treeCover": float(treecover_sl.value),
                    "vegStart": int(vegstart_sl.value),
                    "vegEnd": int(vegend_sl.value),
                    "albVeg": float(albveg_sl.value),
                    "latGrss": float(latgrss_sl.value),
                    "latTree": float(lattree_sl.value),
                    "rurVegCover": float(rurveg_sl.value),
                    "dtsim": int(dt_sec),
                    "dtweather": int(dt_sec),
                    "glzr": float(glzr_sl.value),
                    "shgc": float(shgc_sl.value),
                    "albroof": float(albroof_sl.value),
                }

                schtraffic = _scale_schtraffic(float(traffic_scale_sl.value))

                patched_path = os.path.join(uwg_dir, "initialize_residential_patched.uwg")
                patch_uwg_param_file(
                    template_path=template_path,
                    out_path=patched_path,
                    updates=updates,
                    schtraffic=schtraffic,
                )

                # run UWG
                nday = int(nday_sl.value)
                out_name = f"{city}_UWG_{os.path.splitext(os.path.basename(epw_sel))[0]}_nd{nday}_tsph{tsph}.epw"

                print("✅ Clicked Generate")
                print("City:", city)
                print("Original EPW:", epw_sel)
                print("UWG param file:", patched_path)
                print("Detected zone:", detected)
                print("Using zone:", zone_val)

                model = UWG.from_param_args(
                    epw_path=epw_sel,
                    month=1,
                    day=1,
                    nday=nday,
                    new_epw_dir=uwg_dir,
                    new_epw_name=out_name,
                    dtsim=int(dt_sec),
                    dtweather=int(dt_sec),
                    # urban
                    bldheight=float(bldheight_sl.value),
                    blddensity=float(blddensity_sl.value),
                    vertohor=float(vertohor_sl.value),
                    charlength=float(charlength_sl.value),
                    albroad=float(albroad_sl.value),
                    sensanth=float(sensanth_sl.value),
                    grasscover=float(grasscover_sl.value),
                    treecover=float(treecover_sl.value),
                    zone=str(zone_val),
                    # building
                    glzr=float(glzr_sl.value),
                    shgc=float(shgc_sl.value),
                    albroof=float(albroof_sl.value),
                    # traffic schedule (if supported)
                    schtraffic=tuple(tuple(r) for r in schtraffic),
                )

                model.generate()
                model.simulate()
                model.write_epw()

                out_epw = model.new_epw_path
                print("\n✅ UWG EPW generated:", out_epw)

                # quick plot
                base_df = read_epw(epw_sel)
                uwg_df = read_epw(out_epw)

                y = int(base_df["Year"].mode().iloc[0])
                start = pd.Timestamp(year=y, month=int(plot_month_dd.value), day=int(plot_day_dd.value))
                end = start + pd.Timedelta(days=int(plot_days_dd.value))
                bsub = base_df[(base_df["timestamp"] >= start) & (base_df["timestamp"] < end)]
                usub = uwg_df[(uwg_df["timestamp"] >= start) & (uwg_df["timestamp"] < end)]

                plt.figure(figsize=(12,4))
                plt.plot(bsub["timestamp"], bsub["DryBulb_C"], label="Original")
                plt.plot(usub["timestamp"], usub["DryBulb_C"], label="UWG")
                plt.ylabel("Dry-bulb (°C)")
                plt.title("Original vs UWG (selected window)")
                plt.grid(True)
                plt.legend()
                plt.show()

            except Exception:
                print("\n❌ UWG run failed. Full traceback:\n")
                print(traceback.format_exc())

    run_btn.on_click(on_run)


if __name__ == "__main__":
    launch_ui()
