from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import csv
import threading
import ctypes
import sys


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
    import re as _re
    from pathlib import Path as _Path
    import uwg as _uwg

    uwg_root = _Path(list(_uwg.__path__)[0])  # e.g., /content/uwg
    pkg_dir_candidates = []
    if (uwg_root / "uwg").exists():
        pkg_dir_candidates.append(uwg_root / "uwg")
    pkg_dir_candidates.append(uwg_root)

    def _find_target_module(legacy_name: str) -> str | None:
        try:
            importlib.import_module(legacy_name)
            return legacy_name
        except Exception:
            pass

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
        alt_tails = [tail]
        if tail.lower() in LEGACY_TAIL_MAP:
            alt_tails = LEGACY_TAIL_MAP[tail.lower()] + alt_tails

        for pkg_dir in pkg_dir_candidates:
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
        tail = legacy_mod.split(".", 1)[1]
        shim_path = uwg_root / f"{tail}.py"
        if shim_path.exists():
            return
        shim_path.write_text(
            "# Auto-generated shim for legacy DOERefBuildings pickle compatibility\n"
            f"# Maps '{legacy_mod}' -> '{target_mod}'\n"
            f"from {target_mod} import *\n"
        )

    LEGACY_TAIL_MAP = {"schedule": ["schdef", "sch", "schedules"]}
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

    importlib.invalidate_caches()


try:
    from uwg import UWG
except Exception:
    from uwg.uwg import UWG

from uwg_setup_utils import (
    detect_cities,
    find_rmy_epw,
    ensure_city_uwg_dir,
    patch_readDOE_mass_wall_roof,
    build_initialize_residential,
    set_key_line_in_file,
)


class _WidgetStdout:
    """Redirect stdout/stderr into an ipywidgets.Output widget (thread-safe enough for prints)."""
    def __init__(self, out: widgets.Output):
        self.out = out

    def write(self, s):
        if not s:
            return
        try:
            self.out.append_stdout(str(s))
        except Exception:
            pass

    def flush(self):
        return


def launch_ui(wxbuild_root: str = "/content/wxbuild", uwg_root: str = "/content/uwg"):
    """
    Student-friendly UWG UI:
    - Auto-detect cities under <wxbuild_root>/epwdata/
    - Uses the EPW found in <city>/rmy/
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
    # Hide city selector from UI (still used internally)
    city_dd.layout.display = "none"

    # ---- layout knobs ----
    DESC_W      = "360px"
    SLIDER_W    = "640px"
    COL_W       = "860px"
    GAP_PX      = 4  # reduced gap between Urban and Building columns

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

    # Building design
    glzr  = widgets.FloatSlider(value=0.30, min=0.05, max=0.80, step=0.01,
                                description="Glazing Ratio (Window-to-Wall, –)",
                                style=style, layout=slider_layout, continuous_update=False)
    shgc  = widgets.FloatSlider(value=0.35, min=0.10, max=0.80, step=0.01,
                                description="Window SHGC (–)",
                                style=style, layout=slider_layout, continuous_update=False)
    albroof = widgets.FloatSlider(value=0.60, min=0.05, max=0.90, step=0.01,
                                  description="Roof Albedo (–)",
                                  style=style, layout=slider_layout, continuous_update=False)

    run_btn  = widgets.Button(description="Generate UWG EPW (Full Year)", button_style="success",
                              layout=widgets.Layout(width="220px"))
    stop_btn = widgets.Button(description="Stop simulation", button_style="warning",
                              layout=widgets.Layout(width="170px"))
    stop_btn.disabled = True

    buttons_box = widgets.HBox([run_btn, stop_btn], layout=widgets.Layout(gap="10px"))

    out = widgets.Output()

    # ---- Land coverage UI helpers ----
    landcov_title = widgets.HTML("<h4 style='margin:10px 0 4px 0;'>Land Coverage</h4>")
    landcov_note = widgets.HTML(
        "<i>Note: In UWG, Building Footprint Density + Green Space Cover + Tree Canopy Cover should approximately sum to 1.0.</i>"
    )

    # Align Low / Medium / High under the slider TRACK (start / center / end)
    # We do: [spacer (DESC_W)] + [flex box with width matching the visible track]
    TRACK_W = "240px"
    _lmh = (
        "<div style='width:100%; display:flex; justify-content:space-between; "
        "font-size:11px; color:#666; padding:0; margin:0;'>"
        "<span>Low</span><span>Medium</span><span>High</span></div>"
    )
    _spacer = widgets.HTML("", layout=widgets.Layout(width=DESC_W))
    def _lmh_row():
        return widgets.HBox(
            [_spacer, widgets.HTML(_lmh, layout=widgets.Layout(width=TRACK_W, margin="-6px 0 0 0"))],
            layout=widgets.Layout(width=SLIDER_W, margin="0 0 0 0"),
        )

    # Compact vertical spacing to match other sliders
    land_row_m = widgets.Layout(margin="0 0 6px 0")
    row_m      = widgets.Layout(margin="0 0 10px 0")
    landcov_note.layout = widgets.Layout(margin="0 0 8px 0")

    land_tree  = widgets.VBox([treecover,  _lmh_row()], layout=land_row_m)
    land_grass = widgets.VBox([grasscover, _lmh_row()], layout=land_row_m)
    land_dens  = widgets.VBox([blddensity, _lmh_row()], layout=land_row_m)

    # ---- Simulation threading + stop ----
    _state = {"thread": None, "thread_id": None, "running": False}

    def _async_raise(tid, exctype):
        if tid is None:
            return
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
        if res == 0:
            return
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)

    def _run_logic():
        # This is the original working on_run logic (kept intact),
        # just with updated params and UI ordering.
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = _WidgetStdout(out)
        try:
            with out:
                clear_output()
            city = city_dd.value

            EPW_PATH = str(find_rmy_epw(wxbuild_root, city))
            OUT_DIR  = str(ensure_city_uwg_dir(wxbuild_root, city))

            patch_readDOE_mass_wall_roof(uwg_root)

            PARAM_UWG = str(Path(OUT_DIR) / "initialize_residential.uwg")
            if not Path(PARAM_UWG).exists():
                # fixed default zone internally (was dropdown previously)
                build_initialize_residential(uwg_root, PARAM_UWG, default_zone="5A")

            print("City:", city)
            print("Original EPW:", EPW_PATH)
            print("UWG output folder:", OUT_DIR)
            print("Param file:", PARAM_UWG)

            set_key_line_in_file(PARAM_UWG, "bldHeight,",   f"bldHeight,{bldheight.value},")
            set_key_line_in_file(PARAM_UWG, "bldDensity,",  f"bldDensity,{blddensity.value},")
            set_key_line_in_file(PARAM_UWG, "verToHor,",    f"verToHor,{vertohor.value},")
            set_key_line_in_file(PARAM_UWG, "albRoad,",     f"albRoad,{albroad.value},")
            set_key_line_in_file(PARAM_UWG, "grasscover,",  f"grasscover,{grasscover.value},")
            set_key_line_in_file(PARAM_UWG, "treeCover,",   f"treeCover,{treecover.value},")
            set_key_line_in_file(PARAM_UWG, "sensAnth,",    f"sensAnth,{sensanth.value},")
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

            m = UWG.from_param_file(PARAM_UWG, EPW_PATH, OUT_DIR, out_name)

            _ensure_uwg_legacy_pickle_shims()
            m.generate()

            b = m.BEM[0]
            print("BEM wall name:", getattr(getattr(b, "wall", None), "name", "NA"))
            print("BEM roof name:", getattr(getattr(b, "roof", None), "name", "NA"))

            # This call prints "Simulating Day ..." in the UWG library.
            m.simulate()
            m.write_epw()
            print("\n✅ wrote:", m.new_epw_path)

            # Quick comparison plot
            orig = pd.read_csv(EPW_PATH, skiprows=8, header=None)
            uwg_df = pd.read_csv(m.new_epw_path, skiprows=8, header=None)

            base = pd.DataFrame({
                "hoy": np.arange(len(orig)),
                "dbt": pd.to_numeric(orig.iloc[:, 6], errors="coerce"),
            })
            uwg_run = pd.DataFrame({
                "hoy": np.arange(len(uwg_df)),
                "dbt": pd.to_numeric(uwg_df.iloc[:, 6], errors="coerce"),
            })

            n = int(min(len(base), len(uwg_run)))
            base = base.iloc[:n].reset_index(drop=True)
            uwg_run = uwg_run.iloc[:n].reset_index(drop=True)

            ddbt = (uwg_run["dbt"].to_numpy() - base["dbt"].to_numpy())

            print("\nDry-bulb ΔT (UWG - Original), °C")
            print("  mean:", float(np.mean(ddbt)))
            print("  min :", float(np.min(ddbt)))
            print("  max :", float(np.max(ddbt)))

            COL_BASE = "0.25"     # dark grey
            COL_UWG  = "#7A0000"  # deep red

            plt.figure(figsize=(12, 4))
            plt.plot(base["hoy"], base["dbt"], label=f"Base ({Path(EPW_PATH).name})",
                     linewidth=2.0, color=COL_BASE)
            plt.plot(uwg_run["hoy"], uwg_run["dbt"], label=Path(m.new_epw_path).stem,
                     linewidth=2.2, color=COL_UWG)

            plt.title(f"{city} — Dry-bulb temperature: Base vs UWG run")
            plt.xlabel("Hour of year")
            plt.ylabel("Dry-bulb Temperature (°C)")

            ax = plt.gca()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", which="both", direction="out")
            ax.grid(True, alpha=0.25)
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.show()

        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    def on_run(_):
        if _state["running"]:
            return
        _state["running"] = True
        run_btn.disabled = True
        stop_btn.disabled = False

        def _worker():
            try:
                _run_logic()
            except KeyboardInterrupt:
                out.append_stdout("\n⏹ Simulation stopped.\n")
            except Exception as e:
                out.append_stdout(f"\n❌ Error: {e}\n")
            finally:
                _state["running"] = False
                run_btn.disabled = False
                stop_btn.disabled = True

        t = threading.Thread(target=_worker, daemon=True)
        _state["thread"] = t
        t.start()
        _state["thread_id"] = t.ident

    def on_stop(_):
        if _state.get("running"):
            _async_raise(_state.get("thread_id"), KeyboardInterrupt)

    run_btn.on_click(on_run)
    stop_btn.on_click(on_stop)

    # ---- UI layout ----
    urban_box = widgets.VBox([
        widgets.HTML("<h3>Urban Design</h3>"),
        landcov_title,
        land_tree,
        land_grass,
        land_dens,
        landcov_note,
        widgets.VBox([bldheight], layout=row_m),
        widgets.VBox([vertohor], layout=row_m),
        widgets.VBox([albroad], layout=row_m),
        widgets.VBox([sensanth], layout=row_m),
    ], layout=col_layout)

    bldg_box = widgets.VBox([
        widgets.HTML("<h3>Building Design</h3>"),
        glzr, shgc, albroof
    ], layout=col_layout)

    display(widgets.VBox([
        city_dd,  # hidden
        widgets.HBox([urban_box, widgets.HTML(f"<div style='width:{GAP_PX}px;'></div>"), bldg_box]),
        buttons_box,
        out
    ]))


launch_ui()
