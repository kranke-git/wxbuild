# uwg_ui_residential.py
from __future__ import annotations

import os
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from google.colab import output

from uwg import UWG

from uwg_setup_utils import (
    detect_cities,
    find_rmy_epw,
    ensure_city_uwg_dir,
    patch_readDOE_mass_wall_roof,
    build_initialize_residential,
    set_key_line_in_file,
    infer_uwg_zone_from_epw,
)


# Enable widgets in Colab
output.enable_custom_widget_manager()


# ----------------------------
# UI / layout constants
# ----------------------------
style = {"description_width": "220px"}
slider_layout = widgets.Layout(width="680px")
col_layout = widgets.Layout(width="720px")


# ----------------------------
# Template (initialize_residential.uwg)
# ----------------------------
PARAM_UWG = """# =================================================
# Sample UWGv4.2 simulation initialization parameters
# Chris Mackey,2017
# =================================================

# =================================================
# REQUIRED PARAMETERS
# =================================================

# Urban characteristics
bldHeight,31.0,
bldDensity,0.25,
verToHor,0.55,
h_mix,1,           # fraction of building HVAC waste heat set to the street canyon [as opposed to the roof]
charLength,1000,  # dimension of a square that encompasses the whole neighborhood [aka. characteristic length] (m)
albRoad,0.12,
dRoad,0.5,        # road pavement thickness (m)
kRoad,1,          # road pavement conductivity (W/m K)
cRoad,1600000,    # road volumetric heat capacity (J/m^3 K)
sensAnth,1.0,

# Climate Zone (Eg. City)
# 1A(Miami)
# 2A(Houston)
# 2B(Phoenix)
# 3A(Atlanta)
# 3B-CA(Los Angeles)
# 3B(Las Vegas)
# 3C(San Francisco)
# 4A(Baltimore)
# 4B(Albuquerque)
# 4C(Seattle)
# 5A(Chicago)
# 5B(Boulder)
# 6A(Minneapolis)
# 6B(Helena)
# 7(Duluth)
# 8(Fairbanks)

zone,5A,

# Vegetation parameters
grasscover,0.3,
treeCover,0.25,
vegStart,4,       # The month in which vegetation starts to evapotranspire (leaves are out)
vegEnd,10,        # The month in which vegetation stops evapotranspiring (leaves fall)
albVeg,0.25,      # Vegetation albedo
latGrss,0.4,      # Fraction of the heat absorbed by grass that is latent (goes to evaporating water)
latTree,0.6,      # Fraction of the heat absorbed by trees that is latent (goes to evaporating water)
rurVegCover,0.9,  # Fraction of the rural ground covered by vegetation

# Traffic schedule [1 to 24 hour],
SchTraffic,
0.2,0.2,0.2,0.2,0.2,0.4,0.7,0.9,0.9,0.6,0.6,0.6,0.6,0.6,0.7,0.8,0.9,0.9,0.8,0.8,0.7,0.3,0.2,0.2, # Weekday
0.2,0.2,0.2,0.2,0.2,0.3,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.6,0.7,0.7,0.7,0.7,0.5,0.4,0.3,0.2,0.2, # Saturday
0.2,0.2,0.2,0.2,0.2,0.3,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.3,0.3,0.2,0.2, # Sunday

# Fraction of building stock for each DOE Building type (pre-80's build, 80's-present build, new)
# Note that sum(bld) must be equal to 1
# ...
"""


# ----------------------------
# Build UI
# ----------------------------
def build_ui(epwdata_root: str = "/content/wxbuild/epwdata"):
    cities = detect_cities(epwdata_root)
    if not cities:
        display(widgets.HTML("<b>No cities found under epwdata.</b>"))
        return

    city_dd = widgets.Dropdown(
        options=cities,
        value=cities[0],
        description="City",
        style=style,
        layout=widgets.Layout(width="520px"),
    )

    base_dd = widgets.Dropdown(
        options=[],
        description="EPW (baseline)",
        style=style,
        layout=widgets.Layout(width="720px"),
    )

    def refresh_epw_options(*_):
        city_root = Path(epwdata_root) / city_dd.value
        epw = find_rmy_epw(str(city_root))
        if epw is None:
            base_dd.options = []
            base_dd.value = None
        else:
            base_dd.options = [epw]
            base_dd.value = epw

    refresh_epw_options()
    city_dd.observe(refresh_epw_options, names="value")

    # Urban sliders
    bldheight = widgets.FloatSlider(
        value=31.0, min=3.0, max=80.0, step=1.0,
        description="Building Height (m)",
        style=style, layout=slider_layout, continuous_update=False
    )
    blddensity = widgets.FloatSlider(
        value=0.25, min=0.05, max=0.90, step=0.01,
        description="Footprint Density (0–1)",
        style=style, layout=slider_layout, continuous_update=False
    )
    vertohor = widgets.FloatSlider(
        value=0.55, min=0.10, max=3.00, step=0.01,
        description="Canyon H/W Proxy (verToHor)",
        style=style, layout=slider_layout, continuous_update=False
    )
    grasscover = widgets.FloatSlider(
        value=0.30, min=0.0, max=0.90, step=0.01,
        description="Grass Cover (0–1)",
        style=style, layout=slider_layout, continuous_update=False
    )
    treecover = widgets.FloatSlider(
        value=0.25, min=0.0, max=0.90, step=0.01,
        description="Tree Cover (0–1)",
        style=style, layout=slider_layout, continuous_update=False
    )
    albroad = widgets.FloatSlider(
        value=0.12, min=0.02, max=0.50, step=0.01,
        description="Road Albedo (albRoad)",
        style=style, layout=slider_layout, continuous_update=False
    )
    sensanth = widgets.FloatSlider(
        value=1.0, min=0.0, max=30.0, step=0.5,
        description="Traffic / Anthropogenic Heat (W/m²)",
        style=style, layout=slider_layout, continuous_update=False
    )

    # --- climate zone (auto-infer from the selected EPW, with manual override) ---
    auto_zone = widgets.Checkbox(value=True, description="Auto climate zone (infer from EPW)", indent=False)
    zone_dd = widgets.Dropdown(
        options=["1A","2A","2B","3A","3B","3C","4A","4B","4C","5A","5B","6A","6B","7","8"],
        value="5A",
        description="Manual zone",
        style=style,
        layout=widgets.Layout(width="420px"),
    )
    zone_note = widgets.HTML("")

    def refresh_zone_note(*_):
        epw = base_dd.value
        if not epw:
            zone_note.value = "<i>No EPW selected.</i>"
            return
        z = infer_uwg_zone_from_epw(epw, fallback=zone_dd.value)
        zone_note.value = f"<b>Inferred zone from EPW:</b> {z}"
    refresh_zone_note()
    base_dd.observe(refresh_zone_note, names="value")
    zone_dd.observe(refresh_zone_note, names="value")

    # Button
    run_btn = widgets.Button(description="Generate", button_style="primary")

    out = widgets.Output()

    # Layout
    urban_box = widgets.VBox([
        widgets.HTML("<h3>Urban Design</h3>"),
        bldheight,
        blddensity,
        vertohor,
        grasscover,
        treecover,
        albroad,
        sensanth,

        widgets.HTML("<h4 style='margin:10px 0 4px 0;'>Climate Zone</h4>"),
        auto_zone,
        zone_dd,
        zone_note,
    ], layout=col_layout)

    display(widgets.VBox([
        widgets.HTML("<h2>UWG Residential UI</h2>"),
        widgets.HBox([city_dd]),
        base_dd,
        urban_box,
        run_btn,
        out
    ]))

    # ----------------------------
    # Run handler
    # ----------------------------
    def on_run_clicked(_):
        with out:
            clear_output(wait=True)
            try:
                city = city_dd.value
                epw_path = base_dd.value
                if not epw_path:
                    print("❌ No EPW selected.")
                    return

                city_root = Path(epwdata_root) / city
                uwg_dir = ensure_city_uwg_dir(str(city_root))

                print("✅ Clicked Generate")
                print("City:", city)
                print("Original EPW:", epw_path)
                print("UWG output folder:", uwg_dir)

                # Create initialize_residential.uwg from template (keeps your existing workflow)
                param_path = Path(uwg_dir) / "initialize_residential.uwg"
                build_initialize_residential(str(param_path), PARAM_UWG)

                # Apply edits (keeping the same key replacement behavior)
                set_key_line_in_file(param_path, "bldHeight,",   f"bldHeight,{bldheight.value},")
                set_key_line_in_file(param_path, "bldDensity,",  f"bldDensity,{blddensity.value},")
                set_key_line_in_file(param_path, "verToHor,",    f"verToHor,{vertohor.value},")
                set_key_line_in_file(param_path, "grasscover,",  f"grasscover,{grasscover.value},")
                set_key_line_in_file(param_path, "treeCover,",   f"treeCover,{treecover.value},")
                set_key_line_in_file(param_path, "albRoad,",     f"albRoad,{albroad.value},")
                set_key_line_in_file(param_path, "sensAnth,",    f"sensAnth,{sensanth.value},")

                # Climate zone
                if auto_zone.value:
                    zone_val = infer_uwg_zone_from_epw(epw_path, fallback=zone_dd.value)
                else:
                    zone_val = zone_dd.value
                set_key_line_in_file(param_path, "zone,", f"zone,{zone_val},")
                print("UWG zone:", zone_val)

                # Optional patch (best effort)
                patched = patch_readDOE_mass_wall_roof(str(Path(__file__).parent))
                if patched:
                    print("✅ Patched readDOE.py")
                else:
                    # keep your warning behavior
                    print("⚠️ readDOE.py: no replacements made (patterns not found).")
                    print("Param file:", str(param_path))

                # Run UWG from param file
                model = UWG.from_param_file(str(param_path), epw_path=epw_path)

                # You want annual output
                model.generate()
                model.simulate()
                model.write_epw()

                out_epw = model.new_epw_path
                print("✅ UWG EPW written:", out_epw)

                # quick sanity plot (optional, minimal)
                try:
                    import pandas as pd
                    df = pd.read_csv(out_epw, skiprows=8, header=None)
                    t = pd.to_numeric(df.iloc[:, 6], errors="coerce")
                    plt.figure(figsize=(12,3))
                    plt.plot(t.values[:24*7])
                    plt.title("UWG Dry-bulb (first week)")
                    plt.ylabel("°C")
                    plt.grid(True)
                    plt.show()
                except Exception:
                    pass

            except Exception:
                print("❌ UWG run failed. Full traceback:\n")
                print(traceback.format_exc())

    run_btn.on_click(on_run_clicked)


# For Colab: call build_ui() at import-time if desired
if __name__ == "__main__":
    build_ui()

