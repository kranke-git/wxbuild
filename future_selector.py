# future_selector.py
"""Step 2 — Future selector (Scenario + Horizon) + Auto-generate FAMYs.

This module is designed for Google Colab/Jupyter teaching notebooks.

What it does
------------
- Displays two controls:
  1) Emissions scenario (toggle buttons)
  2) Time horizon (slider with discrete steps)
- On click, calls `current_amys.with_futureShifts(...)` to generate/save future AMYs (FAMYs).
- Writes into notebook globals:
    - experiment (e.g., 'ssp585')
    - futyear (e.g., 2050)
    - future_amys

Dependencies
------------
- ipywidgets
- IPython

Usage (in Colab)
----------------
from future_selector import show_future_selector
show_future_selector(current_amys=current_amys)  # current_amys loaded in Step 1
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import ipywidgets as widgets
from IPython.display import display, clear_output


DEFAULT_SCENARIOS: List[Tuple[str, str]] = [
    ("SSP1-2.6", "ssp126"),
    ("SSP2-4.5", "ssp245"),
    ("SSP3-7.0", "ssp370"),
    ("SSP5-8.5", "ssp585"),
]

DEFAULT_HORIZONS: List[int] = [2050, 2075, 2100]


def show_future_selector(
    *,
    current_amys=None,
    model: str = "CanESM5",
    scenarios: Sequence[Tuple[str, str]] = tuple(DEFAULT_SCENARIOS),
    horizons: Sequence[int] = tuple(DEFAULT_HORIZONS),
    default_scenario: str = "ssp585",
    default_horizon: int = 2050,
    saveflag: bool = True,
    width: str = "520px",
    scenario_label: str = "Scenario",
    horizon_label: str = "Horizon",
):
    """Display scenario + horizon controls and generate FAMYs on click.

    Parameters
    ----------
    current_amys:
        epw_collection for AMYs (recommended). If None, will try globals()['current_amys'].
    model:
        Fixed CMIP6 model used for future shifts (e.g., 'CanESM5').
    scenarios:
        List of (label, value) tuples for toggle buttons.
    horizons:
        Discrete list of year options for the slider.
    default_scenario/default_horizon:
        Initial UI defaults.
    saveflag:
        Passed through to `with_futureShifts(..., saveflag=...)`.
    width:
        Widget width.
    """
    if current_amys is None:
        g = globals()
        if "current_amys" not in g:
            raise RuntimeError("current_amys not found. Run Step 1 first (Location Picker + load).")
        current_amys = g["current_amys"]

    if default_scenario not in [v for _, v in scenarios]:
        default_scenario = scenarios[-1][1]
    if default_horizon not in list(horizons):
        default_horizon = list(horizons)[0]

    scenario_w = widgets.ToggleButtons(
        options=list(scenarios),
        value=default_scenario,
        description=scenario_label,
        style={"description_width": "110px"},
        layout=widgets.Layout(width=width),
    )

    year_w = widgets.SelectionSlider(
        options=list(horizons),
        value=default_horizon,
        description=horizon_label,
        continuous_update=False,
        style={"description_width": "110px"},
        layout=widgets.Layout(width=width),
    )

    run_btn = widgets.Button(description="Generate FAMYs", icon="play", button_style="primary")
    out = widgets.Output()

    def _generate(_):
        with out:
            clear_output()

            futyear = int(year_w.value)
            experiment = str(scenario_w.value)

            g = globals()
            g["futyear"] = futyear
            g["experiment"] = experiment

            print(f"✅ Selected: {experiment}, {futyear}")
            print(f"Generating future AMYs using model={model} (this may take a moment)…")

            try:
                future_amys = current_amys.with_futureShifts(
                    params={"model": model, "futexp": experiment, "futyear": futyear},
                    saveflag=saveflag,
                )

                g["future_amys"] = future_amys
                n_files = len(getattr(future_amys, "files", []))
                print(f"✅ future_amys created. Files: {n_files}")
            except Exception as e:
                print("❌ Failed to generate future AMYs.")
                print("Error:", e)

    run_btn.on_click(_generate)

    display(widgets.VBox([scenario_w, year_w, run_btn, out]))
    return scenario_w, year_w, out
