"""wxbuild future selector UI helpers.

Updated (single-button) UI:
- One button: "Generate all futures"
- Generates future AMYs (FAMYs) for all SSP scenarios and all years from start_year to end_year.
- Displays a separate progress bar and status line for each scenario.

Notebook usage:

    from future_selector import show_generate_all_futures
    show_generate_all_futures(target_globals=globals())

The generated results are written into the notebook namespace:
- futures_all: dict[scenario_code][year] -> future_amys object
- future_scenarios, future_years, future_model

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import ipywidgets as widgets
from IPython.display import display, clear_output


DEFAULT_SCENARIOS: List[Tuple[str, str]] = [
    ("SSP1-2.6", "ssp126"),
    ("SSP2-4.5", "ssp245"),
    ("SSP3-7.0", "ssp370"),
    ("SSP5-8.5", "ssp585"),
]


def show_generate_all_futures(
    *,
    current_amys=None,
    target_globals: Optional[Dict[str, Any]] = None,
    model: str = "CanESM5",
    scenarios: Sequence[Tuple[str, str]] = tuple(DEFAULT_SCENARIOS),
    # "from now until 2100" -> defaults chosen for performance + practicality
    start_year: int = 2030,
    end_year: int = 2100,
    step_years: int = 5,
    saveflag: bool = True,
    width: str = "720px",
):
    """One-button generator for all futures (all scenarios × all years).

    Parameters
    ----------
    current_amys:
        wxbuild AMY collection (must support .with_futureShifts).
        If None, will read from target_globals['current_amys'].
    target_globals:
        Notebook globals() so results can be written back.
    model:
        GCM/emulator model string passed to with_futureShifts.
    scenarios:
        List of tuples (label, scenario_code) e.g., ("SSP5-8.5", "ssp585").
    start_year, end_year, step_years:
        Year range for future shifts.
    saveflag:
        Whether to persist generated EPWs to disk (wxbuild behavior).
    width:
        UI width.

    Returns
    -------
    (run_button, output_widget)
    """

    if target_globals is None:
        target_globals = globals()

    if current_amys is None:
        if "current_amys" not in target_globals:
            raise RuntimeError(
                "current_amys not found. Run Step 1 first (Location Picker + load)."
            )
        current_amys = target_globals["current_amys"]

    years = list(range(int(start_year), int(end_year) + 1, int(step_years)))
    if not years:
        raise ValueError("No years to generate. Check start_year/end_year/step_years.")

    # --- UI ---
    title = widgets.HTML(
        value=(
            "<b>Generate all futures</b><br>"
            f"Years: {years[0]}–{years[-1]} (step {step_years}) · Model: {model}"
        )
    )

    run_btn = widgets.Button(
        description="Generate all futures",
        icon="play",
        button_style="primary",
        layout=widgets.Layout(width="240px"),
    )

    out = widgets.Output(layout=widgets.Layout(width=width))

    pbars: Dict[str, widgets.IntProgress] = {}
    stat_lbls: Dict[str, widgets.HTML] = {}

    rows = []
    for label, code in scenarios:
        p = widgets.IntProgress(
            value=0,
            min=0,
            max=len(years),
            description=label,
            bar_style="",
            layout=widgets.Layout(width=width),
            style={"description_width": "120px"},
        )
        s = widgets.HTML(value="Ready")
        pbars[code] = p
        stat_lbls[code] = s
        rows.append(widgets.VBox([p, s], layout=widgets.Layout(margin="0 0 8px 0")))

    grid = widgets.VBox(rows)

    def _reset_ui():
        for _, code in scenarios:
            pbars[code].value = 0
            pbars[code].bar_style = ""
            stat_lbls[code].value = "Ready"

    def _set_running(running: bool):
        run_btn.disabled = running

    def _generate_all(_):
        _set_running(True)
        _reset_ui()

        futures_all: Dict[str, Dict[int, Any]] = {}

        with out:
            clear_output()
            print("▶ Generating all futures…")
            print(f"Scenarios: {[c for _, c in scenarios]}")
            print(f"Years: {years}\n")

        try:
            for label, code in scenarios:
                futures_all[code] = {}
                pbars[code].bar_style = "info"
                stat_lbls[code].value = f"Running… ({label})"

                for i, yr in enumerate(years, start=1):
                    stat_lbls[code].value = f"Running… {label} · {yr}"

                    # this call is the core wxbuild API
                    future_amys = current_amys.with_futureShifts(
                        params={"model": model, "futexp": code, "futyear": int(yr)},
                        saveflag=saveflag,
                    )

                    futures_all[code][int(yr)] = future_amys
                    pbars[code].value = i

                pbars[code].bar_style = "success"
                stat_lbls[code].value = f"✅ Done ({label})"

            # write results back into the notebook namespace
            target_globals["futures_all"] = futures_all
            target_globals["future_scenarios"] = [c for _, c in scenarios]
            target_globals["future_years"] = years
            target_globals["future_model"] = model

            with out:
                print("\n✅ All futures generated.")
                print("Saved to: futures_all[scenario][year]")

        except Exception as e:
            # mark any in-progress scenario as failed
            for _, code in scenarios:
                if pbars[code].bar_style == "info":
                    pbars[code].bar_style = "danger"
                    stat_lbls[code].value = "❌ Failed"
            with out:
                print("\n❌ Failed during generation.")
                print("Error:", e)

            # still save any partial results
            target_globals["futures_all"] = futures_all
            target_globals["future_scenarios"] = [c for _, c in scenarios]
            target_globals["future_years"] = years
            target_globals["future_model"] = model

        finally:
            _set_running(False)

    run_btn.on_click(_generate_all)

    display(widgets.VBox([title, run_btn, grid, out]))
    return run_btn, out


# Backward compatibility: if older notebooks import/show_future_selector, route them
# to the single-button UI instead of presenting scenario/year controls.

def show_future_selector(*, target_globals: Optional[Dict[str, Any]] = None, **kwargs):
    """Deprecated: retained for older notebooks.

    This now renders the single-button UI (Generate all futures).
    Any kwargs are forwarded to show_generate_all_futures.
    """

    return show_generate_all_futures(target_globals=target_globals, **kwargs)
