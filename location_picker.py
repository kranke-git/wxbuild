# location_picker.py
"""Step 1 — Location Picker + Auto-load TMY/RMY/AMY on selection.

This module is designed for Google Colab/Jupyter teaching notebooks.

Key detail
----------
When code lives in an imported module, calling `globals()` writes to the module namespace,
NOT the notebook namespace. To write variables into the notebook, pass `target_globals=globals()`
from the notebook when calling `show_location_picker()`.

What it does
------------
- Scrapes the wxbuild_data directory listing for available locations.
- Shows a dropdown + refresh + "Use this location" button.
- On click, loads TMY/RMY/AMY via `epw_collection` and writes into `target_globals`:
    - location
    - current_tmy
    - current_rmy
    - current_amys

Dependencies
------------
- requests
- beautifulsoup4
- ipywidgets

Usage (in Colab)
----------------
from location_picker import show_location_picker
show_location_picker(epw_collection=epw_collection, target_globals=globals())
"""

from __future__ import annotations

from typing import Iterable, Optional, Set, Tuple, Dict, Any

import re  # <-- MINIMAL ADDITION

import requests
from bs4 import BeautifulSoup

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML


WXBUILD_DATA_ROOT = "https://svante.mit.edu/~pgiani/wxbuild_data/"
DEFAULT_EXCLUDE_DIRS: Set[str] = {"cmip6", "~pgiani", "pgiani"}


def _inject_button_css(button_min_width: str) -> None:
    """Inject CSS to widen ipywidget buttons so long labels are fully visible.

    This is intentionally global (within the notebook) so it also affects
    scenario buttons rendered by other modules/cells.
    """
    css = f"""
    <style>
      /* Make widget buttons wide enough to show full filenames */
      .jupyter-widgets .widget-button {{
        min-width: {button_min_width} !important;
        width: auto !important;
      }}
      .jupyter-widgets .widget-button .widget-label {{
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        max-width: none !important;
      }}
    </style>
    """
    display(HTML(css))


def list_locations_from_wxbuild(
    root: str = WXBUILD_DATA_ROOT,
    exclude: Optional[Iterable[str]] = None,
    timeout: int = 20,
) -> list[str]:
    """Return location folder keys found under the wxbuild data root."""
    exclude_set = set(exclude) if exclude is not None else set(DEFAULT_EXCLUDE_DIRS)

    r = requests.get(root, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    folders: list[str] = []
    for a in soup.find_all("a"):
        href = a.get("href", "")
        if href.endswith("/") and href not in ("../", "./"):
            name = href[:-1].strip()
            folders.append(name)

    # MINIMAL ADDITION: accept City_CC (e.g., Budapest_HU) in addition to legacy formats
    city_cc_pat = re.compile(r"^.+_[A-Z]{2,3}$")  # one underscore, 2–3 uppercase letters at end

    locations = sorted(
        f for f in folders
        if f
        and f not in exclude_set
        and not f.startswith("~")
        and (
            ("__" in f)                  # legacy: Boston__MA__USA
            or (f.count("_") >= 2)       # legacy-ish: City_State_Country
            or bool(city_cc_pat.match(f))# NEW: Budapest_HU
        )
    )

    if not locations:
        raise RuntimeError(f"No locations found at {root}. Folders seen: {folders}")

    return locations


def show_location_picker(
    *,
    epw_collection=None,
    target_globals: Optional[Dict[str, Any]] = None,
    root: str = WXBUILD_DATA_ROOT,
    exclude: Optional[Iterable[str]] = None,
    timeout: int = 20,
    dropdown_width: str = "520px",
    button_min_width: str = "340px",
    description: str = "Location",
) -> Tuple[widgets.Dropdown, widgets.Output]:
    """Display the location picker widget and auto-load EPW collections on selection.

    Parameters
    ----------
    epw_collection:
        Pass `epw_collection` OR leave None to import from `epwclass`.
    target_globals:
        Dict-like namespace to write outputs into (pass `globals()` from the notebook).
    """
    if epw_collection is None:
        from epwclass import epw_collection as _epw_collection  # type: ignore
        epw_collection = _epw_collection

    # IMPORTANT: default to module globals if user doesn't pass a namespace
    if target_globals is None:
        target_globals = globals()

    # Ensure long scenario/file buttons across the notebook render at full width.
    _inject_button_css(button_min_width)

    locations = list_locations_from_wxbuild(root=root, exclude=exclude, timeout=timeout)

    loc_dropdown = widgets.Dropdown(
        options=locations,
        value=locations[0],
        description=description,
        style={"description_width": "110px"},
        layout=widgets.Layout(width=dropdown_width),
    )
    # Widen buttons so long labels (e.g., scenario/file names in downstream UI) are fully visible.
    refresh_btn = widgets.Button(
        description="Refresh",
        icon="refresh",
        layout=widgets.Layout(min_width=button_min_width),
    )
    load_btn = widgets.Button(
        description="Use this location",
        icon="play",
        button_style="primary",
        layout=widgets.Layout(min_width=button_min_width),
    )
    out = widgets.Output()

    def refresh_locations(_):
        with out:
            clear_output()
            try:
                new_locs = list_locations_from_wxbuild(root=root, exclude=exclude, timeout=timeout)
                loc_dropdown.options = new_locs
                if loc_dropdown.value not in new_locs:
                    loc_dropdown.value = new_locs[0]
                print(f"✅ Refreshed. Found {len(new_locs)} location(s).")
            except Exception as e:
                print("❌ Refresh failed:", e)

    def load_location_and_data(_):
        with out:
            clear_output()
            selected = str(loc_dropdown.value)
            print("✅ Selected location:", selected)

            try:
                current_tmy = epw_collection(filetype="tmy", location=selected)
                current_rmy = epw_collection(filetype="rmy", location=selected)
                current_amys = epw_collection(filetype="amy", location=selected)

                # Write to notebook namespace
                target_globals["location"] = selected
                target_globals["current_tmy"] = current_tmy
                target_globals["current_rmy"] = current_rmy
                target_globals["current_amys"] = current_amys

                print("✅ Loaded collections:")
                print("  TMY files:", len(getattr(current_tmy, "files", [])))
                print("  RMY files:", len(getattr(current_rmy, "files", [])))
                print("  AMY files:", len(getattr(current_amys, "files", [])))

                # --- Minimal UI add-on: SSP subheadings (SSP 1-26, 2-45, 5-85) ---
                # If FRMY scenario files are present in the RMY collection, show them grouped
                # under SSP headings, with wide buttons to display full names.
                files = list(getattr(current_rmy, "files", []) or [])
                if files:
                    def _basename(x):
                        try:
                            return str(x).split("/")[-1]
                        except Exception:
                            return str(x)

                    ssp_groups = {
                        "SSP 1-26": [f for f in files if "SSP126" in _basename(f) or "ssp126" in _basename(f)],
                        "SSP 2-45": [f for f in files if "SSP245" in _basename(f) or "ssp245" in _basename(f)],
                        "SSP 5-85": [f for f in files if "SSP585" in _basename(f) or "ssp585" in _basename(f)],
                    }

                    # Only render headings if at least one group has files.
                    if any(ssp_groups.values()):
                        display(widgets.HTML("<br><b>FRMY files detected (grouped by SSP)</b>"))
                        for heading, group in ssp_groups.items():
                            if not group:
                                continue
                            display(widgets.HTML(f"<br><b>{heading}</b>"))
                            btns = []
                            for f in group:
                                btns.append(
                                    widgets.Button(
                                        description=_basename(f),
                                        layout=widgets.Layout(min_width=button_min_width),
                                    )
                                )
                            display(widgets.VBox(btns))
            except Exception as e:
                print("❌ Failed to load EPW collections for this location.")
                print("Error:", e)

    refresh_btn.on_click(refresh_locations)
    load_btn.on_click(load_location_and_data)

    display(widgets.HBox([loc_dropdown, refresh_btn, load_btn]))
    display(out)

    return loc_dropdown, out
