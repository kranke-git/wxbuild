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
from IPython.display import display, clear_output


WXBUILD_DATA_ROOT = "https://svante.mit.edu/~pgiani/wxbuild_data/"
DEFAULT_EXCLUDE_DIRS: Set[str] = {"cmip6", "~pgiani", "pgiani"}


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

    locations = list_locations_from_wxbuild(root=root, exclude=exclude, timeout=timeout)

    loc_dropdown = widgets.Dropdown(
        options=locations,
        value=locations[0],
        description=description,
        style={"description_width": "110px"},
        layout=widgets.Layout(width=dropdown_width),
    )
    refresh_btn = widgets.Button(description="Refresh", icon="refresh")
    load_btn = widgets.Button(description="Use this location", icon="play", button_style="primary")
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
            except Exception as e:
                print("❌ Failed to load EPW collections for this location.")
                print("Error:", e)

    refresh_btn.on_click(refresh_locations)
    load_btn.on_click(load_location_and_data)

    display(widgets.HBox([loc_dropdown, refresh_btn, load_btn]))
    display(out)

    return loc_dropdown, out
