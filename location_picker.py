# location_picker.py
"""Step 1 — Location Picker + Auto-load TMY/RMY/AMY on selection.

This module is designed for Google Colab/Jupyter teaching notebooks.

What it does
------------
- Scrapes the wxbuild_data directory listing for available locations.
- Shows a dropdown + refresh + "Use this location" button.
- On click, loads TMY/RMY/AMY via `epw_collection` and writes into notebook globals:
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
show_location_picker(epw_collection=epw_collection)
"""

from __future__ import annotations

from typing import Iterable, Optional, Set, Tuple

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

    locations = sorted(
        f for f in folders
        if f
        and f not in exclude_set
        and not f.startswith("~")
        and "__" in f  # enforce location-style keys like Boston__MA__USA
    )

    if not locations:
        raise RuntimeError(f"No locations found at {root}. Folders seen: {folders}")

    return locations


def show_location_picker(
    *,
    epw_collection=None,
    root: str = WXBUILD_DATA_ROOT,
    exclude: Optional[Iterable[str]] = None,
    timeout: int = 20,
    dropdown_width: str = "520px",
    description: str = "Location",
) -> Tuple[widgets.Dropdown, widgets.Output]:
    """Display the location picker widget and auto-load EPW collections on selection."""
    if epw_collection is None:
        from epwclass import epw_collection as _epw_collection  # type: ignore
        epw_collection = _epw_collection

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

                g = globals()
                g["location"] = selected
                g["current_tmy"] = current_tmy
                g["current_rmy"] = current_rmy
                g["current_amys"] = current_amys

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
