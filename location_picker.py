# location_picker.py
# Minimal, GitHub-friendly location picker that discovers city folders under epw_collection.root_dir
# Supports BOTH formats:
#   - Legacy:   City__State__Country    (e.g., Boston__MA__USA)
#   - New:      City_CountryCode        (e.g., Budapest_HU)
#
# Usage in Colab:
#   from location_picker import show_location_picker
#   show_location_picker(epw_collection=epw_collection, target_globals=globals())

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

try:
    import ipywidgets as widgets
    from IPython.display import display
except Exception:
    widgets = None
    display = None


# Folders that should never be treated as "locations"
DEFAULT_EXCLUDE_DIRS = {
    "cmip6",
    ".ipynb_checkpoints",
    "__pycache__",
    ".git",
    ".github",
    "plots",
    "docs",
}


# Recognize BOTH location folder formats:
#   1) City__State__Country  (legacy)
#   2) City_CountryCode      (newer, like Budapest_HU)
LOCATION_DIR_PATTERNS = [
    re.compile(r".+__.+__.+$"),      # legacy: Boston__MA__USA
    re.compile(r".+_[A-Z]{2,3}$"),   # new: Budapest_HU, Nairobi_KE, etc.
]


def _is_location_dir(name: str, exclude_dirs: set) -> bool:
    if not name or name in exclude_dirs:
        return False
    # ignore hidden dirs
    if name.startswith("."):
        return False
    # must match at least one pattern
    return any(pat.match(name) for pat in LOCATION_DIR_PATTERNS)


def list_locations_from_wxbuild(root_dir: str, exclude_dirs: Optional[set] = None) -> List[str]:
    """
    Returns a sorted list of location directory names found directly under root_dir.
    """
    exclude_dirs = set(exclude_dirs or DEFAULT_EXCLUDE_DIRS)

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    names = []
    for n in os.listdir(root_dir):
        p = os.path.join(root_dir, n)
        if os.path.isdir(p) and _is_location_dir(n, exclude_dirs):
            names.append(n)

    # sort for nice UI
    names.sort(key=lambda s: s.lower())
    return names


def _pretty_location_label(loc_dirname: str) -> str:
    """
    Makes a friendly dropdown label.
    Examples:
      Boston__MA__USA  -> Boston, MA, USA
      Budapest_HU      -> Budapest, HU
    """
    if "__" in loc_dirname:
        parts = loc_dirname.split("__")
        return ", ".join(parts)
    if "_" in loc_dirname:
        parts = loc_dirname.split("_")
        return ", ".join(parts)
    return loc_dirname


def _infer_root_dir(epw_collection=None, root_dir: Optional[str] = None) -> str:
    """
    Tries to infer root directory from an epw_collection (common patterns),
    or uses explicit root_dir.
    """
    if root_dir:
        return root_dir

    if epw_collection is None:
        # sensible default for your Colab layout
        return "/content/wxbuild/epwdata"

    # Common attribute names used in notebooks/repos
    for attr in ("root_dir", "root", "base_dir", "epw_root", "data_root", "epwdata_root"):
        if hasattr(epw_collection, attr):
            val = getattr(epw_collection, attr)
            if isinstance(val, str) and val:
                return val

    # Fallback to standard path
    return "/content/wxbuild/epwdata"


def show_location_picker(
    epw_collection=None,
    root_dir: Optional[str] = None,
    exclude_dirs: Optional[set] = None,
    target_globals: Optional[Dict] = None,
    target_key: str = "CITY",
):
    """
    Displays an ipywidgets dropdown listing all location folders.
    On selection, stores:
      - target_globals[target_key] = selected folder name (e.g., "Budapest_HU")
    And prints the selected directory.

    Parameters
    ----------
    epw_collection : optional
        Your EPWCollection object if you have one.
    root_dir : str, optional
        Override root directory to scan. If None, inferred from epw_collection or defaults.
    exclude_dirs : set, optional
        Extra directories to exclude.
    target_globals : dict, optional
        Pass globals() from your notebook if you want the selection written into a variable.
    target_key : str
        Variable name to store selection into (default "CITY").
    """
    if widgets is None or display is None:
        raise ImportError("ipywidgets not available. Install/enable ipywidgets in your notebook environment.")

    root = _infer_root_dir(epw_collection=epw_collection, root_dir=root_dir)
    ex = set(DEFAULT_EXCLUDE_DIRS)
    if exclude_dirs:
        ex |= set(exclude_dirs)

    locations = list_locations_from_wxbuild(root, exclude_dirs=ex)
    if not locations:
        raise RuntimeError(
            f"No location folders found under: {root}\n"
            f"Checked for patterns: City__State__Country OR City_CountryCode (e.g., Budapest_HU)\n"
            f"Excluded: {sorted(ex)}"
        )

    # Build label -> folder mapping
    label_map: List[Tuple[str, str]] = [(_pretty_location_label(x), x) for x in locations]

    dropdown = widgets.Dropdown(
        options=label_map,
        description="City:",
        value=locations[0],
        layout=widgets.Layout(width="420px"),
    )

    out = widgets.Output()

    def _on_change(change):
        if change.get("name") != "value":
            return
        selection = change["new"]
        with out:
            out.clear_output()
            print(f"Selected: {selection}")
            print(f"Path: {os.path.join(root, selection)}")
        if target_globals is not None:
            target_globals[target_key] = selection

    dropdown.observe(_on_change, names="value")

    # initial print
    _on_change({"name": "value", "new": dropdown.value})

    display(widgets.VBox([dropdown, out]))
    return dropdown
