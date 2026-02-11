# location_picker.py
# Location picker that auto-detects the EPW root directory and lists city folders.
# Supports BOTH formats:
#   - Legacy: City__State__Country  (e.g., Boston__MA__USA)
#   - New:    City_CountryCode      (e.g., Budapest_HU)
#
# Usage (same as before):
#   from location_picker import show_location_picker
#   show_location_picker(target_globals=globals())
#
# Optional:
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


DEFAULT_EXCLUDE_DIRS = {
    "cmip6",
    ".ipynb_checkpoints",
    "__pycache__",
    ".git",
    ".github",
    "plots",
    "docs",
}

# Recognize BOTH location folder formats
LOCATION_DIR_PATTERNS = [
    re.compile(r".+__.+__.+$"),      # legacy: Boston__MA__USA
    re.compile(r".+_[A-Z]{2,3}$"),   # new: Budapest_HU
]

# Candidate roots to try (ordered)
ROOT_CANDIDATES = [
    # Most common wxbuild layouts
    "/content/wxbuild/epwdata",
    "/content/wxbuild_data",
    "/content/wxbuild_data/epwdata",
    "/content/drive/MyDrive/wxbuild_data",
    "/content/drive/MyDrive/wxbuild_data/epwdata",
    # Some repos mount under /content/epwdata or /content/data
    "/content/epwdata",
    "/content/data/epwdata",
    "/content/data",
]


def _is_location_dir(name: str, exclude_dirs: set) -> bool:
    if not name or name in exclude_dirs:
        return False
    if name.startswith("."):
        return False
    return any(pat.match(name) for pat in LOCATION_DIR_PATTERNS)


def _find_existing_root(epw_collection=None, root_dir: Optional[str] = None) -> str:
    """
    Auto-detect a valid root directory that contains city folders.
    Priority:
      1) explicit root_dir if provided
      2) epw_collection attributes (root_dir/root/base_dir/...)
      3) environment variables
      4) known candidate paths
    """
    # 1) explicit
    if root_dir and os.path.isdir(root_dir):
        return root_dir

    # 2) epw_collection common attrs
    if epw_collection is not None:
        for attr in ("root_dir", "root", "base_dir", "epw_root", "data_root", "epwdata_root"):
            if hasattr(epw_collection, attr):
                val = getattr(epw_collection, attr)
                if isinstance(val, str) and val and os.path.isdir(val):
                    return val

    # 3) environment hints
    for env_key in ("WXBUILD_EPWDATAROOT", "EPWDATA_ROOT", "WXBUILD_ROOT"):
        val = os.environ.get(env_key, "")
        if val and os.path.isdir(val):
            return val

    # 4) known candidates
    for cand in ROOT_CANDIDATES:
        if os.path.isdir(cand):
            return cand

    # Nothing found
    tried = []
    if root_dir:
        tried.append(root_dir)
    tried.extend(ROOT_CANDIDATES)
    tried_txt = "\n  - " + "\n  - ".join(tried)
    raise FileNotFoundError(
        "Could not find the EPW root directory.\n"
        "Tried these locations:" + tried_txt + "\n\n"
        "Fix: pass root_dir explicitly, e.g.\n"
        "  show_location_picker(root_dir='/path/to/epwdata', target_globals=globals())"
    )


def list_locations_from_wxbuild(root_dir: str, exclude_dirs: Optional[set] = None) -> List[str]:
    exclude_dirs = set(exclude_dirs or DEFAULT_EXCLUDE_DIRS)

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    names: List[str] = []
    for n in os.listdir(root_dir):
        p = os.path.join(root_dir, n)
        if os.path.isdir(p) and _is_location_dir(n, exclude_dirs):
            names.append(n)

    names.sort(key=lambda s: s.lower())
    return names


def _pretty_location_label(loc_dirname: str) -> str:
    if "__" in loc_dirname:
        return ", ".join(loc_dirname.split("__"))
    if "_" in loc_dirname:
        return ", ".join(loc_dirname.split("_"))
    return loc_dirname


def show_location_picker(
    epw_collection=None,
    root_dir: Optional[str] = None,
    exclude_dirs: Optional[set] = None,
    target_globals: Optional[Dict] = None,
    target_key: str = "CITY",
):
    """
    Displays a dropdown listing all location folders found under the EPW root.
    Writes selection into target_globals[target_key] if provided.
    """
    if widgets is None or display is None:
        raise ImportError("ipywidgets not available in this environment.")

    root = _find_existing_root(epw_collection=epw_collection, root_dir=root_dir)

    ex = set(DEFAULT_EXCLUDE_DIRS)
    if exclude_dirs:
        ex |= set(exclude_dirs)

    locations = list_locations_from_wxbuild(root, exclude_dirs=ex)
    if not locations:
        raise RuntimeError(
            f"No location folders found under: {root}\n"
            f"Expected folder formats like Boston__MA__USA or Budapest_HU.\n"
            f"Excluded: {sorted(ex)}"
        )

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
            print(f"Root: {root}")
            print(f"Path: {os.path.join(root, selection)}")
        if target_globals is not None:
            target_globals[target_key] = selection

    dropdown.observe(_on_change, names="value")

    # initial state
    _on_change({"name": "value", "new": dropdown.value})

    display(widgets.VBox([dropdown, out]))
    return dropdown
