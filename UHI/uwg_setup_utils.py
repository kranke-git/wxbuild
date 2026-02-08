from __future__ import annotations

import re
import shutil
from pathlib import Path


def detect_cities(wxbuild_root: str | Path) -> list[str]:
    """Return folder names under <wxbuild_root>/epwdata/."""
    wxbuild_root = Path(wxbuild_root)
    epwdata = wxbuild_root / "epwdata"
    if not epwdata.exists():
        return []
    return sorted([p.name for p in epwdata.iterdir() if p.is_dir()])


def find_rmy_epw(wxbuild_root: str | Path, city: str) -> Path:
    """Find an EPW inside <wxbuild_root>/epwdata/<city>/rmy/ (prefer *RMY*.epw)."""
    wxbuild_root = Path(wxbuild_root)
    rmy_dir = wxbuild_root / "epwdata" / city / "rmy"
    if not rmy_dir.exists():
        raise FileNotFoundError(f"Missing rmy folder: {rmy_dir}")

    epws = sorted(rmy_dir.glob("*.epw"))
    if not epws:
        raise FileNotFoundError(f"No EPW files found in: {rmy_dir}")

    rmy_named = [p for p in epws if "rmy" in p.stem.lower()]
    return rmy_named[0] if rmy_named else epws[0]


def ensure_city_uwg_dir(wxbuild_root: str | Path, city: str) -> Path:
    """Create <wxbuild_root>/epwdata/<city>/uwg if missing and return it."""
    wxbuild_root = Path(wxbuild_root)
    out_dir = wxbuild_root / "epwdata" / city / "uwg"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def patch_readDOE_mass_wall_roof(uwg_root: str | Path) -> None:
    """
    Patch UWG readDOE.py:
      "SteelFrame" -> "MassWall"
      "IEAD"      -> "MassRoof"
    Creates a .bak once.
    """
    uwg_root = Path(uwg_root)
    readDOE = uwg_root / "uwg" / "readDOE.py"
    if not readDOE.exists():
        raise FileNotFoundError(f"Can't find readDOE.py at: {readDOE}")

    bak = readDOE.with_suffix(".py.bak")
    if not bak.exists():
        shutil.copy2(readDOE, bak)
        print(f"✅ Backed up: {bak}")

    txt = readDOE.read_text()
    txt2 = txt

    txt2, n_wall = re.subn(r'"SteelFrame"', '"MassWall"', txt2)
    txt2, n_roof = re.subn(r'"IEAD"', '"MassRoof"', txt2)

    if txt2 == txt:
        print("⚠️ readDOE.py: no replacements made (patterns not found).")
        return

    readDOE.write_text(txt2)
    print(f"✅ Patched readDOE.py: SteelFrame→MassWall ({n_wall}), IEAD→MassRoof ({n_roof})")


def build_initialize_residential(
    uwg_root: str | Path,
    out_param_path: str | Path,
    default_zone: str = "5A",
) -> Path:
    """
    Write initialize_residential.uwg using UWG template.
    Removes any existing building stock rows and inserts:
      midriseapartment, new, 1.0
    """
    uwg_root = Path(uwg_root)
    out_param_path = Path(out_param_path)

    template = uwg_root / "resources" / "initialize_singapore.uwg"
    if not template.exists():
        raise FileNotFoundError(f"Missing template: {template}")

    lines = template.read_text().splitlines()

    def set_key_line(prefix: str, newline: str) -> None:
        for i, ln in enumerate(lines):
            if ln.strip().lower().startswith(prefix.lower()):
                lines[i] = newline
                return
        lines.append(newline)

    # Stable defaults
    set_key_line("zone,",       f"zone,{default_zone},")
    set_key_line("nDay,",       "nDay,365,")
    set_key_line("dtSim,",      "dtSim,15,")
    set_key_line("sensAnth,",   "sensAnth,1,")
    set_key_line("bldDensity,", "bldDensity,0.25,")
    set_key_line("verToHor,",   "verToHor,0.40,")
    set_key_line("grasscover,", "grasscover,0.30,")
    set_key_line("treeCover,",  "treeCover,0.25,")

    # Replace stock lines
    stock_pat = re.compile(r"(?i)^\s*([a-z0-9_]+)\s*,\s*(pre80|pst80|new)\s*,\s*([0-9]*\.?[0-9]+)\s*,?\s*$")
    stock_idxs = [i for i, ln in enumerate(lines) if stock_pat.match(ln.strip())]
    for i in reversed(stock_idxs):
        lines.pop(i)

    insert_at = stock_idxs[0] if stock_idxs else len(lines)
    lines.insert(insert_at, "midriseapartment, new, 1.0")

    out_param_path.parent.mkdir(parents=True, exist_ok=True)
    out_param_path.write_text("\n".join(lines))
    print("✅ Wrote param file:", out_param_path)
    return out_param_path


def set_key_line_in_file(path: str | Path, prefix: str, value_line: str) -> None:
    """Edit a key line in-place in the .uwg param file."""
    path = Path(path)
    lines = path.read_text().splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith(prefix.lower()):
            lines[i] = value_line
            path.write_text("\n".join(lines))
            return
    lines.append(value_line)
    path.write_text("\n".join(lines))


