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

    set_key_line("zone,",       f"zone,{default_zone},")
    set_key_line("nDay,",       "nDay,365,")
    set_key_line("dtSim,",      "dtSim,15,")
    set_key_line("sensAnth,",   "sensAnth,1,")
    set_key_line("bldDensity,", "bldDensity,0.25,")
    set_key_line("verToHor,",   "verToHor,0.40,")
    set_key_line("grasscover,", "grasscover,0.30,")
    set_key_line("treeCover,",  "treeCover,0.25,")

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




# ============================================================
# NEW (2026-02): Template patcher + climate zone autodetect
# ============================================================

def parse_epw_location(epw_path: str):
    """Parse the first EPW header line for city/state/country/lat/lon.

    Supports both standard EPW:
      LOCATION,City,State,Country,Source,WMO,Lat,Lon,TimeZone,Elevation
    and 'custom' headers that omit the leading LOCATION token.
    """
    with open(epw_path, "r", encoding="utf-8", errors="ignore") as f:
        line1 = f.readline().strip()
    parts = [p.strip() for p in line1.split(",")]
    if not parts:
        return {}

    if parts[0].upper() == "LOCATION":
        # Standard EPW
        city = parts[1] if len(parts) > 1 else ""
        state = parts[2] if len(parts) > 2 else ""
        country = parts[3] if len(parts) > 3 else ""
        lat = parts[6] if len(parts) > 6 else ""
        lon = parts[7] if len(parts) > 7 else ""
    else:
        # Often: CityKey,City,State,Country,...,Lat,Lon,TimeZone,Elevation
        city = parts[1] if len(parts) > 1 else parts[0]
        state = parts[2] if len(parts) > 2 else ""
        country = parts[3] if len(parts) > 3 else ""
        lat = parts[6] if len(parts) > 6 else ""
        lon = parts[7] if len(parts) > 7 else ""

    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None

    return {
        "raw": line1,
        "city": city,
        "state": state,
        "country": country,
        "lat": _to_float(lat),
        "lon": _to_float(lon),
    }


def detect_ashrae_zone_from_epw(epw_path: str):
    """Heuristic ASHRAE-like zone guess for UWG 'zone' input.

    Important: EPW files do not store an ASHRAE 169 climate zone.
    UWG uses the 'zone' string to select DOE reference-building archetypes.

    Strategy:
      1) Parse city/state from EPW header line.
      2) Use a small lookup for common teaching/demo cities.
      3) If unknown, return None (caller can fall back to template default).
    """
    loc = parse_epw_location(epw_path)
    city = (loc.get("city") or "").strip().lower()
    state = (loc.get("state") or "").strip().lower()

    # Minimal, pragmatic map (extend as needed)
    # UWG examples list these zones/cities in the template comments.
    LUT = {
        ("phoenix", "az"): "2B",
        ("las vegas", "nv"): "3B",
        ("los angeles", "ca"): "3B",
        ("seattle", "wa"): "4C",
        ("boston", "ma"): "5A",
        ("chicago", "il"): "5A",
        ("houston", "tx"): "2A",
        ("miami", "fl"): "1A",
        ("atlanta", "ga"): "3A",
        ("san francisco", "ca"): "3C",
        ("boulder", "co"): "5B",
        ("albuquerque", "nm"): "4B",
        ("minneapolis", "mn"): "6A",
        ("new york", "ny"): "4A",
        ("denver", "co"): "5B",
    }

    for (c, s), z in LUT.items():
        if city.startswith(c) and (not s or state == s):
            return z

    # Also handle common header city tokens like 'Phoenix_AZ_USA'
    if "_" in city:
        bits = city.split("_")
        if len(bits) >= 2:
            c = bits[0]
            s = bits[1]
            return LUT.get((c, s), None)

    return None


def patch_uwg_param_file(template_path: str, out_path: str, updates: dict, schtraffic=None):
    """Patch a .uwg text template by overwriting known key lines.

    - Works with the Ladybug Tools UWG template syntax:
        key,value,
      optionally with trailing comments after '#'.

    - If schtraffic is provided, it must be a 3x24 iterable (weekday/sat/sun).
      The function overwrites the 3 schedule lines *after* the SchTraffic line.

    This avoids readDOE.py templating issues (pattern mismatches) and is robust
    to minor whitespace / capitalization differences.
    """
    import re

    def _split_comment(line):
        if "#" in line:
            a, b = line.split("#", 1)
            return a.rstrip(), "#" + b
        return line.rstrip(), ""

    def _fmt_key_val(key, value, comment):
        # Preserve key token as seen in the template line
        return f"{key},{value}," + (" " if comment and not comment.startswith(" ") else "") + comment + "\n"

    with open(template_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Build case-insensitive map
    upd = {str(k).lower(): v for k, v in (updates or {}).items()}

    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        raw, comment = _split_comment(line)
        stripped = raw.strip()

        # Preserve blank/comment-only lines
        if not stripped or stripped.startswith("#"):
            out_lines.append(line if line.endswith("\n") else (line + "\n"))
            i += 1
            continue

        parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
        key_token = parts[0] if parts else ""
        key_lc = key_token.lower()

        # SchTraffic block handling
        if key_lc == "schtraffic":
            out_lines.append(line if line.endswith("\n") else (line + "\n"))
            # Next 3 lines should be weekday/sat/sun schedules
            if schtraffic is not None:
                sched = schtraffic
                # normalize list/tuple
                sched = [list(day) for day in sched]
                if len(sched) != 3 or any(len(day) != 24 for day in sched):
                    raise ValueError("schtraffic must be 3x24 (weekday/sat/sun).")

                # overwrite the next 3 lines (or insert if missing)
                labels = ["# Weekday", "# Saturday", "# Sunday"]
                for j in range(3):
                    vals = [float(x) for x in sched[j]]
                    val_str = ",".join(f"{v:.3f}".rstrip("0").rstrip(".") if abs(v - round(v)) > 1e-9 else str(int(round(v))) for v in vals)
                    sched_line = f"{val_str}, {labels[j]}\n"
                    if i + 1 + j < len(lines):
                        out_lines.append(sched_line)
                    else:
                        out_lines.append(sched_line)
                i += 1 + 3
                continue

            i += 1
            continue

        # Generic key patch
        if key_lc in upd:
            val = upd[key_lc]
            # Keep ints as ints in file, floats otherwise
            if isinstance(val, bool):
                val_out = int(val)
            elif isinstance(val, (int,)) and not isinstance(val, bool):
                val_out = int(val)
            else:
                try:
                    # avoid scientific notation for typical ranges
                    val_out = float(val)
                    # format like template (no forced decimals)
                    if abs(val_out - round(val_out)) < 1e-10:
                        val_out = int(round(val_out))
                    else:
                        val_out = f"{val_out:.6g}"
                except Exception:
                    val_out = str(val)

            out_lines.append(_fmt_key_val(key_token, val_out, comment))
            i += 1
            continue

        out_lines.append(line if line.endswith("\n") else (line + "\n"))
        i += 1

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    return out_path
