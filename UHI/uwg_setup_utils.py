# uwg_setup_utils.py
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


# --------------------------------------------
# Helpers for repo + EPW file selection
# --------------------------------------------
def detect_cities(epwdata_root: str) -> List[str]:
    root = Path(epwdata_root)
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def find_rmy_epw(city_root: str) -> Optional[str]:
    """
    Best-effort search for an EPW to use:
    - prefers a file in a 'Chosen_scenario' folder if present
    - otherwise falls back to any .epw under the city directory
    """
    city_root = str(city_root)
    chosen = Path(city_root) / "Chosen_scenario"
    if chosen.exists():
        epws = sorted(chosen.glob("*.epw"))
        if epws:
            return str(epws[0])

    epws = sorted(Path(city_root).rglob("*.epw"))
    return str(epws[0]) if epws else None


def ensure_city_uwg_dir(city_root: str) -> str:
    uwg_dir = Path(city_root) / "uwg"
    uwg_dir.mkdir(parents=True, exist_ok=True)
    return str(uwg_dir)


def copy_example_epw(dst_path: str) -> str:
    """
    If you need an example EPW placeholder.
    (Kept for compatibility with earlier notebooks.)
    """
    # No-op placeholder: you likely won't use this in your workflow.
    return dst_path


# --------------------------------------------
# Patching UWG readDOE.py patterns (best effort)
# --------------------------------------------
def patch_readDOE_mass_wall_roof(uwg_repo_root: str) -> bool:
    """
    Best-effort patch for certain UWG versions where readDOE.py patterns
    might not match DOE template tokens (MassWall/MassFloor etc).
    If the file layout differs, we do nothing and return False.
    """
    try:
        repo = Path(uwg_repo_root)
        readDOE = repo / "uwg" / "readDOE.py"
        if not readDOE.exists():
            # sometimes installed package layout differs
            return False

        txt = readDOE.read_text(encoding="utf-8", errors="ignore")

        # This is deliberately conservative: only patch if we find expected tokens.
        if "MassWall" in txt or "MassFloor" in txt:
            return False

        # Otherwise, you could insert patches here.
        # For now: keep behavior as best effort without breaking.
        return False
    except Exception:
        return False


def ensure_and_patch_uwg(uwg_repo_root: str) -> None:
    """
    Placeholder for compatibility: in your Colab repo workflow you may patch
    uwg's readDOE if needed.
    """
    _ = patch_readDOE_mass_wall_roof(uwg_repo_root)


# --------------------------------------------
# UWG param file writer helper
# --------------------------------------------
def set_key_line_in_file(path: str | Path, prefix: str, value_line: str) -> None:
    """
    Replace the first line starting with `prefix` with `value_line`.
    If prefix not found, append value_line.
    Keeps rest of file intact.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    replaced = False
    for i, line in enumerate(lines):
        if line.strip().lower().startswith(prefix.strip().lower()):
            lines[i] = value_line
            replaced = True
            break

    if not replaced:
        lines.append(value_line)

    path.write_text("\n".join(lines))


# ============================================================================
# Climate-zone inference (EPW -> UWG `zone`)
#
# UWG's `zone` parameter is an ASHRAE-like climate-zone code (e.g. 2B, 3C, 5A).
# EPW files do not reliably store ASHRAE zone metadata, so this is a best-effort:
#   1) try infer from EPW LOCATION line (city name) via lookup for common cities
#   2) fallback to simple HDD18-based zone number + humidity letter heuristic
#
# If you want perfect ASHRAE-169 classification, you'd need an explicit dataset
# or a full standard-compliant implementation. This keeps things lightweight
# and works well for your UWG teaching workflow.
# ============================================================================

# A practical city-name lookup for your typical workflow cities + UWG examples.
# Keys are normalized substrings to search in EPW LOCATION line.
CITY_TO_ZONE_LOOKUP: Dict[str, str] = {
    "phoenix": "2B",
    "miami": "1A",
    "houston": "2A",
    "las vegas": "3B",
    "los angeles": "3C",
    "san francisco": "3C",
    "atlanta": "3A",
    "baltimore": "4A",
    "albuquerque": "4B",
    "seattle": "4C",
    "chicago": "5A",
    "boston": "5A",
    "everett": "5A",
    "new york": "4A",      # rough
    "minneapolis": "6A",
    "boulder": "5B",
    "helena": "6B",
    "duluth": "7",
    "fairbanks": "8",
    "singapore": "1A",     # rough
    "nairobi": "3A",       # rough
    "ahmedabad": "2B",     # rough
    "cairo": "2B",         # rough
    "lisbon": "3C",        # rough
    "tokyo": "3A",         # rough
}

VALID_UWG_ZONES = {"1A", "2A", "2B", "3A", "3B", "3C", "4A", "4B", "4C", "5A", "5B", "6A", "6B", "7", "8"}


def _read_epw_location_line(epw_path: str) -> str:
    with open(epw_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.readline().strip()


def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\\s\\-_,.]", " ", s)
    s = re.sub(r"\\s+", " ", s).strip()
    return s


def _infer_zone_from_city_name(location_line: str) -> Optional[str]:
    loc = _normalize(location_line)
    for key, zone in CITY_TO_ZONE_LOOKUP.items():
        if key in loc:
            return zone
    return None


def _read_epw_hourly_basic(epw_path: str) -> pd.DataFrame:
    """
    Minimal EPW parse: read DryBulb and RH from standard columns.
    Assumes 8 header rows.
    """
    # EPW standard columns (35)
    cols = [
        "Year","Month","Day","Hour","Minute","DataSource",
        "DryBulb_C","DewPoint_C","RelHum_%","AtmPress_Pa",
        "ExtHorzRad_Whm2","ExtDirNormRad_Whm2","HorzIRSky_Whm2",
        "GlobHorzRad_Whm2","DirNormRad_Whm2","DiffHorzRad_Whm2",
        "GlobHorzIll_lux","DirNormIll_lux","DiffHorzIll_lux","ZenLum_cd_m2",
        "WindDir_deg","WindSpd_ms","TotSkyCvr_tenths","OpqSkyCvr_tenths",
        "Visibility_km","Ceiling_m","PresWeathObs","PresWeathCodes",
        "PrecWater_mm","AerosolOptDepth","SnowDepth_cm","DaysSinceSnow",
        "Albedo","LiqPrecDepth_mm","LiqPrecQty_hr"
    ]
    df = pd.read_csv(epw_path, skiprows=8, header=None)
    df = df.iloc[:, :len(cols)]
    df.columns = cols
    return df


def _infer_zone_number_from_hdd18(hdd18_c_day: float) -> str:
    """
    Coarse HDD18-based binning to the ASHRAE zone number (3–8 typical).
    This is a pragmatic approximation for teaching + UWG runs.
    """
    # These thresholds are intentionally coarse.
    if hdd18_c_day < 1000:
        return "2"
    if hdd18_c_day < 2000:
        return "3"
    if hdd18_c_day < 3000:
        return "4"
    if hdd18_c_day < 4000:
        return "5"
    if hdd18_c_day < 5000:
        return "6"
    if hdd18_c_day < 6500:
        return "7"
    return "8"


def _infer_moisture_letter(mean_rh: float) -> str:
    """
    Very rough humidity letter:
    - A: humid (high RH)
    - B: dry   (low RH)
    - C: marine-ish (moderate RH)
    """
    if mean_rh >= 60:
        return "A"
    if mean_rh <= 45:
        return "B"
    return "C"


def infer_uwg_zone_from_epw(epw_path: str, fallback: str = "5A") -> str:
    """
    Return an ASHRAE-like UWG climate zone string.
    """
    try:
        loc = _read_epw_location_line(epw_path)
        z = _infer_zone_from_city_name(loc)
        if z in VALID_UWG_ZONES:
            return z
    except Exception:
        pass

    # Heuristic fallback from degree-days + mean RH
    try:
        df = _read_epw_hourly_basic(epw_path)
        t = pd.to_numeric(df["DryBulb_C"], errors="coerce")
        rh = pd.to_numeric(df["RelHum_%"], errors="coerce")

        # HDD18 in °C·day (sum over hours / 24)
        hdd18_c_hour = (18.0 - t).clip(lower=0.0)
        hdd18_c_day = float(hdd18_c_hour.sum() / 24.0)

        zone_num = _infer_zone_number_from_hdd18(hdd18_c_day)

        # Determine moisture letter (skip for zones 7–8 which are typically just number in UWG)
        mean_rh = float(rh.mean(skipna=True))
        letter = _infer_moisture_letter(mean_rh)

        if zone_num in {"7", "8"}:
            z = zone_num
        else:
            z = f"{zone_num}{letter}"

        if z in VALID_UWG_ZONES:
            return z
    except Exception:
        pass

    # Final fallback
    return fallback if fallback in VALID_UWG_ZONES else "5A"


