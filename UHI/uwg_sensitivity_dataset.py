from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------
# EPW utilities
# -----------------------

def _read_epw_col(epw_path: Path, col_idx: int) -> np.ndarray:
    """Read a numeric column from EPW hourly rows (skipping 8 header lines)."""
    df = pd.read_csv(epw_path, skiprows=8, header=None)
    return pd.to_numeric(df.iloc[:, col_idx], errors="coerce").to_numpy()


def _read_epw_months(epw_path: Path, n: int) -> np.ndarray:
    """Read month column (1) for first n rows."""
    df = pd.read_csv(epw_path, skiprows=8, header=None).iloc[:n, :]
    months = pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(1).astype(int).to_numpy()
    return months


# -----------------------
# Folder detection
# -----------------------

def detect_city_dirs(epwdata_root: Path) -> List[Path]:
    """City dirs are those that have rmy/*.epw (ignores folders like cmip6)."""
    out: List[Path] = []
    for p in sorted(epwdata_root.iterdir()):
        if not p.is_dir():
            continue
        rmy = p / "rmy"
        if rmy.exists() and any(rmy.glob("*.epw")):
            out.append(p)
    return out


def pick_base_rmy_epw(city_dir: Path) -> Path:
    """Pick base EPW from <city>/rmy; prefer filenames containing 'rmy'."""
    rmy_dir = city_dir / "rmy"
    epws = sorted(rmy_dir.glob("*.epw"))
    if not epws:
        raise FileNotFoundError(f"No EPWs found in {rmy_dir}")
    rmy_named = [p for p in epws if "rmy" in p.stem.lower()]
    return rmy_named[0] if rmy_named else epws[0]


def find_runs_csv(uwg_dir: Path, city_name: str) -> Optional[Path]:
    """Find <City>_UWG_runs.csv if present; else return None."""
    p = uwg_dir / f"{city_name}_UWG_runs.csv"
    return p if p.exists() else None


def list_uwg_run_epws(uwg_dir: Path, city_name: str) -> List[Tuple[int, Path]]:
    """List EPWs like City_UWG_RunX.epw sorted by X."""
    pat = re.compile(rf"^{re.escape(city_name)}_UWG_Run(\d+)\.epw$", re.IGNORECASE)
    runs: List[Tuple[int, Path]] = []
    for p in uwg_dir.glob(f"{city_name}_UWG_Run*.epw"):
        m = pat.match(p.name)
        if m:
            runs.append((int(m.group(1)), p))
    runs.sort(key=lambda t: t[0])
    return runs


# -----------------------
# Metrics
# -----------------------

def compute_metrics_for_run(
    base_dbt: np.ndarray,
    uwg_dbt: np.ndarray,
    months: np.ndarray,
) -> Dict[str, float]:
    n = int(min(len(base_dbt), len(uwg_dbt), len(months)))
    b = base_dbt[:n]
    u = uwg_dbt[:n]
    m = months[:n]
    dt = u - b

    summer = np.isin(m, [6, 7, 8])
    mjjas = np.isin(m, [5, 6, 7, 8, 9])

    out: Dict[str, float] = {}
    out["dT_mean"] = float(np.nanmean(dt))
    out["dT_p95"] = float(np.nanpercentile(dt, 95))
    out["dT_p99"] = float(np.nanpercentile(dt, 99))
    out["dT_max"] = float(np.nanmax(dt))
    out["dT_summer_mean"] = float(np.nanmean(dt[summer]))
    out["dT_mjjas_mean"] = float(np.nanmean(dt[mjjas]))

    out["CDD18_base_h"] = float(np.nansum(np.maximum(b - 18.0, 0.0)))
    out["CDD18_uwg_h"] = float(np.nansum(np.maximum(u - 18.0, 0.0)))
    out["CDD18_delta_h"] = out["CDD18_uwg_h"] - out["CDD18_base_h"]

    for thr in (26, 30, 35):
        out[f"hrs_base_gt{thr}"] = float(np.nansum(b > thr))
        out[f"hrs_uwg_gt{thr}"] = float(np.nansum(u > thr))
        out[f"hrs_delta_gt{thr}"] = out[f"hrs_uwg_gt{thr}"] - out[f"hrs_base_gt{thr}"]

    return out


# -----------------------
# Main analysis entrypoint
# -----------------------

def analyze_uwg_sensitivity(
    wxbuild_root: str | Path = "/content/wxbuild",
    city: Optional[str] = None,
    out_csv_name: str = "UWG_sensitivity_dataset.csv",
) -> Path:
    """
    Build a tidy dataset that joins:
      - recorded UI parameters from <city>/uwg/<city>_UWG_runs.csv (if present)
      - temperature response metrics computed from EPWs:
            base = <city>/rmy/*.epw (prefer *RMY*)
            runs = <city>/uwg/<city>_UWG_RunX.epw

    If `city` is None:
      - if exactly one city dir exists under epwdata, use it
      - otherwise process ALL city dirs and combine rows.

    Writes per-city CSV to each <city>/uwg folder.
    If multiple cities are processed, also writes a combined CSV to:
      <wxbuild_root>/UHI_outputs/<out_csv_name>

    Returns the path to the written CSV (combined if multi-city; else single-city).
    """
    wxbuild_root = Path(wxbuild_root)
    epwdata_root = wxbuild_root / "epwdata"
    if not epwdata_root.exists():
        raise FileNotFoundError(f"Missing epwdata folder: {epwdata_root}")

    city_dirs = detect_city_dirs(epwdata_root)

    if city is not None:
        city_dirs = [d for d in city_dirs if d.name == city]
        if not city_dirs:
            raise FileNotFoundError(f"City '{city}' not found under {epwdata_root}")

    all_rows: List[Dict] = []
    per_city_paths: List[Path] = []

    for cdir in city_dirs:
        city_name = cdir.name
        uwg_dir = cdir / "uwg"
        if not uwg_dir.exists():
            continue

        base_epw = pick_base_rmy_epw(cdir)
        base_dbt = _read_epw_col(base_epw, 6)
        months = _read_epw_months(base_epw, n=len(base_dbt))

        runs = list_uwg_run_epws(uwg_dir, city_name)
        if not runs:
            continue

        runs_csv = find_runs_csv(uwg_dir, city_name)
        runlog = pd.read_csv(runs_csv) if runs_csv else None

        # index runlog by run_index if possible
        runlog_by_idx: Dict[int, Dict] = {}
        if runlog is not None and "run_index" in runlog.columns:
            for _, rr in runlog.iterrows():
                try:
                    runlog_by_idx[int(rr["run_index"])] = rr.to_dict()
                except Exception:
                    pass

        city_rows: List[Dict] = []
        for run_idx, epw_path in runs:
            uwg_dbt = _read_epw_col(epw_path, 6)
            met = compute_metrics_for_run(base_dbt, uwg_dbt, months)

            row: Dict = {
                "city": city_name,
                "base_epw": base_epw.name,
                "uwg_epw": epw_path.name,
                "run_index": run_idx,
            }
            if run_idx in runlog_by_idx:
                for k, v in runlog_by_idx[run_idx].items():
                    if k not in row:
                        row[k] = v
            row.update(met)

            all_rows.append(row)
            city_rows.append(row)

        out_csv = uwg_dir / out_csv_name
        pd.DataFrame(city_rows).to_csv(out_csv, index=False)
        per_city_paths.append(out_csv)

    if not per_city_paths:
        raise FileNotFoundError(
            "No UWG runs found to analyze. Expected files like <city>/uwg/<city>_UWG_Run*.epw"
        )

    # If multiple cities processed, write combined
    cities_written = sorted({r["city"] for r in all_rows})
    if len(cities_written) > 1:
        combined_dir = wxbuild_root / "UHI_outputs"
        combined_dir.mkdir(parents=True, exist_ok=True)
        combined_csv = combined_dir / out_csv_name
        pd.DataFrame(all_rows).to_csv(combined_csv, index=False)
        return combined_csv

    return per_city_paths[0]


if __name__ == "__main__":
    out = analyze_uwg_sensitivity()
    print("Wrote:", out)
