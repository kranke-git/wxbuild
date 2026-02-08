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


# -----------------------
# Ranking + plots
# -----------------------

def _auto_detect_features_targets(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Heuristic split: features = UI params; targets = derived metrics (dT_*, CDD*, hrs_*)."""
    meta_cols = {"city", "base_epw", "uwg_epw", "run_index"}
    numeric_cols = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]

    # Common UI param names coming from the UWG UI run log
    feature_hint = {
        "bldheight_m", "blddensity", "vertohor_h_w", "albroad", "grasscover", "treecover",
        "sensanth_w_m2", "schtrafficx", "glzr", "shgc", "albroof"
    }

    features = [c for c in numeric_cols if c in feature_hint]
    targets = [c for c in numeric_cols if c.startswith("dT_") or c.startswith("CDD") or c.startswith("hrs_")]

    # Fallbacks if names differ
    if not features:
        features = [c for c in numeric_cols if not (c.startswith("dT_") or c.startswith("CDD") or c.startswith("hrs_"))]
    if not targets:
        targets = [c for c in numeric_cols if c not in features]

    return features, targets




# Pretty labels for student-facing plots
FEATURE_LABELS = {
    "bldheight_m": "Building Height (m)",
    "blddensity": "Building Footprint Density (0–1)",
    "vertohor_h_w": "Canyon Aspect Ratio H/W (–)",
    "albroad": "Road Albedo (–)",
    "grasscover": "Green Space Cover (–)",
    "treecover": "Tree Canopy Cover (–)",
    "sensanth_w_m2": "Traffic / Anthropogenic Heat (W/m²)",
    "schtrafficx": "Traffic Intensity Multiplier (–)",
    "glzr": "Glazing Ratio (Window-to-Wall, –)",
    "shgc": "Window SHGC (–)",
    "albroof": "Roof Albedo (–)",
}

TARGET_LABELS = {
    "dT_mean": "ΔT mean (°C)",
    "dT_p95": "ΔT p95 (°C)",
    "dT_p99": "ΔT p99 (°C)",
    "dT_max": "ΔT max (°C)",
    "dT_summer_mean": "ΔT summer mean (°C)",
    "dT_mjjas_mean": "ΔT MJJAS mean (°C)",
    "CDD18_delta_h": "ΔCDD18 (°C·h)",
    "hrs_delta_gt26": "Δ hours >26°C",
    "hrs_delta_gt30": "Δ hours >30°C",
}

# Default target set for sensitivity (kept intentionally small for low-run experiments)
DEFAULT_TARGETS = [
    "dT_mean",
    "dT_p95",
    "dT_p99",
    "dT_summer_mean",
    "CDD18_delta_h",
    "hrs_delta_gt26",
    "hrs_delta_gt30",
]
def _plot_corr_heatmap(df: pd.DataFrame, features: list[str], targets: list[str], title: str) -> pd.DataFrame:
    """Plot correlation heatmap (features x targets). Returns corr matrix."""
    # Filter targets to a small, student-friendly set if present
    targets = [t for t in DEFAULT_TARGETS if t in targets] or targets

    X = df[features].apply(pd.to_numeric, errors="coerce")
    Y = df[targets].apply(pd.to_numeric, errors="coerce")
    mask = X.notna().all(axis=1) & Y.notna().all(axis=1)
    X, Y = X[mask], Y[mask]

    corr = pd.DataFrame(index=features, columns=targets, dtype=float)
    for t in targets:
        corr[t] = X.corrwith(Y[t])

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Custom red gradient (light → dark) to match your style
    reds = LinearSegmentedColormap.from_list(
        "uwg_reds",
        ["#F6C6C6", "#E88F8F", "#D04A4A", "#A51E2C", "#7A0000"],
        N=256,
    )

    ylabels = [FEATURE_LABELS.get(f, f) for f in corr.index]
    xlabels = [TARGET_LABELS.get(t, t) for t in corr.columns]

    plt.figure(figsize=(max(10, 0.95 * len(xlabels) + 4), max(6, 0.62 * len(ylabels) + 2)))
    im = plt.imshow(corr.values, aspect="auto", cmap=reds, vmin=-1, vmax=1)
    plt.xticks(range(len(xlabels)), xlabels, rotation=35, ha="right")
    plt.yticks(range(len(ylabels)), ylabels)
    plt.title(title)
    cb = plt.colorbar(im, label="corr")
    plt.tight_layout()
    plt.show()

    return corr
def _ensure_sklearn_installed() -> None:
    """Try to import sklearn; if missing (common in fresh Colab), install it."""
    try:
        import sklearn  # noqa: F401
        return
    except Exception:
        import sys, subprocess
        print("Installing scikit-learn...")
        subprocess.check_call([sys.executable, "-m", "pip", "-q", "install", "scikit-learn"])


def _perm_importance_rankings(
    df: pd.DataFrame,
    features: list[str],
    targets: list[str],
    n_splits: int = 5,
    random_state: int = 0,
    top_k: int = 8,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute permutation importance rankings per target using RandomForestRegressor.
    Returns:
      - overall table (normalized mean importance across targets)
      - per-target dict {target: {"cv_r2": float, "importance": Series}}
    """
    _ensure_sklearn_installed()
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    from sklearn.inspection import permutation_importance
    import matplotlib.pyplot as plt
    import numpy as np

    X = df[features].apply(pd.to_numeric, errors="coerce")
    per_target = {}

    # clean rows per target (so each target uses same X rows where y is finite)
    for t in targets:
        y = pd.to_numeric(df[t], errors="coerce")
        mask = X.notna().all(axis=1) & y.notna()
        Xt = X[mask].reset_index(drop=True)
        yt = y[mask].reset_index(drop=True)

        if len(Xt) < max(10, n_splits * 3):
            print(f"⚠️ Skipping {t}: not enough runs after cleaning ({len(Xt)} rows).")
            continue

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        imps = []
        scores = []
        for tr, te in kf.split(Xt):
            model = RandomForestRegressor(
                n_estimators=400,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            )
            model.fit(Xt.iloc[tr], yt.iloc[tr])
            pred = model.predict(Xt.iloc[te])
            scores.append(r2_score(yt.iloc[te], pred))

            pi = permutation_importance(
                model, Xt.iloc[te], yt.iloc[te],
                n_repeats=25,
                random_state=random_state,
                n_jobs=-1
            )
            imps.append(pi.importances_mean)

        mean_r2 = float(np.mean(scores))
        mean_imp = pd.Series(np.mean(imps, axis=0), index=features).sort_values(ascending=False)
        per_target[t] = {"cv_r2": mean_r2, "importance": mean_imp}

        # plot top_k
        top = mean_imp.head(top_k)[::-1]
        plt.figure(figsize=(8, 4))
        plt.barh(top.index, top.values)
        plt.title(f"Top drivers for {t} (Permutation Importance, CV R²≈{mean_r2:.2f})")
        plt.xlabel("Importance (Δ performance when shuffled)")
        plt.tight_layout()
        plt.show()

    if not per_target:
        # Too few runs for ML-based rankings (common early in exploration).
        # Fall back to correlation-based ranking (mean absolute correlation across targets).
        targets_eff = [t for t in DEFAULT_TARGETS if t in targets] or targets
        Xc = df[features].apply(pd.to_numeric, errors="coerce")
        corr_rows = []
        for t in targets_eff:
            yc = pd.to_numeric(df[t], errors="coerce")
            mask = Xc.notna().all(axis=1) & yc.notna()
            if mask.sum() < 3:
                continue
            c = Xc[mask].corrwith(yc[mask]).abs()
            corr_rows.append(c)
        if not corr_rows:
            raise RuntimeError("Not enough valid runs to compute even correlation rankings.")
        mean_abs_corr = pd.concat(corr_rows, axis=1).mean(axis=1).sort_values(ascending=False)
        overall_norm = pd.DataFrame({t: 0.0 for t in targets_eff}, index=features)
        overall_norm["mean_importance"] = mean_abs_corr / (mean_abs_corr.max() + 1e-12)
        return overall_norm.sort_values("mean_importance", ascending=False), {}

    # overall normalized rank
    overall = pd.DataFrame({t: per_target[t]["importance"] for t in per_target}).fillna(0.0)
    overall_norm = overall.apply(lambda s: s / (s.sum() + 1e-12), axis=0)
    overall_norm["mean_importance"] = overall_norm.mean(axis=1)
    overall_norm = overall_norm.sort_values("mean_importance", ascending=False)

    return overall_norm, per_target


def analyze_and_rank(
    wxbuild_root: str | Path = "/content/wxbuild",
    city: Optional[str] = None,
    out_csv_name: str = "UWG_sensitivity_dataset.csv",
    make_plots: bool = True,
) -> dict:
    """
    Convenience wrapper:
      1) builds/loads the sensitivity dataset CSV (per-city + combined if multi-city)
      2) prints + plots correlation matrix + permutation importance rankings

    Returns a dict with paths and tables.
    """
    out_csv = analyze_uwg_sensitivity(wxbuild_root=wxbuild_root, city=city, out_csv_name=out_csv_name)
    df = pd.read_csv(out_csv)

    features, targets = _auto_detect_features_targets(df)
    if not features or not targets:
        raise RuntimeError("Could not detect features/targets. Check your run log columns and computed metrics.")

    results = {"dataset_csv": out_csv, "features": features, "targets": targets}

    if make_plots:
        title = "Correlation heatmap: UWG UI parameters vs temperature response metrics"
        corr = _plot_corr_heatmap(df, features, targets, title=title)
        results["corr"] = corr

        overall_rank, per_target = _perm_importance_rankings(df, features, targets)
        results["overall_rank"] = overall_rank
        results["per_target"] = per_target

        print("\n✅ Overall strongest knobs across targets:")
        # show pretty labels where possible
        display_df = overall_rank[["mean_importance"]].copy()
        display_df.index = [FEATURE_LABELS.get(i, i) for i in display_df.index]
        print(display_df.head(12).to_string())

    return results


if __name__ == "__main__":
    # Colab tip: run with `%run uwg_sensitivity_dataset.py` to display plots inline.
    res = analyze_and_rank()
    print("\nDataset CSV:", res["dataset_csv"])
