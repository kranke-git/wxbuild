"""wxbuild plotting helper: DBT across SSPs.

Usage in Colab:
    from plots.dbt_anomaly_plot import plot_dbt_plus_anomaly_all_ssps
    plot_dbt_plus_anomaly_all_ssps(current_tmy, location=location)
"""

from __future__ import annotations

from typing import Iterable, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_SSP_COLORS: Dict[str, str] = {
    "ssp126": "#f6b0b0",
    "ssp245": "#ef7f7f",
    "ssp370": "#d34a4a",
    "ssp585": "#6e0f0f",
}


def plot_dbt_plus_anomaly_all_ssps(
    current_tmy,
    *,
    location: Optional[str] = None,
    years: Optional[Iterable[int]] = None,
    ssps: Optional[Iterable[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    title_suffix: str = "(rolling mean, window=5)",
    linewidth: float = 3.0,
    figsize=(12.5, 4.5),
    show: bool = True,
):
    """Plot annual dry-bulb temperature (dbt) for multiple SSPs.

    Expects: current_tmy.getVariableAnomalies(params={...})
    returning a DataFrame indexed by year with columns: 'dbt', 'dbt_anomaly'.

    Returns (fig, ax).
    """
    if years is None:
        years_arr = np.arange(2020, 2101)
    else:
        years_arr = np.array(list(years), dtype=int)

    if ssps is None:
        ssps_list = ["ssp126", "ssp245", "ssp370", "ssp585"]
    else:
        ssps_list = list(ssps)

    if colors is None:
        colors = DEFAULT_SSP_COLORS

    fig, ax = plt.subplots(figsize=figsize)

    for ssp in ssps_list:
        df = current_tmy.getVariableAnomalies(
            params={"variable": "dbt", "years": years_arr, "futexp": ssp}
        )

        if ("dbt" not in df.columns) or ("dbt_anomaly" not in df.columns):
            raise ValueError(
                f"Expected columns ['dbt','dbt_anomaly'] from getVariableAnomalies, got: {list(df.columns)}"
            )

        y = df["dbt"].values
        ax.plot(df.index.values, y, label=ssp, linewidth=linewidth, color=colors.get(ssp))

    loc_txt = location or ""
    ax.set_title(f"{loc_txt} — dbt {title_suffix}".strip())
    ax.set_xlabel("Year")
    ax.set_ylabel("Dry-Bulb Temperature (°C)")
    ax.grid(True, alpha=0.25)

    # Remove outer border; keep only x/y axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax
