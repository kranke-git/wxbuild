"""current.py

Minimal helper to plot the CURRENT RMY dry-bulb temperature profile (8760 hours).

Expected object shape (matches wxbuild notebooks):
- current_rmy.files[0].data is a dict-like with key 'dbt' (dry-bulb temp in °C)
"""

from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt


def plot_current_rmy(
    current_rmy: Any,
    *,
    title: str = "Current RMY: Dry-Bulb Temperature (8760h)",
    label: str = "Original RMY",
    color: str = "#333333",  # dark grey
    linewidth: float = 1.5,
    figsize: tuple = (15, 4),
    show: bool = True,
    savepath: Optional[str] = None,
) -> plt.Axes:
    """Plot the current RMY temperature profile."""
    try:
        dbt = current_rmy.files[0].data["dbt"]
    except Exception as e:
        raise ValueError(
            "Could not access current_rmy.files[0].data['dbt'].\n"
            "Make sure you ran the earlier notebook steps that create `current_rmy`."
        ) from e

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(dbt, label=label, linewidth=linewidth, color=color)

    ax.set_title(title)
    ax.set_xlabel("Hour of Year")
    ax.set_ylabel("Dry-Bulb Temperature (°C)")
    ax.grid(True, alpha=0.25)

    # Remove legend border box
    ax.legend(frameon=False)

    # Remove the outer plot border: keep only x/y axes (left & bottom spines)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return ax
