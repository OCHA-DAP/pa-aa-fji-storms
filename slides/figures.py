"""Static figures for the slide decks: forecast maps and rainfall maps."""

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import (  # noqa: E402
    BoundaryNorm,
    ListedColormap,
    LogNorm,
)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.constants import FJI_CRS  # noqa: E402

RED = "#c62828"  # 64 kt within 72 h (action)
ORANGE = "#f57c00"  # 64 kt 72-120 h (readiness only)
GALE = "#9ec5e6"  # 34 kt
OCEAN = "#e8f1f6"
LAND = "#f4f4f2"
EDGE = "#8a959a"
INK = "#1e2a2b"

TOWNS = {
    "Nadi": (177.44, -17.80),
    "Suva": (178.44, -18.14),
    "Labasa": (179.39, -16.43),
}
ISLANDS = {
    "Viti Levu": (177.95, -17.62),
    "Vanua Levu": (179.15, -16.15),
    "Kadavu": (178.2, -19.25),
    "Lau Group": (181.35, -18.45),
    "Yasawa": (177.05, -16.75),
}

# IMERG rainfall map levels/colours (as in the observational emails)
RAIN_LEVELS = [25, 50, 100, 150, 200, 300, 400, 500, 750]
RAIN_COLORS = [
    "#7fff00",
    "#32cd32",
    "#ffff00",
    "#ffd700",
    "#ff8c00",
    "#ff0000",
    "#b22222",
    "#ff00ff",
    "#8b008b",
]


def _base_map(ax, adm, da_wp, extent, show_pop=True, labels=True):
    ax.set_facecolor(OCEAN)
    adm.plot(ax=ax, facecolor=LAND, edgecolor=EDGE, linewidth=0.5, zorder=1)
    if show_pop:
        da = da_wp.where(da_wp > 1)
        ax.pcolormesh(
            da.x,
            da.y,
            da.values,
            cmap="Purples",
            norm=LogNorm(vmin=2, vmax=3000),
            alpha=0.8,
            zorder=2,
            shading="auto",
            rasterized=True,
        )
    if labels:
        for name, (x, y) in ISLANDS.items():
            if extent[0] < x < extent[1] and extent[2] < y < extent[3]:
                ax.text(
                    x,
                    y,
                    name,
                    fontsize=8.5,
                    style="italic",
                    color="#5e6a6b",
                    ha="center",
                    zorder=8,
                )
        for name, (x, y) in TOWNS.items():
            if extent[0] < x < extent[1] and extent[2] < y < extent[3]:
                ax.plot(x, y, "s", ms=3.5, color=INK, zorder=9)
                ax.text(
                    x + 0.07,
                    y - 0.02,
                    name,
                    fontsize=9.5,
                    color=INK,
                    weight="bold",
                    va="top",
                    zorder=9,
                )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(extent[2:]))))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#c3ced3")


def _scale_bar(ax, extent, km=100):
    lat = extent[2] + (extent[3] - extent[2]) * 0.05
    x1 = extent[1] - (extent[1] - extent[0]) * 0.05
    dlon = km / (111.32 * np.cos(np.deg2rad(lat)))
    x0 = x1 - dlon
    ax.plot(
        [x0, x1],
        [lat, lat],
        color=INK,
        lw=2.5,
        zorder=10,
        solid_capstyle="butt",
    )
    ax.text(
        (x0 + x1) / 2,
        lat + (extent[3] - extent[2]) * 0.012,
        f"{km} km",
        ha="center",
        va="bottom",
        fontsize=9,
        color=INK,
        zorder=10,
    )


def _geom(gdf, speed):
    g = gdf[gdf["buffer_speed"] == speed].geometry.iloc[0]
    return None if g is None or g.is_empty else g


def _plot_geom(ax, geom, **kw):
    if geom is None or geom.is_empty:
        return
    gpd.GeoSeries([geom], crs=FJI_CRS).plot(ax=ax, **kw)


def _figsize(extent, width=8.0):
    """Figure size matching the map's aspect, so there is no white margin."""
    dx = (extent[1] - extent[0]) * np.cos(np.deg2rad(np.mean(extent[2:])))
    dy = extent[3] - extent[2]
    return (width, width * dy / dx)


def forecast_map(
    result,
    adm,
    da_wp,
    extent,
    path,
    show_cone=True,
    show_legend=True,
    extra_tracks=None,
):
    """Map of one forecast: track, 64-kt zone split at 72 h, 34-kt zone."""
    fig, ax = plt.subplots(figsize=_figsize(extent))
    _base_map(ax, adm, da_wp, extent)

    if show_cone and result.get("cone") is not None:
        _plot_geom(
            ax,
            result["cone"],
            facecolor="none",
            edgecolor="#6b7a80",
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=3,
        )

    g34 = _geom(result["buffers"], 34)
    _plot_geom(ax, g34, facecolor=GALE, edgecolor="none", alpha=0.45, zorder=3)
    g64_r = _geom(result["buffers"], 64)
    g64_a = _geom(result["buffers_action"], 64)
    if g64_r is not None:
        g64_late = g64_r.difference(g64_a) if g64_a is not None else g64_r
        _plot_geom(
            ax,
            g64_late,
            facecolor=ORANGE,
            edgecolor=ORANGE,
            alpha=0.55,
            linewidth=1.2,
            zorder=4,
        )
    _plot_geom(
        ax,
        g64_a,
        facecolor=RED,
        edgecolor=RED,
        alpha=0.6,
        linewidth=1.2,
        zorder=4,
    )

    for trk in extra_tracks or []:
        ax.plot(
            trk["Longitude"], trk["Latitude"], color="#9aa6ab", lw=1, zorder=5
        )

    trk = result["track"]
    x, y = trk["Longitude"].values, trk["Latitude"].values
    ax.plot(x, y, color=INK, lw=1.4, zorder=6)
    for _, r in trk.iterrows():
        lt = r["leadtime"]
        if lt % 12:
            continue
        early = lt <= 72
        cat = int(r["Category"]) if r["Category"] == r["Category"] else 0
        ax.scatter(
            r["Longitude"],
            r["Latitude"],
            s=150 if lt % 24 == 0 else 70,
            color=INK if early else "#8a959a",
            edgecolor="white",
            linewidth=1,
            zorder=7,
        )
        if lt % 24 == 0:
            ax.text(
                r["Longitude"],
                r["Latitude"],
                str(cat) if cat else "",
                color="white",
                fontsize=8,
                weight="bold",
                ha="center",
                va="center",
                zorder=8,
                clip_on=True,
            )
            if 0 < lt <= 120:
                ax.text(
                    r["Longitude"] - 0.16,
                    r["Latitude"],
                    f"+{lt} h",
                    fontsize=8.5,
                    color=INK,
                    ha="right",
                    va="center",
                    zorder=8,
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.6),
                    clip_on=True,
                )

    _scale_bar(ax, extent)

    if show_legend:
        handles = [
            Patch(
                fc=RED,
                ec=RED,
                alpha=0.6,
                label="64-kt winds, next 72 h (action)",
            ),
            Patch(
                fc=ORANGE,
                ec=ORANGE,
                alpha=0.55,
                label="64-kt winds, 72–120 h (readiness)",
            ),
            Patch(fc=GALE, alpha=0.45, label="34-kt (gale) winds"),
            Line2D(
                [],
                [],
                marker="o",
                color=INK,
                lw=1.4,
                ms=8,
                label="Forecast track (number = category)",
            ),
        ]
        if show_cone and result.get("cone") is not None:
            handles.append(
                Line2D(
                    [],
                    [],
                    color="#6b7a80",
                    linestyle=(0, (4, 3)),
                    label="Uncertainty cone",
                )
            )
        handles.append(
            Patch(fc="#9e9ac8", alpha=0.7, label="Where people live")
        )
        ax.legend(
            handles=handles,
            loc="lower left",
            fontsize=8.5,
            framealpha=0.92,
            edgecolor="#dde5e8",
        )

    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def rain_map(da_rain, adm, extent, path, track=None):
    """2-day rainfall map in the style of the observational emails."""
    w, h = _figsize(extent)
    fig, ax = plt.subplots(figsize=(w * 1.12, h))
    _base_map(ax, adm, None, extent, show_pop=False)
    cmap = ListedColormap(RAIN_COLORS[:-1])
    cmap.set_over(RAIN_COLORS[-1])
    norm = BoundaryNorm(RAIN_LEVELS, cmap.N)
    da = da_rain.where(da_rain >= RAIN_LEVELS[0])
    m = ax.pcolormesh(
        da.x,
        da.y,
        da.values,
        cmap=cmap,
        norm=norm,
        zorder=2,
        shading="auto",
        rasterized=True,
    )
    adm.plot(
        ax=ax, facecolor="none", edgecolor="#5e6a6b", linewidth=0.5, zorder=3
    )
    if track is not None:
        ax.plot(
            track["Longitude"], track["Latitude"], color=INK, lw=1.4, zorder=6
        )
        ax.scatter(
            track["Longitude"],
            track["Latitude"],
            s=40,
            color=INK,
            edgecolor="white",
            zorder=7,
        )
    _scale_bar(ax, extent)
    cb = fig.colorbar(m, ax=ax, shrink=0.7, extend="max", pad=0.02)
    cb.set_label("2-day rainfall (mm)")
    fig.tight_layout(pad=0.3)
    fig.savefig(path, dpi=170)
    plt.close(fig)
