from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

PREFERRED_CJK_FONTS = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "PingFang SC",
    "WenQuanYi Zen Hei",
    "Arial Unicode MS",
]


def _available_font_names() -> set[str]:
    return {font.name for font in font_manager.fontManager.ttflist}


def pick_cjk_font() -> str:
    available = _available_font_names()
    for font_name in PREFERRED_CJK_FONTS:
        if font_name in available:
            return font_name
    return "DejaVu Sans"


def configure_matplotlib() -> str:
    font_name = pick_cjk_font()
    fallback_fonts = [name for name in PREFERRED_CJK_FONTS if name != font_name]
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_name, *fallback_fonts, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.35,
        }
    )
    return font_name


def save_figure(
    fig: Figure,
    path: str | Path,
    *,
    dpi: int = 160,
    tight_layout_pad: float = 1.2,
    left: float | None = None,
    right: float | None = None,
    bottom: float | None = None,
    top: float | None = None,
) -> None:
    adjust_args = {
        key: value
        for key, value in {
            "left": left,
            "right": right,
            "bottom": bottom,
            "top": top,
        }.items()
        if value is not None
    }
    if adjust_args:
        fig.subplots_adjust(**adjust_args)
    fig.tight_layout(pad=tight_layout_pad)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
