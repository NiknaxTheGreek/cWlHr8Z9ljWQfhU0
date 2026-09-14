from __future__ import annotations

import os
import tempfile
from pathlib import Path

import matplotlib.figure


def mean_sd_text(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def display_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def save_figure_atomic(fig: matplotlib.figure.Figure, path: Path | str, dpi: int = 160) -> None:
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{p.stem}.",suffix=p.suffix,dir=p.parent); os.close(fd)
    try:
        fig.savefig(tmp,dpi=dpi,bbox_inches="tight")
        if Path(tmp).stat().st_size<=0: raise IOError(f"Empty figure output: {tmp}")
        os.replace(tmp,p)
    except Exception:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise
