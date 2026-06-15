"""Render the markdown research note to PDF.

Prefers pandoc + a LaTeX engine (tectonic > xelatex > pdflatex). Locates pandoc
and tectonic even when they are installed per-user and not on PATH (common on
Windows). Raises with a clear message if no toolchain is found.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import utils


def _find_pandoc() -> str | None:
    p = shutil.which("pandoc")
    if p:
        return p
    for c in [Path(os.environ.get("LOCALAPPDATA", "")) / "Pandoc" / "pandoc.exe"]:
        if c.exists():
            return str(c)
    return None


def _find_engine() -> str | None:
    # tectonic (self-contained) first, then xelatex/pdflatex on PATH
    cand = Path(os.environ.get("LOCALAPPDATA", "")) / "tectonic" / "tectonic.exe"
    if cand.exists():
        return str(cand)
    for e in ("tectonic", "xelatex", "lualatex", "pdflatex"):
        p = shutil.which(e)
        if p:
            return p
    return None


def render_note(cfg: dict) -> Path:
    pandoc = _find_pandoc()
    engine = _find_engine()
    note_md = utils.resolve(cfg["output"]["note_md"])
    out_pdf = utils.resolve(cfg["output"]["pdf_path"])
    header = note_md.parent / "_pandoc_header.tex"

    if not pandoc:
        raise RuntimeError("pandoc not found. Install via `winget install JohnMacFarlane.Pandoc`.")
    if not engine:
        raise RuntimeError(
            "No LaTeX engine found. Install tectonic (recommended) or MiKTeX/TeX Live, "
            "or render the HTML fallback."
        )

    cmd = [
        pandoc, str(note_md), "-o", str(out_pdf),
        f"--pdf-engine={engine}",
        f"--resource-path={note_md.parent}",
        "-V", "geometry:margin=2cm", "-V", "fontsize=10pt",
        "-V", "colorlinks=true", "-V", "linkcolor=blue", "-V", "urlcolor=blue",
    ]
    if header.exists():
        cmd[3:3] = [f"--include-in-header={header}"]

    print(f"[render] pandoc={pandoc}\n[render] engine={engine}")
    subprocess.run(cmd, check=True)
    print(f"[render] wrote {out_pdf}")
    return out_pdf


if __name__ == "__main__":
    render_note(utils.load_config())
