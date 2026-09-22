"""Export module impact results as Quarto slides without audio dependencies."""

from html import escape
from pathlib import Path
from urllib.parse import quote


def write_impact_slides(
    rows: list[dict],
    output_dir: str | Path,
    *,
    title: str = "Module impact analysis",
) -> Path:
    """Write slides.qmd beside existing plot/audio artifacts and return its path.

    Accepts analyze_module_impact results or rows loaded from results.csv.
    Trials are ordered by descending MAE. Missing plots and failed trials get
    explanatory slides. Render with `quarto render slides.qmd`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        rows,
        key=lambda row: float(row["mae"]) if row.get("mae") not in (None, "") else -1,
        reverse=True,
    )
    parts = [
        "---\nformat:\n  revealjs:\n    theme: dark\n    slide-number: true\n"
        "    transition: fade\n    width: 1600\n    height: 1000\n"
        "    margin: 0.08\n    center: false\n    slide-level: 2\n---\n",
        f"## {escape(title)}\n",
        f"{len(rows)} trials · ranked by decreasing MAE\n",
        "Reference: the unmodified model reconstruction. Metrics use all channels "
        "without amplitude normalization; plots show the first channel.\n",
    ]
    baseline = output_dir / "baseline.wav"
    if baseline.is_file():
        parts.append(_audio("baseline.wav", "Baseline"))
    for row in ranked:
        parts.extend([
            f"## {escape(str(row['layer_path']))} · {escape(str(row['operation']))}\n",
            f"<p style=\"font-size:0.65em\">{escape(str(row['layer_type']))} "
            f"· <code>{escape(str(row['parameters']))}</code></p>\n",
        ])
        if row.get("mae") not in (None, "") and row.get("mrstft") not in (None, ""):
            parts.append(f"MAE **{float(row['mae']):.6g}** · MRSTFT **{float(row['mrstft']):.6g}**\n")
        plot = row.get("plot")
        if plot and (output_dir / plot).is_file():
            parts.append(f'![Baseline and warped audio comparison]({quote(str(plot), safe="/")}){{height=600px}}\n')
        else:
            parts.append("Comparison plot unavailable for this trial.\n")
        if row.get("error"):
            parts.append(f"<p style=\"font-size:0.6em\">{escape(str(row['error']))}</p>\n")
        audio = row.get("audio")
        if audio and (output_dir / audio).is_file():
            parts.append(_audio(str(audio), "Warped audio"))
    path = output_dir / "slides.qmd"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _audio(filename: str, label: str) -> str:
    return (
        f'<div style="font-size:0.55em">{escape(label)} '
        f'<audio controls preload="none" src="{quote(filename, safe="/")}" '
        f'aria-label="{escape(label, quote=True)}"></audio></div>\n'
    )
