import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _parameters(row: dict) -> dict:
    try:
        return json.loads(row.get("parameters") or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}


def _column_key(row: dict) -> str:
    operation = row["operation"]
    parameters = _parameters(row)
    if operation == "skip":
        return "skip"
    if operation == "repeat":
        return f"repeat_{int(parameters.get('repeats', 1))}"
    if operation == "multiply":
        factor = float(parameters.get("weight_mul", 1.0))
        return f"multiply_{factor:g}"
    if operation == "add":
        weight_add = float(parameters.get("weight_add", 0.0))
        bias_add = float(parameters.get("bias_add", 0.0))
        return f"add_{weight_add:g}_{bias_add:g}"
    return operation


def _column_label(column: str) -> str:
    if column == "skip":
        return "Skip"
    if match := re.fullmatch(r"repeat_(\d+)", column):
        return f"R{match.group(1)}"
    if match := re.fullmatch(r"multiply_([\d.]+)", column):
        return f"M{match.group(1)}"
    if match := re.fullmatch(r"add_([-\d.]+)_([-\d.]+)", column):
        return f"A{match.group(1)}/{match.group(2)}"
    return column


def _layer_sort_key(layer_path: str) -> tuple[int, int]:
    parts = layer_path.split(".")
    network_order = 0 if parts[0] == "encoder" else 1
    try:
        layer_index = int(parts[-1])
    except ValueError:
        layer_index = 0
    return network_order, layer_index


def _layer_label(layer_path: str) -> str:
    network = "E" if layer_path.startswith("encoder.") else "D"
    return f"{network}{layer_path.rsplit('.', 1)[-1]}"


def _setting_sort_key(setting: str) -> tuple[int, float]:
    operation = setting.split("_", 1)[0]
    order = {"skip": 0, "repeat": 1, "multiply": 2, "add": 3}.get(operation, 4)
    if operation == "skip":
        return order, 0.0
    try:
        value = float(
            setting.rsplit("_", 1)[-1]
            if operation == "add"
            else setting.split("_", 1)[1]
        )
    except (IndexError, ValueError):
        value = 0.0
    return order, value


def render_heatmap(
    rows: list[dict],
    metric: str,
    input_name: str,
    output_stem: Path,
) -> Path:
    if metric not in {"mae", "mrstft"}:
        raise ValueError("metric must be 'mae' or 'mrstft'")

    values = []
    for row in rows:
        try:
            value = float(row.get(metric))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        values.append(
            {
                "layer": row["layer_path"],
                "setting": _column_key(row),
                "value": value,
            }
        )
    if not values:
        raise ValueError(f"No valid {metric.upper()} results are available to plot")

    data = pd.DataFrame(values)
    layer_order = sorted(data["layer"].unique(), key=_layer_sort_key)
    setting_order = sorted(data["setting"].unique(), key=_setting_sort_key)
    heatmap = data.pivot_table(
        index="layer", columns="setting", values="value", aggfunc="median"
    ).reindex(index=layer_order, columns=setting_order)

    palette = "YlOrRd" if metric == "mae" else "YlGnBu"
    metric_label = "MAE" if metric == "mae" else "MRSTFT"
    color_map = plt.colormaps.get_cmap(palette).copy()
    color_map.set_bad("whitesmoke")
    plt.rcParams["font.family"] = "Times New Roman"
    figure, axis = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    sns.heatmap(
        heatmap,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 6},
        cmap=color_map,
        xticklabels=[_column_label(column) for column in setting_order],
        yticklabels=[_layer_label(layer) for layer in layer_order],
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": metric_label},
        ax=axis,
    )
    axis.set_title(
        f"{metric_label} by layer and intervention\n{input_name}", fontsize=10
    )
    axis.set_xlabel("Intervention setting")
    axis.set_ylabel("Model layer")
    axis.tick_params(axis="x", labelrotation=35, labelsize=7)
    axis.tick_params(axis="y", labelrotation=0, labelsize=7)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png_output = output_stem.with_suffix(".png")
    figure.savefig(png_output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return png_output
