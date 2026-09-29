import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns


INTERVENTION_COLUMNS = [
    "skip",
    "repeat_3",
    "repeat_5",
    "repeat_10",
    "multiply_1.0 (Control)",
    "multiply_5",
    "multiply_10",
    "multiply_20",
    "add_0.1",
]

SHORT_COLUMN_LABELS = {
    "skip": "Skip",
    "repeat_3": "R3",
    "repeat_5": "R5",
    "repeat_10": "R10",
    "multiply_1.0 (Control)": "Ctrl",
    "multiply_5": "M5",
    "multiply_10": "M10",
    "multiply_20": "M20",
    "add_0.1": "A.1",
}


def intervention_column(row: pd.Series) -> str:
    operation = str(row["operation"])
    try:
        parameters = json.loads(row["parameters"])
    except (json.JSONDecodeError, TypeError):
        parameters = {}

    if operation == "skip":
        return "skip"
    if operation == "repeat":
        return f"repeat_{parameters.get('repeats', 3)}"
    if operation.startswith("mult") or operation == "multiply":
        factor = float(parameters.get("weight_mul", parameters.get("factor", 1.0)))
        return "multiply_1.0 (Control)" if factor == 1.0 else f"multiply_{factor:g}"
    if operation.startswith("add") or operation == "add":
        return "add_0.1"
    return operation


def layer_sort_key(layer_path: str) -> tuple[int, int]:
    parts = str(layer_path).split(".")
    network_order = 0 if parts[0] == "encoder" else 1
    try:
        layer_index = int(parts[-1])
    except ValueError:
        layer_index = 0
    return network_order, layer_index


def short_layer_label(layer_path: str) -> str:
    network_label = "E" if str(layer_path).startswith("encoder.") else "D"
    return f"{network_label}{str(layer_path).rsplit('.', 1)[-1]}"


def generate_paper_heatmap(raw_csv: Path, output: Path) -> None:
    data = pd.read_csv(raw_csv)
    required_columns = {"layer", "operation", "parameters", "mae_normalized", "audio_file"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {missing}")

    data["intervention_column"] = data.apply(intervention_column, axis=1)
    clipping_flag = data.get("clipping", False)
    data["is_clipped"] = clipping_flag.astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    if "status" in data:
        data["is_clipped"] |= data["status"].astype(str).str.upper().eq("CLIPPED")
    heatmap_data = data.pivot_table(
        index="layer",
        columns="intervention_column",
        values="mae_normalized",
        aggfunc="median",
    )

    sorted_layers = sorted(heatmap_data.index, key=layer_sort_key)
    available_columns = [
        column for column in INTERVENTION_COLUMNS if column in heatmap_data.columns
    ]
    heatmap_data = heatmap_data.loc[sorted_layers, available_columns]
    clipped_data = data.pivot_table(
        index="layer",
        columns="intervention_column",
        values="is_clipped",
        aggfunc="max",
        fill_value=False,
    ).reindex(index=sorted_layers, columns=available_columns, fill_value=False)

    figure, axis = plt.subplots(figsize=(3.5, 5.2))
    color_map = plt.colormaps.get_cmap("YlOrRd").copy()
    color_map.set_bad("whitesmoke")
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 5.5},
        xticklabels=False,
        yticklabels=False,
        cmap=color_map,
        cbar_kws={
            "label": "Median normalized MAE",
            "orientation": "horizontal",
            "pad": 0.08,
            "shrink": 0.82,
        },
        linewidths=0.25,
        linecolor="white",
        ax=axis,
    )

    for row_index, layer in enumerate(sorted_layers):
        for column_index, column in enumerate(available_columns):
            if bool(clipped_data.loc[layer, column]):
                axis.add_patch(
                    Rectangle(
                        (column_index, row_index),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.2,
                    )
                )

    encoder_count = sum(str(layer).startswith("encoder.") for layer in sorted_layers)
    if 0 < encoder_count < len(sorted_layers):
        axis.axhline(encoder_count, color="black", linewidth=0.8)

    axis.set_xticks([column_index + 0.5 for column_index in range(len(available_columns))])
    axis.set_xticklabels(
        [SHORT_COLUMN_LABELS[column] for column in available_columns],
        rotation=0,
        fontsize=7,
    )
    axis.set_yticks([layer_index + 0.5 for layer_index in range(len(sorted_layers))])
    axis.set_yticklabels(
        [short_layer_label(layer) for layer in sorted_layers],
        rotation=0,
        fontsize=7,
    )
    axis.set_xlabel("Intervention setting", fontsize=8)
    axis.set_ylabel("Model layer", fontsize=8)
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    png_output = output.with_suffix(".png")
    figure.savefig(png_output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved single-column heatmap to {output} and {png_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a compact, single-column heatmap for the paper."
    )
    parser.add_argument("raw_csv", type=Path, help="Full corpus raw_results.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/interventions_heatmap_single_column.pdf"),
        help="Output PDF path",
    )
    arguments = parser.parse_args()
    generate_paper_heatmap(arguments.raw_csv, arguments.output)


if __name__ == "__main__":
    main()