import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import seaborn as sns

from generate_paper_heatmap import (
    INTERVENTION_COLUMNS,
    SHORT_COLUMN_LABELS,
    intervention_column,
    layer_sort_key,
    short_layer_label,
)


def generate_dual_heatmap(raw_csv: Path, output: Path) -> None:
    data = pd.read_csv(raw_csv)
    required_columns = {
        "layer",
        "operation",
        "parameters",
        "mae_normalized",
        "mrstft",
    }
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"CSV is missing required columns: {missing}")

    data["intervention_column"] = data.apply(intervention_column, axis=1)
    clipping_flag = data["clipping"] if "clipping" in data else pd.Series(
        False, index=data.index
    )
    data["is_clipped"] = clipping_flag.astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    if "status" in data:
        data["is_clipped"] |= data["status"].astype(str).str.upper().eq("CLIPPED")

    sorted_layers = sorted(data["layer"].unique(), key=layer_sort_key)
    available_columns = [
        column for column in INTERVENTION_COLUMNS
        if column in data["intervention_column"].unique()
    ]

    def make_pivot(metric: str) -> pd.DataFrame:
        return data.pivot_table(
            index="layer",
            columns="intervention_column",
            values=metric,
            aggfunc="median",
        ).reindex(index=sorted_layers, columns=available_columns)

    mae_data = make_pivot("mae_normalized")
    mrstft_data = make_pivot("mrstft")
    clipped_data = data.pivot_table(
        index="layer",
        columns="intervention_column",
        values="is_clipped",
        aggfunc="max",
        fill_value=False,
    ).reindex(index=sorted_layers, columns=available_columns, fill_value=False)

    figure, axes = plt.subplots(
        ncols=2,
        figsize=(6.8, 5.2),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.08},
    )
    panels = [
        (axes[0], mae_data, "YlOrRd", "Median normalized MAE"),
        (axes[1], mrstft_data, "YlGnBu", "Median MR-STFT"),
    ]

    for panel_index, (axis, values, palette, colorbar_label) in enumerate(panels):
        color_map = plt.colormaps.get_cmap(palette).copy()
        color_map.set_bad("whitesmoke")
        sns.heatmap(
            values,
            annot=True,
            fmt=".3f",
            annot_kws={"size": 5.0},
            xticklabels=False,
            yticklabels=False,
            cmap=color_map,
            cbar_kws={
                "label": colorbar_label,
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

        axis.set_xticks(
            [column_index + 0.5 for column_index in range(len(available_columns))]
        )
        axis.set_xticklabels(
            [SHORT_COLUMN_LABELS[column] for column in available_columns],
            rotation=0,
            fontsize=7,
        )
        axis.set_xlabel(
            "Intervention setting" if panel_index == 0 else "",
            fontsize=8,
        )
        axis.set_title("(a) MAE" if panel_index == 0 else "(b) MR-STFT", fontsize=9)

    axes[0].set_ylabel("Model layer", fontsize=8)
    axes[0].set_yticks([row_index + 0.5 for row_index in range(len(sorted_layers))])
    axes[0].set_yticklabels(
        [short_layer_label(layer) for layer in sorted_layers],
        rotation=0,
        fontsize=7,
    )

    encoder_count = sum(str(layer).startswith("encoder.") for layer in sorted_layers)
    if 0 < encoder_count < len(sorted_layers):
        for axis in axes:
            axis.axhline(encoder_count, color="black", linewidth=0.8)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, bbox_inches="tight")
    png_output = output.with_suffix(".png")
    figure.savefig(png_output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved dual heatmap to {output} and {png_output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a dual MAE/MR-STFT heatmap for the paper."
    )
    parser.add_argument("raw_csv", type=Path, help="Full corpus raw_results.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/interventions_heatmap_dual.pdf"),
        help="Output PDF path",
    )
    arguments = parser.parse_args()
    generate_dual_heatmap(arguments.raw_csv, arguments.output)


if __name__ == "__main__":
    main()
