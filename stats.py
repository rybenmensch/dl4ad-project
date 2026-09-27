import os
import pandas as pd

def aggregate_results(df: pd.DataFrame, output_dir: str = "results"):
    if df.empty:
        return pd.DataFrame()

    def iqr(x):
        return x.quantile(0.75) - x.quantile(0.25)

    group_columns = ["layer", "operation"]
    if "parameters" in df.columns:
        group_columns.append("parameters")

    summary = (
        df.groupby(group_columns, as_index=False)
        .agg(
            runs=("audio_file", "count"),
            mae_median=("mae_normalized", "median"),
            mae_iqr=("mae_normalized", iqr),
            mae_mean=("mae_normalized", "mean"),
            raw_mae_median=("raw_mae", "median"),
            mrstft_median=("mrstft", "median"),
            valid_ratio=("status", lambda s: s.isin(["VALID", "CLIPPED"]).mean()),
            clipped_ratio=("clipping", "mean"),
            silent_ratio=("status", lambda s: (s == "SILENT").mean()),
            invalid_ratio=("status", lambda s: s.isin(["INVALID", "ERROR"]).mean()),
        )
        .sort_values("mae_median", ascending=False)
    )

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "statistical_summary.csv")
    summary.to_csv(summary_path, index=False)
    return summary