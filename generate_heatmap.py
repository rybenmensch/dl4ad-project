import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_intervention_column(row):
    """Hilfsfunktion zur feingranularen Spaltenzuordnung (jede Parameterstufe separat)."""
    op = row['operation']
    try:
        params = json.loads(row['parameters'])
    except (json.JSONDecodeError, TypeError):
        params = {}
        
    if op == 'skip':
        return 'skip'
    elif op == 'repeat':
        repeats = params.get('repeats', 3)
        return f'repeat_{repeats}'
    elif op.startswith('mult') or op == 'multiply':
        mul = params.get('weight_mul', params.get('factor', 1.0))
        if mul == 1.0:
            return 'multiply_1.0 (Control)'
        return f'multiply_{int(mul) if mul.is_integer() else mul}'
    elif op.startswith('add') or op == 'add':
        return 'add_0.1'
    return op

def layer_sort_key(layer_name):
    """Sortierung der Layer: Encoder aufsteigend, danach Decoder aufsteigend."""
    parts = layer_name.split('.')
    net_type = 0 if parts[0] == 'encoder' else 1
    num = int(parts[2]) if len(parts) > 2 else 0
    return (net_type, num)

def generate_interventions_heatmap(raw_csv_path: str = "results/raw_results.csv", output_dir: str = "results"):
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(raw_csv_path):
        print(f"Error: '{raw_csv_path}' not found. Run the analysis pipeline first.")
        return
        
    df = pd.read_csv(raw_csv_path)
    df['intervention_col'] = df.apply(get_intervention_column, axis=1)
    
    desired_columns = [
        'skip', 
        'repeat_3', 'repeat_5', 'repeat_10', 
        'multiply_5', 'multiply_10', 'multiply_20', 
        'add_0.1', 
        'multiply_1.0 (Control)'
    ]

    # --- 1. Aggregated heatmap (median across all recordings) ---
    pivot_agg = df.pivot_table(
        index='layer',
        columns='intervention_col',
        values='mae_normalized',
        aggfunc='median'
    )
    
    sorted_layers = sorted(pivot_agg.index, key=layer_sort_key)
    pivot_agg = pivot_agg.loc[sorted_layers]
    existing_cols = [c for c in desired_columns if c in pivot_agg.columns]
    pivot_agg = pivot_agg[existing_cols]
    
    plt.figure(figsize=(12, 9))
    cmap = plt.colormaps.get_cmap("YlOrRd").copy()
    cmap.set_bad("whitesmoke")
    
    sns.heatmap(
        pivot_agg, 
        annot=True, 
        fmt=".3f", 
        cmap=cmap, 
        cbar_kws={'label': 'Median Normalized MAE (n=4)'},
        linewidths=0.5,
        linecolor='lightgray'
    )
    
    plt.title(
        "Intervention Heatmap: Median Normalized MAE\n"
        "EnCodec (48 kHz) | Aggregated across four recordings", 
        fontsize=13, pad=15, weight='bold'
    )
    plt.xlabel("Intervention and Parameter Setting", fontsize=11, labelpad=10)
    plt.ylabel("Model Layer (Layer Path)", fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    
    png_agg = os.path.join(output_dir, "interventions_heatmap.png")
    pdf_agg = os.path.join(output_dir, "interventions_heatmap.pdf")
    plt.savefig(png_agg, dpi=300)
    plt.savefig(pdf_agg, dpi=300)
    plt.close()
    print(f"Aggregated heatmap saved to:\n- {png_agg}\n- {pdf_agg}")

    # --- 2. Individual heatmaps for each recording ---
    audio_files = df['audio_file'].unique()
    for audio_name in audio_files:
        df_audio = df[df['audio_file'] == audio_name]
        
        pivot_audio = df_audio.pivot_table(
            index='layer',
            columns='intervention_col',
            values='mae_normalized',
            aggfunc='mean' # Falls pro Datei mehrere Trials existieren, ansonsten greift mean/median gleich
        )
        
        pivot_audio = pivot_audio.reindex(index=sorted_layers)
        existing_cols_audio = [c for c in desired_columns if c in pivot_audio.columns]
        pivot_audio = pivot_audio[existing_cols_audio]
        
        plt.figure(figsize=(12, 9))
        sns.heatmap(
            pivot_audio, 
            annot=True, 
            fmt=".3f", 
            cmap=cmap, 
            cbar_kws={'label': 'Normalized MAE'},
            linewidths=0.5,
            linecolor='lightgray'
        )
        
        clean_name = os.path.splitext(audio_name)[0]
        plt.title(
            f"Intervention Heatmap for Stimulus: {audio_name}\n"
            "EnCodec (48 kHz)", 
            fontsize=13, pad=15, weight='bold'
        )
        plt.xlabel("Intervention and Parameter Setting", fontsize=11, labelpad=10)
        plt.ylabel("Model Layer (Layer Path)", fontsize=11, labelpad=10)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        
        png_single = os.path.join(output_dir, f"interventions_heatmap_{clean_name}.png")
        pdf_single = os.path.join(output_dir, f"interventions_heatmap_{clean_name}.pdf")
        plt.savefig(png_single, dpi=300)
        plt.savefig(pdf_single, dpi=300)
        plt.close()
        print(f"Individual heatmap for '{audio_name}' saved to:\n- {png_single}")

if __name__ == "__main__":
    generate_interventions_heatmap()