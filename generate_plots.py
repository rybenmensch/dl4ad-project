import matplotlib
matplotlib.use('Agg')  # Zwingt Matplotlib dazu, rein im Hintergrund zu rendern
from pathlib import Path
import pandas as pd
import torchaudio
from plotting import plot_comparison

# Ordner definieren
output_dir = Path("test_output")
csv_path = output_dir / "results.csv"
baseline_path = output_dir / "baseline.wav"

# Prüfen, ob Baseline und CSV da sind
if not csv_path.exists() or not baseline_path.exists():
    print("Fehler: results.csv oder baseline.wav nicht gefunden!")
    exit()

print("Lade Baseline-Audio...")
baseline, sr = torchaudio.load(baseline_path)

print("Lade Ergebnisse aus CSV...")
df = pd.read_csv(csv_path)

print("Generiere fehlende Plots nachträglich...")
for index, row in df.iterrows():
    audio_filename = row.get("audio")
    if pd.isna(audio_filename):
        continue
        
    audio_path = output_dir / audio_filename
    if not audio_path.exists():
        continue
        
    # Entsprechenden Namen für das PNG ableiten (gleicher Stem wie die Audio-Datei)
    stem = audio_path.stem
    plot_path = output_dir / f"{stem}.png"
    
    # Wenn der Plot schon existiert, überspringen
    if plot_path.exists():
        continue
        
    try:
        reconstruction, _ = torchaudio.load(audio_path)
        
        # Plot erzeugen und speichern
        plot_comparison(
            baseline, 
            reconstruction, 
            sr,
            title=f"{row['layer_path']}: {row['operation']} {row['parameters']}",
            save_path=str(plot_path), 
            show=False
        )
        
        # CSV-Eintrag aktualisieren
        df.loc[index, "plot"] = plot_path.name
        print(f"Erstellt: {plot_path.name}")
        
    except Exception as e:
        print(f"Fehler bei Zeile {index} ({audio_filename}): {e}")

# Aktualisierte CSV speichern
df.to_csv(csv_path, index=False)
print("Fertig! Alle Plots wurden nachträglich generiert.")