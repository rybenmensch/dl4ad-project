from analysis_kit import analyze_module_impact
from encodec_lib import EncodecNNModel, raw_encodec_model
import torchaudio
from pathlib import Path

# 1. Encodec-Modell laden (hier mit 48 kHz Standard-Sample-Rate)
print("Lade Encodec Modell...")
raw_model = raw_encodec_model(48_000)
model = EncodecNNModel(raw_model)

# 2. Test-Audio aus deinem Projektordner laden
audio_path = Path("audio/source/GLM.wav") # Alternativ "audio/source/tof.wav"
print(f"Lade Audio-Datei von '{audio_path}'...")
wav, sr = torchaudio.load(audio_path)

# 3. Analyse aufrufen und Ergebnisse speichern
output_folder = Path("test_output")
print("Starte Analyse-Modul...")
results = analyze_module_impact(
    model=model,
    wav=(wav, sr),
    output_dir=output_folder,
    nets=("encoder", "decoder"),
    plots=True
)

print(f"Fertig! Ergebnisse wurden in '{output_folder}' gespeichert.")