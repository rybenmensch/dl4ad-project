import numpy as np

def normalize_audio(audio: np.ndarray, target_rms: float = 0.1, peak_limit: float = 0.99) -> np.ndarray:
    """
    Normalisiert ein Audiosignal auf einen Ziel-RMS-Wert und limitiert den Peak,
    um künstlichen Lautstärke-Bias bei der Metrik-Berechnung zu verhindern.
    """
    # Aktuellen RMS (Root Mean Square) berechnen
    current_rms = np.sqrt(np.mean(audio**2))
    
    if current_rms > 1e-6:  # Schutz vor Division durch Null / Totstille
        audio = audio * (target_rms / current_rms)
    
    # Peak-Limiter, um digitales Übersteuern bei der Normalisierung abzufangen
    peak = np.max(np.abs(audio))
    if peak > peak_limit:
        audio = audio * (peak_limit / peak)
        
    return audio