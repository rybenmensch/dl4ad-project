import numpy as np
import librosa

def normalize_audio(audio: np.ndarray, target_rms: float = 0.1, peak_limit: float = 0.99) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    rms = np.sqrt(np.mean(audio**2))
    if rms > 1e-8:
        audio = audio * (target_rms / rms)
    peak = np.max(np.abs(audio))
    if peak > peak_limit:
        audio = audio * (peak_limit / peak)
    return audio


def classify_audio_state(audio: np.ndarray, mrstft_loss: float | None = None) -> tuple[str, int]:
    if not np.all(np.isfinite(audio)):
        return "INVALID", 5

    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-5:
        return "SILENT", 4

    peak_ratio = np.mean(np.abs(audio) >= 0.99)
    if peak_ratio > 0.05:
        return "CLIPPED", 3

    if mrstft_loss is not None and mrstft_loss > 5.0:
        return "DISTORTED", 2

    if mrstft_loss is not None and mrstft_loss > 1.5:
        return "SLIGHT_DISTORTION", 1

    return "VALID", 0


def compute_audio_features(audio: np.ndarray, sr: int) -> dict:
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr).ravel()
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr).ravel()
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr).ravel()
    rms = librosa.feature.rms(y=audio).ravel()

    return {
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
        "spectral_rolloff": float(np.mean(rolloff)),
        "rms_energy": float(np.mean(rms)),
    }


def compute_error_metrics(baseline: np.ndarray, candidate: np.ndarray) -> dict:
    base = np.asarray(baseline, dtype=np.float32)
    cand = np.asarray(candidate, dtype=np.float32)

    raw_mae = float(np.mean(np.abs(base - cand)))
    raw_rms = float(np.sqrt(np.mean(cand**2)))

    base_norm = normalize_audio(base)
    cand_norm = normalize_audio(cand)

    mae_norm = float(np.mean(np.abs(base_norm - cand_norm)))

    return {
        "raw_mae": raw_mae,
        "mae_normalized": mae_norm,
        "rms_before": float(np.sqrt(np.mean(base**2))),
        "rms_after": raw_rms,
        "peak_before": float(np.max(np.abs(base))),
        "peak_after": float(np.max(np.abs(cand))),
    }