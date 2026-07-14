import torch
import numpy as np
import matplotlib.pyplot as plt

try:
    import auraloss

    _HAS_AURALOSS = True
except ImportError:
    _HAS_AURALOSS = False


def _to_mono_numpy(x: torch.Tensor) -> np.ndarray:
    """ersten Kanal (Mono) in numpy-Array, für plotten
    Erwartet Shape [Kanaele, Samples]."""
    return x[0].detach().cpu().numpy()


def plot_comparison(
    clean: torch.Tensor,
    degraded: torch.Tensor,
    sr: int,
    title: str = "Comparison: clean vs. degradet)",
    save_path: str | None = None,
    show: bool = True,
) -> dict:
    """
    Vergleicht output_clean (unverändertes Modell) mit
    Output des manipuliertes Modells ("degraded"), für gleichen input.


    clean:    output_clean = model_clean.process_audio(waveform)
    degraded: output_tensor = model.process_audio(waveform)

    Erwartet Tensoren mit Shape [Kanaele, Samples]
    Bei Stereo nur der erste Kanal geplottet

    Gibt ein Dictionary mit den berechneten Metriken zurueck
    """
    clean_np = _to_mono_numpy(clean)
    degraded_np = _to_mono_numpy(degraded)

    # Falls die beiden Signale unterschiedlich lang sind auf die kkürzere Länge
    min_len = min(len(clean_np), len(degraded_np))
    clean_np = clean_np[:min_len]
    degraded_np = degraded_np[:min_len]
    time_axis = np.arange(min_len) / sr

    diff_np = clean_np - degraded_np

    # Metriken
    mean_abs_diff = float(np.mean(np.abs(diff_np)))
    std_abs_diff = float(np.std(np.abs(diff_np)))

    spectral_loss = None
    if _HAS_AURALOSS:
        mrstft = auraloss.freq.MultiResolutionSTFTLoss()
        clean_t = torch.from_numpy(clean_np).float().unsqueeze(0).unsqueeze(0)
        degraded_t = torch.from_numpy(degraded_np).float().unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            spectral_loss = float(mrstft(degraded_t, clean_t))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    # (0,0) Wellenformen übereinander
    ax = axes[0, 0]
    ax.plot(
        time_axis,
        clean_np,
        label="clean model",
        color="#2a78d6",
        linewidth=0.8,
        alpha=0.8,
    )
    ax.plot(
        time_axis,
        degraded_np,
        label="degraded model",
        color="#e34948",
        linewidth=0.8,
        alpha=0.8,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("comparison waveform")
    ax.legend(loc="upper right", fontsize=8)

    # Differenz-Wellenform
    ax = axes[0, 1]
    ax.plot(time_axis, diff_np, color="#7a4fbf", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Difference (clean - degraded)")
    ax.set_title("Difference between outputs")
    ax.axhline(0, color="black", linewidth=0.5)

    # Spektrogramm cleanes Modell
    ax = axes[1, 0]
    ax.specgram(clean_np, Fs=sr, NFFT=1024, noverlap=512, cmap="magma")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram clean")

    # Spektrogramm manipuliertes Modell
    ax = axes[1, 1]
    ax.specgram(degraded_np, Fs=sr, NFFT=1024, noverlap=512, cmap="magma")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram degraded")

    # Titel mit Metriken
    subtitle_parts = [f"mean|diff|={mean_abs_diff:.4f}", f"std={std_abs_diff:.4f}"]
    if spectral_loss is not None:
        subtitle_parts.append(f"auraloss={spectral_loss:.4f}")
    fig.suptitle(f"{title}\n({', '.join(subtitle_parts)})", fontsize=12)

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "mean_abs_diff": mean_abs_diff,
        "std_abs_diff": std_abs_diff,
        "spectral_loss": spectral_loss,
    }
