import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_mono_numpy(x: torch.Tensor) -> np.ndarray:
    """Return the first channel of a [channels, samples] tensor as a NumPy array."""
    return x[0].detach().cpu().numpy()


def plot_comparison(
    clean: torch.Tensor,
    degraded: torch.Tensor,
    sr: int,
    mae: float,
    mrstft: float,
    title: str = "Comparison: clean vs. degraded)",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot baseline and modified model outputs with their precomputed metrics.

    Audio tensors are expected to have shape [channels, samples]. The plots use
    the first channel; MAE and MRSTFT are the all-channel values from analysis.
    """
    clean_np = _to_mono_numpy(clean)
    degraded_np = _to_mono_numpy(degraded)

    # Trim both signals to their shared length before plotting.
    min_len = min(len(clean_np), len(degraded_np))
    clean_np = clean_np[:min_len]
    degraded_np = degraded_np[:min_len]
    time_axis = np.arange(min_len) / sr

    diff_np = clean_np - degraded_np

    plt.rcParams["font.family"] = "Times New Roman"
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    ax = axes[0, 0]

    ylim = max(np.max(np.abs(a)) for a in (clean_np, degraded_np))
    ax.set_ylim(-ylim, ylim)
    ax.plot(
        time_axis,
        clean_np,
        label="clean reconstruction",
        color="#2a78d6",
        linewidth=0.8,
        alpha=0.5,
    )
    ax.plot(
        time_axis,
        degraded_np,
        label="degraded reconstruction",
        color="#e34948",
        linewidth=0.8,
        alpha=0.5,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform comparison")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[0, 1]
    ylim = np.max(np.abs(diff_np))
    ax.set_ylim(-ylim, ylim)
    ax.plot(time_axis, diff_np, color="#7a4fbf", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Difference (clean - degraded)")
    ax.set_title("Difference between reconstructions")
    ax.axhline(0, color="black", linewidth=0.5)

    ax = axes[1, 0]
    ax.specgram(clean_np, Fs=sr, NFFT=1024, noverlap=512, cmap="magma")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Clean spectrogram")

    ax = axes[1, 1]
    ax.specgram(degraded_np, Fs=sr, NFFT=1024, noverlap=512, cmap="magma")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Degraded spectrogram")

    # Use the all-channel metrics already calculated by the analysis sweep.
    fig.suptitle(f"{title}\n", fontsize=12)

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    if save_path is not None:
        fig.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)
