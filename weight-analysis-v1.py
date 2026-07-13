import csv
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import auraloss
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from torch.nn.utils import remove_weight_norm

from lib import *


# ============================================================
# 1. CONFIGURATION
# ============================================================

run_path = Path("./models/run/")
source_path = check_path("./audio/source/")
reconstructed_root = check_path("./audio/")

file_name = "GLM.wav"

scaling_factors = np.array([
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
    3.00,
    4.00,
    6.00,
    8.00,
    10.00,
])

depth_positions = {
    "early": 0.10,
    "middle": 0.50,
    "late": 0.90,
}

save_factors = {
    float(scaling_factors[0]),
    1.0,
    float(scaling_factors[len(scaling_factors) // 2]),
    float(scaling_factors[-1]),
}

feature_n_fft = 2048
feature_hop_length = 512
feature_rolloff_percent = 0.85


# ============================================================
# 2. CREATE A NEW RESULT DIRECTORY FOR THIS RUN
# ============================================================

run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
source_stem = Path(file_name).stem

result_directory = (
    reconstructed_root
    / "results"
    / f"{source_stem}_{run_timestamp}"
)

encoder_audio_directory = result_directory / "encoder"
decoder_audio_directory = result_directory / "decoder"
analysis_directory = result_directory / "analysis"

encoder_audio_directory.mkdir(parents=True, exist_ok=False)
decoder_audio_directory.mkdir(parents=True, exist_ok=False)
analysis_directory.mkdir(parents=True, exist_ok=False)

section_directories = {
    "encoder": encoder_audio_directory,
    "decoder": decoder_audio_directory,
}

print(f"Results for this run will be saved to:\n  {result_directory}")


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def natural_sort_key(text: str) -> list[Any]:
    """Sort strings containing numbers naturally."""
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def get_submodule(
    root_module: torch.nn.Module,
    module_path: str,
) -> torch.nn.Module:
    """Navigate to a nested module using a dotted module path."""
    module = root_module

    for part in module_path.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)

    return module


def align_audio(
    first: torch.Tensor,
    second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align channel count and sample count for comparison."""
    if first.ndim == 1:
        first = first.unsqueeze(0)

    if second.ndim == 1:
        second = second.unsqueeze(0)

    minimum_channels = min(first.shape[0], second.shape[0])
    minimum_samples = min(first.shape[-1], second.shape[-1])

    return (
        first[:minimum_channels, :minimum_samples],
        second[:minimum_channels, :minimum_samples],
    )


def safe_db(
    value: float,
    epsilon: float = 1e-12,
) -> float:
    """Convert a linear amplitude value to decibels."""
    return 20.0 * math.log10(max(value, epsilon))


def format_factor(factor: float) -> str:
    """Convert a factor to a filename-safe string."""
    return f"{factor:.2f}".replace(".", "_")


def factor_is_saved(factor: float) -> bool:
    """Check whether audio should be saved for a scaling factor."""
    return any(
        np.isclose(factor, save_factor)
        for save_factor in save_factors
    )


def shorten_module_name(
    module_path: str,
    maximum_length: int = 45,
) -> str:
    """Shorten long module paths for plot titles."""
    if len(module_path) <= maximum_length:
        return module_path

    return "..." + module_path[-maximum_length:]


# ============================================================
# 4. FIND ENCODER AND DECODER WEIGHT MODULES
# ============================================================

def parameter_key_to_module_path(
    parameter_key: str,
) -> str | None:
    """Convert a state-dict weight key into its module path."""
    supported_suffixes = (
        ".weight",
        ".weight_g",
        ".weight_v",
        ".weight_orig",
    )

    for suffix in supported_suffixes:
        if parameter_key.endswith(suffix):
            return parameter_key[:-len(suffix)]

    return None


def collect_weight_module_paths(
    model: torch.nn.Module,
    section_name: str,
) -> list[str]:
    """Collect valid weight-bearing modules in one model section."""
    module_paths = set()

    for parameter_key in model.state_dict().keys():
        if section_name.lower() not in parameter_key.lower():
            continue

        module_path = parameter_key_to_module_path(parameter_key)

        if module_path is not None:
            module_paths.add(module_path)

    sorted_paths = sorted(
        module_paths,
        key=natural_sort_key,
    )

    valid_paths = []

    for module_path in sorted_paths:
        try:
            module = get_submodule(model, module_path)

            has_weight = (
                hasattr(module, "weight")
                or hasattr(module, "weight_g")
                or hasattr(module, "weight_v")
            )

            if has_weight:
                valid_paths.append(module_path)

        except (
            AttributeError,
            IndexError,
            KeyError,
            TypeError,
        ):
            continue

    return valid_paths


def select_representative_depths(
    module_paths: list[str],
    positions: dict[str, float],
) -> list[tuple[str, str]]:
    """Select early, middle and late modules."""
    if not module_paths:
        return []

    selected = []
    used_paths = set()

    for depth_name, relative_position in positions.items():
        index = round(
            (len(module_paths) - 1) * relative_position
        )

        index = max(
            0,
            min(index, len(module_paths) - 1),
        )

        module_path = module_paths[index]

        if module_path not in used_paths:
            selected.append((depth_name, module_path))
            used_paths.add(module_path)

    return selected


def remove_module_weight_norm(
    module: torch.nn.Module,
) -> bool:
    """Remove legacy PyTorch weight normalization."""
    try:
        remove_weight_norm(module)
        return True
    except (ValueError, AttributeError):
        return False


def scale_module_weight(
    model: torch.nn.Module,
    module_path: str,
    factor: float,
) -> dict[str, Any]:
    """Scale the effective weight of a selected module."""
    module = get_submodule(model, module_path)

    weight_norm_removed = remove_module_weight_norm(module)

    if not hasattr(module, "weight"):
        raise AttributeError(
            f"Module '{module_path}' does not expose a weight tensor."
        )

    with torch.no_grad():
        original_weight = module.weight.detach().clone()
        module.weight.mul_(factor)

    return {
        "module_path": module_path,
        "factor": factor,
        "shape": list(original_weight.shape),
        "original_mean": original_weight.mean().item(),
        "original_std": original_weight.std().item(),
        "original_norm": original_weight.norm().item(),
        "scaled_norm": module.weight.detach().norm().item(),
        "weight_norm_removed": weight_norm_removed,
    }


# ============================================================
# 5. AUDIO FEATURE EXTRACTION
# ============================================================

def calculate_zero_crossing_rate(
    waveform: torch.Tensor,
) -> float:
    """Calculate the mean zero-crossing rate."""
    if waveform.numel() < 2:
        return 0.0

    signs = torch.sign(waveform)
    crossings = signs[1:] != signs[:-1]

    return crossings.float().mean().item()


def extract_audio_features(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    rolloff_percent: float = 0.85,
) -> dict[str, float]:
    """Extract global time-domain and spectral features."""
    audio = waveform.detach().float().cpu()

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    if audio.ndim != 2:
        raise ValueError(
            "Expected audio in [channels, samples] format, "
            f"but received {list(audio.shape)}."
        )

    mono = audio.mean(dim=0)

    duration_seconds = mono.shape[-1] / sample_rate

    rms = torch.sqrt(
        torch.mean(mono.square()) + 1e-12
    ).item()

    peak = torch.max(torch.abs(mono)).item()
    crest_factor = peak / max(rms, 1e-12)

    zero_crossing_rate = calculate_zero_crossing_rate(mono)

    effective_n_fft = min(n_fft, mono.shape[-1])

    if effective_n_fft < 2:
        raise ValueError(
            "The audio signal is too short for spectral analysis."
        )

    effective_hop_length = min(
        hop_length,
        max(1, effective_n_fft // 4),
    )

    window = torch.hann_window(
        effective_n_fft,
        dtype=mono.dtype,
        device=mono.device,
    )

    stft = torch.stft(
        mono,
        n_fft=effective_n_fft,
        hop_length=effective_hop_length,
        win_length=effective_n_fft,
        window=window,
        return_complex=True,
        center=True,
    )

    magnitude = stft.abs()
    power = magnitude.square()

    frequencies = torch.fft.rfftfreq(
        effective_n_fft,
        d=1.0 / sample_rate,
    )

    frequencies_column = frequencies.unsqueeze(1)
    magnitude_sum = magnitude.sum(dim=0).clamp_min(1e-12)

    frame_centroids = (
        frequencies_column * magnitude
    ).sum(dim=0) / magnitude_sum

    spectral_centroid = frame_centroids.mean().item()

    frame_bandwidth = torch.sqrt(
        (
            (
                frequencies_column
                - frame_centroids.unsqueeze(0)
            ).square()
            * magnitude
        ).sum(dim=0)
        / magnitude_sum
    )

    spectral_bandwidth = frame_bandwidth.mean().item()

    cumulative_power = torch.cumsum(power, dim=0)
    total_power = cumulative_power[-1].clamp_min(1e-12)
    rolloff_threshold = rolloff_percent * total_power

    rolloff_indices = (
        cumulative_power
        >= rolloff_threshold.unsqueeze(0)
    ).float().argmax(dim=0)

    spectral_rolloff = (
        frequencies[rolloff_indices]
        .float()
        .mean()
        .item()
    )

    spectral_flatness_per_frame = (
        torch.exp(
            torch.mean(
                torch.log(power.clamp_min(1e-12)),
                dim=0,
            )
        )
        / torch.mean(
            power.clamp_min(1e-12),
            dim=0,
        )
    )

    spectral_flatness = (
        spectral_flatness_per_frame.mean().item()
    )

    return {
        "duration_seconds": duration_seconds,
        "rms": rms,
        "rms_dbfs": safe_db(rms),
        "peak": peak,
        "peak_dbfs": safe_db(peak),
        "crest_factor": crest_factor,
        "crest_factor_db": safe_db(crest_factor),
        "zero_crossing_rate": zero_crossing_rate,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_bandwidth_hz": spectral_bandwidth,
        "spectral_rolloff_hz": spectral_rolloff,
        "spectral_flatness": spectral_flatness,
    }


def calculate_feature_deltas(
    features: dict[str, float],
    reference_features: dict[str, float],
) -> dict[str, float]:
    """Calculate feature differences from a reference signal."""
    return {
        f"{feature_name}_delta": (
            value - reference_features[feature_name]
        )
        for feature_name, value in features.items()
        if feature_name in reference_features
    }


# ============================================================
# 6. CSV OUTPUT
# ============================================================

def save_dict_rows_to_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save dictionary rows to a CSV file."""
    if not rows:
        print(f"No data available for {output_path.name}")
        return

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved table: {output_path}")


# ============================================================
# 7. LOAD AUDIO AND CLEAN MODEL
# ============================================================

input_path = source_path / file_name

if not input_path.exists():
    raise FileNotFoundError(
        f"Source audio does not exist: {input_path}"
    )

waveform, sample_rate = torchaudio.load(input_path)

print(f"\nLoaded source: {input_path}")
print(f"Source shape: {list(waveform.shape)}")
print(f"Sample rate: {sample_rate} Hz")

print("\nLoading clean baseline model...")

model_clean = rave_from_checkpoint(str(run_path))
model_clean.eval()

with torch.no_grad():
    output_clean = process_audio(
        model_clean,
        waveform,
    )

waveform_cpu = waveform.detach().cpu().float()
output_clean = output_clean.detach().cpu().float()

clean_output_path = (
    result_directory
    / f"{source_stem}_clean_reconstruction.wav"
)

torchaudio.save(
    str(clean_output_path),
    output_clean,
    sample_rate,
)

print(f"Saved clean reconstruction: {clean_output_path}")


# ============================================================
# 8. SOURCE AND CLEAN FEATURES
# ============================================================

source_features = extract_audio_features(
    waveform_cpu,
    sample_rate,
    n_fft=feature_n_fft,
    hop_length=feature_hop_length,
    rolloff_percent=feature_rolloff_percent,
)

clean_features = extract_audio_features(
    output_clean,
    sample_rate,
    n_fft=feature_n_fft,
    hop_length=feature_hop_length,
    rolloff_percent=feature_rolloff_percent,
)

feature_rows = [
    {
        "signal": "source",
        "network_section": "",
        "depth": "",
        "module_path": "",
        "scaling_factor": "",
        **source_features,
    },
    {
        "signal": "clean_reconstruction",
        "network_section": "",
        "depth": "",
        "module_path": "",
        "scaling_factor": 1.0,
        **clean_features,
        **calculate_feature_deltas(
            clean_features,
            source_features,
        ),
    },
]


# ============================================================
# 9. SELECT ENCODER AND DECODER MODULES
# ============================================================

encoder_module_paths = collect_weight_module_paths(
    model_clean,
    "encoder",
)

decoder_module_paths = collect_weight_module_paths(
    model_clean,
    "decoder",
)

encoder_targets = select_representative_depths(
    encoder_module_paths,
    depth_positions,
)

decoder_targets = select_representative_depths(
    decoder_module_paths,
    depth_positions,
)

experiments = [
    ("encoder", depth_name, module_path)
    for depth_name, module_path in encoder_targets
]

experiments.extend([
    ("decoder", depth_name, module_path)
    for depth_name, module_path in decoder_targets
])

print("\nSelected experiment modules:")

for section, depth_name, module_path in experiments:
    print(
        f"  {section:>7} {depth_name:>6}: "
        f"{module_path}"
    )

if not experiments:
    raise RuntimeError(
        "No encoder or decoder weight modules were found."
    )


# ============================================================
# 10. RUN PERTURBATION EXPERIMENTS
# ============================================================

mrstft = auraloss.freq.MultiResolutionSTFTLoss()

metric_rows = []
experiment_results = {}

for network_section, depth_name, module_path in experiments:
    experiment_name = f"{network_section}_{depth_name}"

    print(
        f"\nEvaluating {network_section} {depth_name}:\n"
        f"  {module_path}"
    )

    factor_results = []

    for factor in scaling_factors:
        factor = float(factor)

        model_manipulated = rave_from_checkpoint(
            str(run_path)
        )

        model_manipulated.eval()

        weight_information = scale_module_weight(
            model_manipulated,
            module_path,
            factor,
        )

        with torch.no_grad():
            manipulated_output = process_audio(
                model_manipulated,
                waveform,
            )

        manipulated_output = (
            manipulated_output
            .detach()
            .cpu()
            .float()
        )

        comparable_output, comparable_clean = align_audio(
            manipulated_output,
            output_clean,
        )

        difference = comparable_output - comparable_clean
        absolute_difference = torch.abs(difference)

        mean_absolute_difference = (
            absolute_difference.mean().item()
        )

        absolute_difference_std = (
            absolute_difference.std().item()
        )

        root_mean_squared_error = torch.sqrt(
            torch.mean(difference.square())
        ).item()

        mrstft_loss = mrstft(
            comparable_output.unsqueeze(0),
            comparable_clean.unsqueeze(0),
        ).item()

        output_features = extract_audio_features(
            manipulated_output,
            sample_rate,
            n_fft=feature_n_fft,
            hop_length=feature_hop_length,
            rolloff_percent=feature_rolloff_percent,
        )

        clean_feature_deltas = calculate_feature_deltas(
            output_features,
            clean_features,
        )

        source_feature_deltas = {
            key.replace("_delta", "_delta_vs_source"): value
            for key, value in calculate_feature_deltas(
                output_features,
                source_features,
            ).items()
        }

        metric_row = {
            "network_section": network_section,
            "depth": depth_name,
            "module_path": module_path,
            "scaling_factor": factor,
            "mean_absolute_difference": (
                mean_absolute_difference
            ),
            "absolute_difference_std": (
                absolute_difference_std
            ),
            "rmse": root_mean_squared_error,
            "mrstft_loss": mrstft_loss,
            "weight_shape": str(
                weight_information["shape"]
            ),
            "weight_mean": (
                weight_information["original_mean"]
            ),
            "weight_std": (
                weight_information["original_std"]
            ),
            "original_weight_norm": (
                weight_information["original_norm"]
            ),
            "scaled_weight_norm": (
                weight_information["scaled_norm"]
            ),
            "weight_norm_removed": (
                weight_information["weight_norm_removed"]
            ),
        }

        feature_row = {
            "signal": "manipulated_output",
            "network_section": network_section,
            "depth": depth_name,
            "module_path": module_path,
            "scaling_factor": factor,
            **output_features,
            **clean_feature_deltas,
            **source_feature_deltas,
        }

        combined_result = {
            **metric_row,
            **output_features,
            **clean_feature_deltas,
        }

        metric_rows.append(metric_row)
        feature_rows.append(feature_row)
        factor_results.append(combined_result)

        if factor_is_saved(factor):
            factor_text = format_factor(factor)

            audio_directory = section_directories[
                network_section
            ]

            output_audio_path = audio_directory / (
                f"{source_stem}_"
                f"{depth_name}_"
                f"factor_{factor_text}.wav"
            )

            torchaudio.save(
                str(output_audio_path),
                manipulated_output,
                sample_rate,
            )

            print(
                f"  factor {factor:>5.2f}: "
                f"saved {output_audio_path.name}"
            )
        else:
            print(f"  factor {factor:>5.2f}: analyzed")

        del model_manipulated

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    experiment_results[experiment_name] = factor_results


# ============================================================
# 11. SAVE CSV RESULTS
# ============================================================

metrics_csv_path = (
    analysis_directory
    / "perturbation_metrics.csv"
)

features_csv_path = (
    analysis_directory
    / "audio_features.csv"
)

save_dict_rows_to_csv(
    metric_rows,
    metrics_csv_path,
)

save_dict_rows_to_csv(
    feature_rows,
    features_csv_path,
)


# ============================================================
# 12. COMBINED ANALYSIS DIAGRAM
# ============================================================

plot_definitions = [
    {
        "key": "mean_absolute_difference",
        "title": "Mean absolute difference",
        "ylabel": "Amplitude difference",
        "clean_reference": 0.0,
        "source_reference": None,
    },
    {
        "key": "mrstft_loss",
        "title": "Multi-resolution STFT loss",
        "ylabel": "MRSTFT loss",
        "clean_reference": 0.0,
        "source_reference": None,
    },
    {
        "key": "rms_dbfs",
        "title": "RMS level",
        "ylabel": "dBFS",
        "clean_reference": clean_features["rms_dbfs"],
        "source_reference": source_features["rms_dbfs"],
    },
    {
        "key": "spectral_centroid_hz",
        "title": "Spectral centroid",
        "ylabel": "Frequency [Hz]",
        "clean_reference": (
            clean_features["spectral_centroid_hz"]
        ),
        "source_reference": (
            source_features["spectral_centroid_hz"]
        ),
    },
    {
        "key": "spectral_bandwidth_hz",
        "title": "Spectral bandwidth",
        "ylabel": "Bandwidth [Hz]",
        "clean_reference": (
            clean_features["spectral_bandwidth_hz"]
        ),
        "source_reference": (
            source_features["spectral_bandwidth_hz"]
        ),
    },
    {
        "key": "spectral_rolloff_hz",
        "title": "Spectral roll-off",
        "ylabel": "Frequency [Hz]",
        "clean_reference": (
            clean_features["spectral_rolloff_hz"]
        ),
        "source_reference": (
            source_features["spectral_rolloff_hz"]
        ),
    },
    {
        "key": "spectral_flatness",
        "title": "Spectral flatness",
        "ylabel": "Flatness ratio",
        "clean_reference": (
            clean_features["spectral_flatness"]
        ),
        "source_reference": (
            source_features["spectral_flatness"]
        ),
    },
]

number_of_rows = len(experiments)
number_of_columns = len(plot_definitions)

figure_width = 4.2 * number_of_columns
figure_height = 3.3 * number_of_rows

figure, axes = plt.subplots(
    number_of_rows,
    number_of_columns,
    figsize=(figure_width, figure_height),
    squeeze=False,
    sharex="col",
)

for row_index, (
    network_section,
    depth_name,
    module_path,
) in enumerate(experiments):
    experiment_name = f"{network_section}_{depth_name}"
    results = experiment_results[experiment_name]

    factors = np.array([
        result["scaling_factor"]
        for result in results
    ])

    row_label = (
        f"{network_section.capitalize()} – "
        f"{depth_name.capitalize()}\n"
        f"{shorten_module_name(module_path)}"
    )

    for column_index, definition in enumerate(
        plot_definitions
    ):
        axis = axes[row_index, column_index]

        values = np.array([
            result[definition["key"]]
            for result in results
        ])

        axis.plot(
            factors,
            values,
            marker="o",
            linewidth=1.8,
        )

        # Marks the unchanged weight factor.
        axis.axvline(
            1.0,
            linestyle=":",
            linewidth=1.2,
            label="Factor 1.0",
        )

        clean_reference = definition["clean_reference"]
        source_reference = definition["source_reference"]

        if clean_reference is not None:
            axis.axhline(
                clean_reference,
                linestyle="--",
                linewidth=1.2,
                label="Clean reconstruction",
            )

        if source_reference is not None:
            axis.axhline(
                source_reference,
                linestyle="-.",
                linewidth=1.2,
                label="Source audio",
            )

        axis.grid(
            visible=True,
            alpha=0.25,
        )

        axis.set_xlabel("Weight scaling factor")
        axis.set_ylabel(definition["ylabel"])

        if row_index == 0:
            axis.set_title(
                definition["title"],
                fontsize=11,
                fontweight="bold",
            )

        if column_index == 0:
            axis.text(
                -0.32,
                0.5,
                row_label,
                transform=axis.transAxes,
                rotation=90,
                verticalalignment="center",
                horizontalalignment="center",
                fontsize=9,
            )

# Use one shared legend instead of repeating it in every subplot.
legend_handles = []
legend_labels = []

for axis in axes.flat:
    handles, labels = axis.get_legend_handles_labels()

    for handle, label in zip(handles, labels):
        if label not in legend_labels:
            legend_handles.append(handle)
            legend_labels.append(label)

figure.suptitle(
    (
        f"RAVE encoder and decoder weight perturbation analysis\n"
        f"Source: {file_name}"
    ),
    fontsize=16,
    fontweight="bold",
)

figure.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.965),
    ncol=max(1, len(legend_labels)),
)

figure.subplots_adjust(
    left=0.09,
    right=0.99,
    bottom=0.06,
    top=0.90,
    hspace=0.42,
    wspace=0.32,
)

combined_plot_path = (
    analysis_directory
    / "combined_analysis.png"
)

figure.savefig(
    combined_plot_path,
    dpi=220,
    bbox_inches="tight",
)

print(f"\nSaved combined plot: {combined_plot_path}")


# ============================================================
# 13. PRINT SUMMARY
# ============================================================

print("\nSource versus clean reconstruction:")

summary_features = [
    "rms_dbfs",
    "peak_dbfs",
    "crest_factor_db",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "spectral_rolloff_hz",
    "spectral_flatness",
]

for feature_name in summary_features:
    source_value = source_features[feature_name]
    clean_value = clean_features[feature_name]

    print(
        f"  {feature_name:>24}: "
        f"source={source_value:>12.4f}, "
        f"clean={clean_value:>12.4f}, "
        f"difference={clean_value - source_value:>+12.4f}"
    )

print("\nRun completed.")
print(f"Encoder audio: {encoder_audio_directory}")
print(f"Decoder audio: {decoder_audio_directory}")
print(f"Analysis data: {analysis_directory}")

plt.show()
