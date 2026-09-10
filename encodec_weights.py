from pathlib import Path
 
import numpy as np
import torch
import torchaudio
from torch.nn.utils import remove_weight_norm
 
from lib import *
from plotting import plot_comparison
from encodec import encodec_from_pretrained
 
# same factors as RAVE weight_analysis_v1.py skript
scaling_factors = [0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00, 8.00, 10.00]
depth_positions = {"early": 0.10, "middle": 0.50, "late": 0.90}
 
WEIGHT_SUFFIXES = ("weight", "weight_g", "weight_v", "weight_orig")
 
 
def encodec_process_audio(model, waveform: torch.Tensor, bandwidth: float | None = None) -> torch.Tensor:
    """Adapter fuer EncodecModel"""
    if bandwidth is None:
        bandwidth = max(model.config.target_bandwidths)
    input_tensor = waveform.unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor, bandwidth=bandwidth)
    return output.audio_values.squeeze(0)
 
 
def get_submodule(root: torch.nn.Module, module_path: str) -> torch.nn.Module:
    """Navigiert per Punkt-Pfad zu einem verschachtelten Modul (z.B. 'encoder.layers.4.conv')."""
    module = root
    for part in module_path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module
 
 
def collect_weight_module_paths(model: torch.nn.Module, section_name: str) -> list[str]:
    """Findet alle Module mit Gewichten in encoder/decoder, unabhaengig davon
    ob normal (.weight) oder weight-normalisiert (.weight_g/.weight_v)."""
    paths = set()
    for key in model.state_dict().keys():
        if section_name.lower() not in key.lower():
            continue
        for suffix in WEIGHT_SUFFIXES:
            if key.endswith("." + suffix):
                paths.add(key[: -(len(suffix) + 1)])
                break
    return sorted(paths)


 
def select_representative_paths(paths: list[str], positions: dict[str, float]) -> list[tuple[str, str]]:
    """Waehlt early/middle/late Module aus der sortierten Pfad-Liste."""
    selected = []
    for depth_name, rel_pos in positions.items():
        idx = max(0, min(round((len(paths) - 1) * rel_pos), len(paths) - 1))
        selected.append((depth_name, paths[idx]))
    return selected
 
 
def scale_module_weight(model: torch.nn.Module, module_path: str, factor: float) -> None:
    """Skaliert das Gewicht eines Moduls, loest dabei Weight Normalization auf."""
    module = get_submodule(model, module_path)
    try:
        remove_weight_norm(module)
    except (ValueError, AttributeError):
        pass
    with torch.no_grad():
        module.weight.mul_(factor)
 
 
def norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.max(torch.abs(x))
 
 
# Modell und Audio Laden
model_name = "facebook/encodec_24khz"
 
source_path: Path = check_path("audio/source")
reconstructed_root: Path = check_path("audio/reconstructed")
 
base_source, sr = torchaudio.load("audio/source/GLM.wav")
if base_source.shape[0] > 1:
    base_source = base_source[0:1, :]
 
model_clean = encodec_from_pretrained(model_name)
base_recon = encodec_process_audio(model_clean, base_source)
torchaudio.save(str(reconstructed_root / "base_reconstruction_encodec.wav"), base_recon, sr)
 
# Layers auswählen
experiments = []
for section in ("encoder", "decoder"):
    paths = collect_weight_module_paths(model_clean, section)
    for depth_name, module_path in select_representative_paths(paths, depth_positions):
        experiments.append((section, depth_name, module_path))
 

 
#  Skalierungsfaktoren
    for factor in scaling_factors:
        model = encodec_from_pretrained(model_name)
        scale_module_weight(model, module_path, factor)
 
        recon = norm(encodec_process_audio(model, base_source))
 
        tag = f"{section}_{depth_name}_factor_{str(factor).replace('.', '_')}"
        fn_a = str(reconstructed_root / f"{tag}.wav")
        fn_p = str(reconstructed_root / f"{tag}.png")
 
        torchaudio.save(fn_a, recon, sr)
        plot_comparison(
            base_recon, recon, sr,
            title=f"{section} {depth_name} x{factor}",
            save_path=fn_p, show=False,
        )

