import math
import argparse
from pathlib import Path

import torch
import torchaudio
import torch.nn as nn

from encodec import EncodecModel
from encodec.utils import convert_audio


def process_audio_resnet_spotlight(
    input_path: str,
    output_path: str,
    lfo_rate_hz: float | None = None,
    glitch_intensity: float = 0.75,
    model: EncodecModel | None = None,
):
    """Decode audio while an LFO moves a soft sign-inversion mask across ResNet blocks.

    If ``lfo_rate_hz`` is omitted, one complete LFO cycle is spread across the
    clip duration. An explicit value is interpreted as cycles per second.
    ``glitch_intensity`` controls the maximum blend with the sign-inverted
    feature map and must be between 0 and 1.
    """
    if lfo_rate_hz is not None and (
        not math.isfinite(lfo_rate_hz) or lfo_rate_hz < 0
    ):
        raise ValueError("lfo_rate_hz must be a finite, nonnegative number.")
    if not math.isfinite(glitch_intensity) or not 0.0 <= glitch_intensity <= 1.0:
        raise ValueError("glitch_intensity must be between 0 and 1.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Reuse a supplied model when processing a folder; otherwise load one here.
    if model is None:
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(6.0)
    model.to(device)
    model.eval()

    # 2. Audio laden und anpassen
    wav, sr = torchaudio.load(input_path)
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0).to(device)

    # 3. Encodieren
    with torch.no_grad():
        encoded_frames = model.encode(wav)

    if not encoded_frames:
        raise RuntimeError("EnCodec returned no encoded frames for the input audio.")

    total_code_frames = sum(codes.shape[-1] for codes, _ in encoded_frames)
    duration_seconds = wav.shape[-1] / model.sample_rate
    if duration_seconds <= 0:
        raise ValueError("Input audio must have a positive duration.")
    if lfo_rate_hz is None:
        lfo_rate_hz = 1.0 / duration_seconds
        print(
            f"Using one LFO cycle across the clip "
            f"({duration_seconds:.2f} s, {lfo_rate_hz:.4f} Hz)."
        )

    latent_rate_hz = total_code_frames / duration_seconds
    segment_stride = model.segment_stride
    frame_step_seconds = (
        segment_stride / model.sample_rate
        if segment_stride is not None
        else duration_seconds
    )

    # 4. WRAPPER: AUTOMATISCHES AUSLESEN & NUR RESNET-BLÖCKE GLITCHEN
    class ResNetSpotlightDecoder(nn.Module):
        def __init__(
            self,
            original_decoder,
            lfo_rate_hz,
            intensity,
            latent_rate_hz,
            frame_step_seconds,
        ):
            super().__init__()
            # Keep the original decoder modules in their existing execution order.
            if hasattr(original_decoder, 'model') and isinstance(original_decoder.model, nn.Sequential):
                self.decoder_layers = nn.ModuleList(list(original_decoder.model.children()))
            else:
                self.decoder_layers = nn.ModuleList(list(original_decoder.children()))

            self.lfo_rate_hz = lfo_rate_hz
            self.intensity = intensity
            self.latent_rate_hz = latent_rate_hz
            self.frame_step_seconds = frame_step_seconds
            self.frame_offset_seconds = 0.0

            self.resnet_indices = []
            for idx, layer in enumerate(self.decoder_layers):
                class_name = layer.__class__.__name__
                if "Resnet" in class_name or "ResNet" in class_name:
                    self.resnet_indices.append(idx)

            print(f"-> Gefundene ResNet-Blöcke im Decoder an den Indizes: {self.resnet_indices}")

        def reset_lfo(self):
            """Start the LFO timeline at the beginning of a decoded clip."""
            self.frame_offset_seconds = 0.0

        def forward(self, x):
            total_res_blocks = len(self.resnet_indices)
            input_sequence_length = x.shape[-1]
            frame_duration_seconds = input_sequence_length / self.latent_rate_hz
            frame_start_seconds = self.frame_offset_seconds

            if total_res_blocks == 0:
                for layer in self.decoder_layers:
                    x = layer(x)
                self.frame_offset_seconds += self.frame_step_seconds
                return x

            for i, layer in enumerate(self.decoder_layers):
                x = layer(x)

                if i in self.resnet_indices:
                    res_idx = self.resnet_indices.index(i)

                    current_sequence_length = x.shape[-1]
                    time_in_frame = (
                        torch.arange(current_sequence_length, device=x.device, dtype=x.dtype)
                        * (frame_duration_seconds / current_sequence_length)
                    )
                    time_seconds = frame_start_seconds + time_in_frame
                    phase = 2 * math.pi * self.lfo_rate_hz * time_seconds
                    lfo = (torch.sin(phase) + 1.0) / 2.0
                    res_pos = lfo * float(max(0, total_res_blocks - 1))
                    distance = torch.abs(res_pos - float(res_idx))
                    spotlight_weight = torch.clamp(1.0 - distance, min=0.0)

                    if spotlight_weight.max() > 0.01:
                        mask = spotlight_weight.view(1, 1, current_sequence_length)
                        mask = mask * self.intensity
                        x = x * (1.0 - 2.0 * mask)

            self.frame_offset_seconds += self.frame_step_seconds
            return x

    # Replace only the decoder execution path; all encoded frames still pass through it.
    spotlight_decoder = ResNetSpotlightDecoder(
        model.decoder,
        lfo_rate_hz,
        glitch_intensity,
        latent_rate_hz,
        frame_step_seconds,
    )
    original_decoder = model.decoder
    model.decoder = spotlight_decoder

    print(f"Starte ResNet-Spotlight Decoder (LFO: {lfo_rate_hz} Hz)...")
    spotlight_decoder.reset_lfo()
    try:
        with torch.no_grad():
            decoded_wav = model.decode(encoded_frames)
    finally:
        model.decoder = original_decoder

    output_audio = decoded_wav.squeeze(0).cpu()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_file), output_audio, model.sample_rate)
    print(f"-> ResNet-Spotlight Audio erfolgreich gespeichert: {output_path}")


def process_wav_folder(
    input_dir: str | Path = "audio/source",
    output_dir: str | Path = "audio/lfo_sweeps",
    lfo_rate_hz: float | None = None,
    glitch_intensity: float = 0.75,
) -> list[Path]:
    """Process every WAV directly inside a folder using one shared EnCodec model."""
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    wav_files = sorted(
        (path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() == ".wav"),
        key=lambda path: path.name.lower(),
    )
    if not wav_files:
        raise FileNotFoundError(f"No WAV files found directly inside '{input_path}'.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(6.0)
    model.to(device)
    model.eval()

    outputs = []
    for wav_path in wav_files:
        output_file = output_path / f"{wav_path.stem}_lfo.wav"
        process_audio_resnet_spotlight(
            input_path=str(wav_path),
            output_path=str(output_file),
            lfo_rate_hz=lfo_rate_hz,
            glitch_intensity=glitch_intensity,
            model=model,
        )
        outputs.append(output_file)

    print(f"Processed {len(wav_files)} WAV files into '{output_path}'.")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply an LFO-driven ResNet spotlight to WAV files in a folder."
    )
    parser.add_argument("--input-dir", default="audio/source", help="Folder to scan for WAV files.")
    parser.add_argument("--output-dir", default="audio/lfo_sweeps", help="Folder for rendered WAV files.")
    parser.add_argument(
        "--lfo-rate-hz",
        type=float,
        default=None,
        help="LFO frequency in Hz. Omit to run one complete cycle over each WAV.",
    )
    parser.add_argument("--glitch-intensity", type=float, default=0.75)
    args = parser.parse_args()

    process_wav_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        lfo_rate_hz=args.lfo_rate_hz,
        glitch_intensity=args.glitch_intensity,
    )