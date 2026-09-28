import copy
from pathlib import Path

import torchaudio

from cli.state import create_model, list_audio_files
from model import LayerInfo, NetTypeEnum
from modules import AdditionLayer, MultiplierLayer, RepeatingLayer, SkippingLayer


class QuitRequested(Exception):
    pass


def _ask(prompt: str) -> str:
    try:
        value = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        raise QuitRequested from None
    if value.lower() in {"q", "quit", "exit"}:
        raise QuitRequested
    return value


def _select_layer(model, operation: str) -> LayerInfo:
    network_name = _ask("Network [encoder/decoder]: ").lower()
    try:
        net_type = NetTypeEnum(network_name)
    except ValueError as exc:
        raise ValueError("Choose encoder or decoder.") from exc

    net = model.get_net(net_type)
    candidates = []
    for index in range(len(net)):
        layer = net[index]
        channels = model.get_layer_channels(layer)
        if operation == "restore":
            eligible = True
        elif operation in {"skip", "repeat"}:
            eligible = channels is None or channels[0] == channels[1]
        else:
            eligible = model.layer_has_weights(layer)
        if eligible:
            candidates.append(LayerInfo.from_net_and_index(model, net, index))

    if not candidates:
        raise ValueError(f"No eligible layers found in {network_name} for {operation}.")

    print("Eligible layers:")
    for layer_info in candidates:
        print(
            f"  {layer_info.index:>2}  {layer_info.layer_path:<32} "
            f"{layer_info.name}  channels={layer_info.inout}"
        )

    try:
        index = int(_ask("Layer index: "))
    except ValueError as exc:
        raise ValueError("Layer index must be an integer.") from exc
    for layer_info in candidates:
        if layer_info.index == index:
            return layer_info
    raise ValueError(f"Layer index {index} is not eligible for {operation}.")


def _apply_intervention(model, operation: str) -> None:
    layer_info = _select_layer(model, operation)
    if operation == "skip":
        replacement = SkippingLayer(layer_info)
    elif operation == "repeat":
        try:
            repeats = int(_ask("Total layer applications (e.g. 2, 3, 5): "))
        except ValueError as exc:
            raise ValueError("Repeat count must be an integer.") from exc
        if repeats < 1:
            raise ValueError("Repeat count must be at least 1.")
        replacement = RepeatingLayer(layer_info, repeats=repeats)
    elif operation == "multiply":
        try:
            weight_factor = float(_ask("Weight multiplication factor: "))
        except ValueError as exc:
            raise ValueError("Weight factor must be numeric.") from exc
        replacement = MultiplierLayer(
            layer_info, weight_mul=weight_factor, bias_mul=1.0
        )
    else:
        try:
            offset = float(_ask("Additive offset (applied to weights and biases): "))
        except ValueError as exc:
            raise ValueError("Additive offset must be numeric.") from exc
        replacement = AdditionLayer(
            layer_info, weight_add=offset, bias_add=offset
        )

    model.get_net(layer_info.net_type)[layer_info.index] = replacement
    print(f"Applied {operation} to {layer_info.layer_path}.")


def _print_model(model) -> None:
    for net_type in NetTypeEnum:
        net = model.get_net(net_type)
        print(f"{net_type.value.upper()}")
        for index, layer in enumerate(net):
            print(f"  {index:>2}  {type(layer).__name__}")


def _restore_layer(model, backup_model) -> None:
    layer_info = _select_layer(model, "restore")
    model.get_net(layer_info.net_type)[layer_info.index] = copy.deepcopy(
        backup_model.get_net(layer_info.net_type)[layer_info.index]
    )
    print(f"Restored {layer_info.layer_path}.")


def generate_loop(args) -> None:
    input_files = list_audio_files(args.input)
    model = create_model(args.model, args.rave_path, args.sample_rate)
    backup_model = create_model(args.model, args.rave_path, args.sample_rate)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(input_files)} input file(s). Output directory: {output_path}")
    print("Commands: skip, repeat, multiply, add, list, restore, reset, listen, write, help, quit")

    while True:
        try:
            command = _ask("[nbform] > ").lower()
            if command in {"q", "quit", "exit"}:
                break
            if command == "help":
                print(
                    "skip/repeat alter layer execution; multiply/add alter layer "
                    "parameters. list shows modules, restore resets one layer, "
                    "reset restores the full pretrained model, listen plays audio, "
                    "and write exports the modified outputs. Enter q at a prompt to quit."
                )
            elif command == "list":
                _print_model(model)
            elif command in {"skip", "repeat", "multiply", "add"}:
                _apply_intervention(model, command)
            elif command == "restore":
                _restore_layer(model, backup_model)
            elif command == "reset":
                model.reset()
                print("Restored the complete pretrained model.")
            elif command == "listen":
                try:
                    import sounddevice as sd
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError(
                        "Playback needs sounddevice; install nbform with the default dependencies."
                    ) from exc
                for file_path in input_files:
                    waveform, sample_rate = torchaudio.load(str(file_path))
                    audio = model((waveform, sample_rate)).detach().cpu()
                    sd.play(audio.numpy().T, samplerate=model.get_sample_rate(), blocking=True)
            elif command == "write":
                for file_path in input_files:
                    waveform, sample_rate = torchaudio.load(str(file_path))
                    audio = model((waveform, sample_rate)).detach().cpu()
                    destination = output_path / file_path.name
                    torchaudio.save(str(destination), audio, model.get_sample_rate())
                    print(f"Wrote {destination}")
            else:
                print("Unknown command. Type help to list available commands.")
        except QuitRequested:
            break
        except (ValueError, FileNotFoundError, IndexError) as exc:
            print(f"Command not applied: {exc}")

    print("Leaving interactive generation.")