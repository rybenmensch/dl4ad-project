import copy
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import prompt_toolkit as pt
import sounddevice as sd
import torch
import torchaudio
from prompt_toolkit.shortcuts.progress_bar import formatters as pb_formatters

from cli.lib import (
    ChoiceOption,
    PromptEnum,
    UserCancelledError,
    auto_complete,
    bindings_with_exit,
    choose,
    format_auto_complete,
    get_input,
    get_param,
    sel,
    usage,
)
from cli.state import AppState, AudioTensor
from library.encodec import EncodecNNModel
from library.layers import AdditionLayer, MultiplierLayer, RepeatingLayer, SkippingLayer
from library.model import (
    LayerInfo,
    Net,
    NetTypeEnum,
    NNModel,
    get_all_layers,
    get_all_layers_from_net,
    get_shape_preserving_layers,
    get_weighted_layers,
)
from library.rave import ExportOptions, RAVEModel


def print_help(_: AppState) -> None:
    ln = max(len(c.key) for c in commands) + 2
    padding = 2

    print("List of commands:")
    for c in commands:
        key_str = format_auto_complete(c.key, c.abbr)
        print(f"{key_str:<{ln + padding}} {c.desc}")


def print_layer_common(
    model: NNModel,
    layers: list[LayerInfo],
    title: str,
    should_print_title: bool,
) -> None:
    max_name_len = max(len(l.name) for l in layers)

    if should_print_title:
        print(title.upper())
        filler = " " * len(title)
    else:
        filler = ""

    for layer in layers:
        inout = layer.inout or ""
        idx = f"({layer.index})"
        fmt = f"{idx:<4} {layer.name:<{max_name_len}} {inout}"
        print(filler, fmt)


class PrintMode(PromptEnum):
    All = "all"
    Encoder = "encoder"
    Decoder = "decoder"


def print_model(app: AppState) -> None:
    mode = PrintMode.get_choice_menu()

    if mode == PrintMode.All:
        for net_type in NetTypeEnum:
            net = app.model.get_net(net_type)
            print_layer_common(
                app.model,
                get_all_layers_from_net(app.model, net),
                net_type.value,
                True,
            )
    else:
        net = app.model.get_net(NetTypeEnum(mode))
        print_layer_common(
            app.model, get_all_layers_from_net(app.model, net), NetTypeEnum(mode), True
        )


def get_diff_layers(model: NNModel, other_model: NNModel) -> list[LayerInfo]:
    results: list[LayerInfo] = []
    for this, other in zip(get_all_layers(model), get_all_layers(other_model)):
        if type(this.get_layer()) != type(other.get_layer()):
            results.append(this)
    return results


def print_diff(app: AppState) -> None:
    layers = get_diff_layers(app.model, app.backup_model)
    if len(layers) == 0:
        print("Model has not yet been modified.")
        return

    encoder_l = [l for l in layers if l.net_type == NetTypeEnum.Encoder]
    decoder_l = [l for l in layers if l.net_type == NetTypeEnum.Decoder]

    # version without selection
    # if len(encoder_l):
    #     print_layer_common(app.model, encoder_l, NetTypeEnum.Encoder.value, True)
    #
    # if len(decoder_l):
    #     print_layer_common(app.model, decoder_l, NetTypeEnum.Decoder.value, True)

    # version with selection
    if len(encoder_l):
        if len(decoder_l):
            net_type, layers = get_net_type_and_layers(layers)
            print_layer_common(app.model, layers, net_type.value, True)
        else:
            print_layer_common(app.model, encoder_l, NetTypeEnum.Encoder.value, True)
    else:
        print_layer_common(app.model, decoder_l, NetTypeEnum.Decoder.value, True)


def get_net_and_layer_info(
    model: NNModel, net_type: NetTypeEnum, index: int
) -> tuple[Net, LayerInfo]:
    net = model.get_net(net_type)
    layer_info = LayerInfo.from_layer(model, net[index])
    return (net, layer_info)


class NetTypePromptEnum(PromptEnum):
    Encoder = "encoder"
    Decoder = "decoder"

    def to_net_type(self) -> NetTypeEnum:
        return NetTypeEnum(self.value)

    @classmethod
    def human_name(cls) -> str:
        return "net type"


def get_net_type_and_layers(
    layers: list[LayerInfo],
) -> tuple[NetTypeEnum, list[LayerInfo]]:
    net_type = NetTypePromptEnum.get_choice_menu().to_net_type()
    layers = [l for l in layers if l.net_type == net_type]
    return (net_type, layers)


def get_net_type_and_index(layers: list[LayerInfo]) -> tuple[NetTypeEnum, int]:
    net_type, layers = get_net_type_and_layers(layers)
    if len(layers) == 0:
        raise IndexError(f" Net {net_type.value} has no layers of interest.")

    index = choose(
        f" Select layer ({usage(sel("index"))})",
        [
            ChoiceOption(
                value=layer.index,
                label=f"({layer.index}) {layer.name} {layer.inout or ''}",
                shortcut=None,
                number_key=layer.index,
            )
            for layer in layers
        ],
    )

    if index not in [layer.index for layer in layers]:
        raise IndexError(f"Index out of bounds: {index}")

    return (net_type, index)


def handle_skip_layer(app: AppState) -> None:
    try:
        net_type, index = get_net_type_and_index(get_shape_preserving_layers(app.model))
    except IndexError as e:
        print(e)
        return

    net, layer_info = get_net_and_layer_info(app.model, net_type, index)
    net[index] = SkippingLayer(layer_info)


def handle_repeat_layer(app: AppState) -> None:
    try:
        net_type, index = get_net_type_and_index(get_shape_preserving_layers(app.model))
    except IndexError as e:
        print(e)
        return

    num_repeats = get_param(int, "number of repetitions")

    net, layer_info = get_net_and_layer_info(app.model, net_type, index)
    net[index] = RepeatingLayer(layer_info, repeats=cast(int, num_repeats))


def handle_multiplier_layer(app: AppState) -> None:
    try:
        net_type, index = get_net_type_and_index(get_weighted_layers(app.model))
    except IndexError as e:
        print(e)
        return

    mult = get_param(float, "multiplication factor")

    net, layer_info = get_net_and_layer_info(app.model, net_type, index)
    net[index] = MultiplierLayer(layer_info, weight_mul=cast(float, mult))


def handle_addition_layer(app: AppState) -> None:
    try:
        net_type, index = get_net_type_and_index(get_weighted_layers(app.model))
    except IndexError as e:
        print(e)
        return

    add = get_param(float, "addition factor")

    net, layer_info = get_net_and_layer_info(app.model, net_type, index)
    net[index] = AdditionLayer(layer_info, weight_add=cast(float, add))


class ListeningMode(PromptEnum):
    Original = "original"
    Modified = "modified"
    Baseline = "baseline"

    @classmethod
    def get_audio_tensor(
        cls, app: AppState, wav: tuple[torch.Tensor, int]
    ) -> AudioTensor:
        if cls == ListeningMode.Modified:
            audio = app.model(wav)
            sr = app.model.get_sample_rate()
        elif cls == ListeningMode.Baseline:
            audio = app.backup_model(wav)
            sr = app.backup_model.get_sample_rate()
        else:
            audio, sr = wav
        return AudioTensor(audio=audio, sr=sr)


def listen(app: AppState) -> None:
    input_index = 0
    if len(app.files) > 1:
        input_index = choose(
            f"Select input file for playback {usage()}",
            ChoiceOption.from_labels([f.path.name for f in app.files]),
        )

    try:
        file = app.files[input_index]
    except IndexError:
        print(f"Index out of bounds: {input_index}")
        return

    mode = ListeningMode.get_choice_menu()
    at = mode.get_audio_tensor(app, file.wav)

    def make_progress_bar(length: str) -> pt.shortcuts.ProgressBar:
        custom_formatter = [
            pb_formatters.Label(),
            pb_formatters.Text(": "),
            pb_formatters.Bar(sym_a="#", sym_b="#", sym_c="."),
            pb_formatters.Text(" "),
            pb_formatters.TimeElapsed(),
            pb_formatters.Text(" / ", style="class:time-left"),
            pb_formatters.Text(length, style="class:time-left"),
        ]

        return pt.shortcuts.ProgressBar(
            title="Now playing",
            key_bindings=bindings_with_exit(),
            formatters=custom_formatter,
        )

    try:
        with make_progress_bar(at.readable_length()) as pb:
            with at.output_stream() as stream:
                for chunk in pb(at.chunks()):
                    stream.write(chunk)
    except KeyboardInterrupt:
        sd.stop()
    except EOFError:
        sd.stop()
        sys.exit(0)


class RestoreMode(PromptEnum):
    All = "all"
    Index = "index"


def restore_model(app: AppState) -> None:
    mode = RestoreMode.get_choice_menu()

    if mode == RestoreMode.All:
        app.model.reset()
        return

    layers = get_diff_layers(app.model, app.backup_model)
    if len(layers) == 0:
        print("Model has not yet been modified.")
        return

    try:
        net_type, index = get_net_type_and_index(layers)
    except IndexError as e:
        print(e)
        return

    net = app.model.get_net(net_type)
    backup_net = app.backup_model.get_net(net_type)

    try:
        net[index] = copy.deepcopy(backup_net[index])
    except IndexError:
        print(f"Index out of bounds: {index}")
        return


class WriteFileMode(PromptEnum):
    Keep = "keep"
    Prepend = "prepend"
    Append = "append"
    Individual = "individual"


def write_file(app: AppState) -> None:
    mode = WriteFileMode.get_choice_menu()

    prep_or_app = ""
    if mode == WriteFileMode.Prepend or mode == WriteFileMode.Append:
        prep_or_app = get_param(str, f"file name {mode.value}")

    parent = Path(app.output)
    for f in app.files:
        path = f.path
        stem, suffix = path.stem, path.suffix

        if mode == WriteFileMode.Keep:
            name = stem + suffix
        elif mode == WriteFileMode.Prepend:
            name = prep_or_app + "_" + stem + suffix
        elif mode == WriteFileMode.Append:
            name = stem + "_" + prep_or_app + suffix
        else:
            name = cast(
                Path, get_param(Path, f"new file name for modified file {path.name}")
            )
            name = name.stem
            name = name + suffix

        path = parent / name
        processed = app.model(f.wav)
        torchaudio.save(path, processed, app.model.get_sample_rate())
        print(f"Wrote file {path}")


class OutputFolderType(PromptEnum):
    Same = "same"
    New = "new"


def export_model(app: AppState) -> None:
    if isinstance(app.model, RAVEModel):
        choice = OutputFolderType.get_choice_menu()
        output_path = None
        if choice == OutputFolderType.Same:
            output_path = app.rave_path
        elif choice == OutputFolderType.New:
            output_path = get_input(
                "Enter output folder", quit_on_q=False, escape_cancels=True
            )
        assert output_path is not None
        output_path = Path(output_path).absolute()

        if not output_path.exists():
            raise FileNotFoundError(f"Output folder does not exist: {output_path}")

        name = get_input("Enter name", quit_on_q=False, escape_cancels=True)
        name = Path(name).stem
        name = name + ".ts"

        output_path = output_path / name

        if output_path.exists():
            raise FileExistsError(f"Output file already exists: {output_path}")

        fidelity = get_param(float, "fidelity")
        app.model.export(ExportOptions(path=output_path, fidelity=fidelity))

    elif isinstance(app.model, EncodecNNModel):
        # just let that error print as it's not implemented anyway
        try:
            app.model.export(None)
        except NotImplementedError as e:
            print(e)


@dataclass(frozen=True)
class Command:
    key: str
    fn: Callable[[AppState], None]
    desc: str
    abbr: str = ""


commands = [
    Command(key="skip", fn=handle_skip_layer, desc="Skip"),
    Command(key="repeat", fn=handle_repeat_layer, desc="Repeat"),
    Command(key="multiply", fn=handle_multiplier_layer, desc="Multiply"),
    Command(key="add", fn=handle_addition_layer, desc="Add"),
    Command(key="quit", fn=lambda _: sys.exit(0), desc="Quit"),
    Command(key="help", fn=print_help, desc="Help"),
    Command(key="print", fn=print_model, desc="Print model"),
    Command(key="diff", fn=print_diff, desc="Print difference to baseline model"),
    Command(key="listen", fn=listen, desc="Listen to the current state"),
    Command(key="restore", abbr="x", fn=restore_model, desc="Restore model"),
    Command(key="write", fn=write_file, desc="Write file"),
    Command(key="export", fn=export_model, desc="Export model to torchscript"),
]


def generate_loop(app: AppState) -> None:
    pt.shortcuts.clear()
    # TODO: splash screen?

    while True:
        try:
            user_input = get_input(
                "Enter command ([h]elp / [q]uit)", escape_cancels=False, quit_on_q=True
            )
        except UserCancelledError:
            sys.exit()

        try:

            for c in commands:
                if auto_complete(user_input, c.key, c.abbr):
                    c.fn(app)
                    break  # skip other commands that match
        except UserCancelledError:
            continue
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)
