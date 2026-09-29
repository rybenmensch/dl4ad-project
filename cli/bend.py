import copy
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import prompt_toolkit as pt
import sounddevice as sd
import torchaudio
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts.progress_bar import formatters as pb_formatters

from cli.lib import (
    ChoiceOption,
    CommandHistory,
    PromptEnum,
    UserCancelledError,
    UserQuitError,
    auto_complete,
    choose,
    format_auto_complete,
    get_input,
    get_param,
    print_filtered,
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
    Swap,
    get_all_layers,
    get_shape_preserving_layers,
    get_swappable_layers,
    get_weighted_layers,
    swap_layers,
)
from library.rave import ExportOptions, RAVEModel

command_history = CommandHistory()
main_session = PromptSession(
    history=command_history,
    enable_suspend=True,
    interrupt_exception=KeyboardInterrupt,
    eof_exception=EOFError,
)


def print_help(_: AppState) -> None:
    ln = max(len(c.key) for c in commands) + 2
    padding = 2

    print("List of commands:")
    for c in commands:
        key_str = format_auto_complete(c.key, c.abbr)
        print(f"{key_str:<{ln + padding}} {c.desc}")


class PrintMode(PromptEnum):
    All = "all"
    Encoder = "encoder"
    Decoder = "decoder"


def print_model(app: AppState) -> None:
    layers = get_all_layers(app.model)

    print_mode = PrintMode.get_choice_menu()
    if print_mode == PrintMode.All:
        for net_type in NetTypeEnum:
            print_filtered(app.model, layers, net_type)

        add_to_command_history("print", print_mode.value)

    elif print_mode == PrintMode.Encoder or print_mode == PrintMode.Decoder:
        net_type = NetTypeEnum(print_mode)
        print_filtered(app.model, layers, net_type)

        add_to_command_history("print", print_mode.value, net_type.value)

    else:
        raise NotImplementedError


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

    print_mode = PrintMode.get_choice_menu()
    if print_mode == PrintMode.All:
        for net_type in NetTypeEnum:
            print_filtered(app.model, layers, net_type)

        add_to_command_history("print", print_mode.value)

    elif print_mode == PrintMode.Encoder or print_mode == PrintMode.Decoder:
        net_type = NetTypeEnum(print_mode)
        print_filtered(app.model, layers, net_type)

        add_to_command_history("print", print_mode.value, net_type.value)

    else:
        raise NotImplementedError


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
        raise IndexError(f"Net {net_type.value} has no layers of interest.")

    index = choose(
        f" Select layer {usage(sel("index"))}",
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


def make_command(command_name: str, *args) -> str:
    args = [command_name] + [a for a in args]
    return " ".join(args)


def add_to_command_history(command_name: str, *args) -> None:
    args = [command_name] + [str(a) for a in args if a != ""]
    argstr = " ".join(args)
    command_history.replace_last(argstr)


def handle_skip_layer(app: AppState) -> None:
    try:
        net_type, index = get_net_type_and_index(get_shape_preserving_layers(app.model))
    except IndexError as e:
        print(e)
        return

    net, layer_info = get_net_and_layer_info(app.model, net_type, index)
    net[index] = SkippingLayer(layer_info)

    add_to_command_history("skip", net_type.value, index)


def handle_repeat_layer(app: AppState) -> None:
    try:
        net_type, index = get_net_type_and_index(get_shape_preserving_layers(app.model))
    except IndexError as e:
        print(e)
        return

    num_repeats = get_param(int, "number of repetitions")

    net, layer_info = get_net_and_layer_info(app.model, net_type, index)
    net[index] = RepeatingLayer(layer_info, repeats=num_repeats)

    add_to_command_history("repeat", net_type.value, index, num_repeats)


def handle_swap_layer(app: AppState) -> None:
    swappable_layers = get_swappable_layers(app.model)
    swappable_info = [l.source for l in swappable_layers]
    try:
        net_type, index = get_net_type_and_index(swappable_info)
    except IndexError as e:
        print(e)
        return

    swap_info = swappable_layers[index]
    swap_encoder = [l for l in swap_info.targets if l.net_type == NetTypeEnum.Encoder]
    swap_decoder = [l for l in swap_info.targets if l.net_type == NetTypeEnum.Decoder]

    if len(swap_encoder) and len(swap_decoder):
        net_type, index = get_net_type_and_index(swappable_info)
    else:
        index = choose(
            f" Select layer {usage(sel("index"))}",
            ChoiceOption.from_labels_and_number_keys(
                [
                    (
                        f"({layer.net_type.value}) ({layer.index}) {layer.name}"
                        f" {layer.inout or ''}",
                        layer.index,
                    )
                    for layer in swap_info.targets
                ],
            ),
        )

    swap = Swap.from_info(swap_info, index)
    swap_layers(app.model, swappable_layers, swap)


def handle_multiplier_layer(app: AppState) -> None:
    try:
        net_type, index = get_net_type_and_index(get_weighted_layers(app.model))
    except IndexError as e:
        print(e)
        return

    mult = get_param(float, "multiplication factor")

    net, layer_info = get_net_and_layer_info(app.model, net_type, index)
    net[index] = MultiplierLayer(layer_info, weight_mul=mult)

    add_to_command_history("multiply", net_type.value, index, mult)


def handle_addition_layer(app: AppState) -> None:
    try:
        net_type, index = get_net_type_and_index(get_weighted_layers(app.model))
    except IndexError as e:
        print(e)
        return

    add = get_param(float, "addition factor")

    net, layer_info = get_net_and_layer_info(app.model, net_type, index)
    net[index] = AdditionLayer(layer_info, weight_add=add)

    add_to_command_history("add", net_type.value, index, add)


class ListeningMode(PromptEnum):
    Original = "original"
    Modified = "modified"
    Baseline = "baseline"


def handle_listen(app: AppState) -> None:
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

    listening_mode = ListeningMode.get_choice_menu()
    if listening_mode == ListeningMode.Modified:
        audio = app.model(file.wav)
        sr = app.model.get_sample_rate()
    elif listening_mode == ListeningMode.Baseline:
        audio = app.backup_model(file.wav)
        sr = app.backup_model.get_sample_rate()
    elif listening_mode == ListeningMode.Original:
        audio, sr = file.wav
    else:
        raise NotImplementedError

    at = AudioTensor.from_tuple((audio, sr))

    stop_playback = threading.Event()
    quit_requested = threading.Event()

    def stop(event) -> None:
        stop_playback.set()
        event.app.exit()

    def quit(event) -> None:
        quit_requested.set()
        stop(event)

    playback_bindings = KeyBindings()
    playback_bindings.add("c-c", eager=True)(stop)
    playback_bindings.add("escape", eager=True)(stop)
    playback_bindings.add("c-d", eager=True)(quit)

    pb = pt.shortcuts.ProgressBar(
        title="Now playing",
        key_bindings=playback_bindings,
        formatters=[
            pb_formatters.Bar(sym_a="#", sym_b="#", sym_c="."),
            pb_formatters.Text(" "),
            pb_formatters.TimeElapsed(),
            pb_formatters.Text(" / ", style="class:time-left"),
            pb_formatters.Text(at.readable_length(), style="class:time-left"),
        ],
    )

    try:
        with pb as pb:
            with at.output_stream() as stream:
                for chunk in pb(at.chunks()):
                    if stop_playback.is_set():
                        break
                    stream.write(chunk)
    finally:
        sd.stop()

    add_to_command_history("listen", input_index, listening_mode.value)

    if quit_requested.is_set():
        sys.exit(0)


class RestoreMode(PromptEnum):
    All = "all"
    Index = "index"


def handle_restore(app: AppState) -> None:
    restore_mode = RestoreMode.get_choice_menu()

    if restore_mode == RestoreMode.All:
        app.model.reset()

        add_to_command_history("restore", restore_mode.value)
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

    add_to_command_history("restore", restore_mode.value, net_type.value, index)


class WriteFileMode(PromptEnum):
    Keep = "keep"
    Prepend = "prepend"
    Append = "append"
    Individual = "individual"


def handle_write(app: AppState) -> None:
    write_mode = WriteFileMode.get_choice_menu()

    prep_or_app = ""
    if write_mode == WriteFileMode.Prepend or write_mode == WriteFileMode.Append:
        prep_or_app = get_param(str, f"file name {write_mode.value}")

    parent = Path(app.output)
    new_names: list[str] = []
    for f in app.files:
        path = f.path
        stem, suffix = path.stem, path.suffix

        if write_mode == WriteFileMode.Keep:
            name = stem + suffix
        elif write_mode == WriteFileMode.Prepend:
            name = prep_or_app + "_" + stem + suffix
        elif write_mode == WriteFileMode.Append:
            name = stem + "_" + prep_or_app + suffix
        elif write_mode == WriteFileMode.Individual:
            name = cast(
                Path, get_param(Path, f"new file name for modified file {path.name}")
            )
            new_names.append(str(name))
            name = name.stem
            name = name + suffix
        else:
            raise NotImplementedError

        path = parent / name
        processed = app.model(f.wav)
        torchaudio.save(path, processed, app.model.get_sample_rate())
        print(f"Wrote file {path}")

    add_to_command_history("write", write_mode, prep_or_app, *new_names)


class OutputFolderType(PromptEnum):
    Same = "same"
    New = "new"


def handle_export(app: AppState) -> None:
    if isinstance(app.model, RAVEModel):
        output_folder_type = OutputFolderType.get_choice_menu()
        if output_folder_type == OutputFolderType.Same:
            output_folder = app.rave_path
        elif output_folder_type == OutputFolderType.New:
            output_folder = get_input("Enter output folder", escape_cancels=True)
        else:
            raise NotImplementedError
        assert output_folder is not None
        output_folder = Path(output_folder).absolute()

        if not output_folder.exists():
            raise FileNotFoundError(f"Output folder does not exist: {output_folder}")

        name = get_input("Enter name", escape_cancels=True)
        name = Path(name).stem
        name_ext = name + ".ts"
        output_path = output_folder / name_ext

        if output_path.exists():
            raise FileExistsError(f"Output file already exists: {output_path}")

        fidelity = get_param(float, "fidelity")
        app.model.export(ExportOptions(path=output_path, fidelity=fidelity))

        add_to_command_history(
            "export", output_folder_type.value, str(output_folder), name, fidelity
        )

    elif isinstance(app.model, EncodecNNModel):
        # just let that error print as it's not implemented anyway
        try:
            app.model.export(None)
        except NotImplementedError as e:
            print(e)


class QuitMode(PromptEnum):
    Yes = "yes"
    No = "no"


def handle_quit(app: AppState) -> None:
    quit_mode = QuitMode.get_choice_menu(prompt="Are you sure?")
    if quit_mode == QuitMode.Yes:
        raise UserQuitError
    elif quit_mode == QuitMode.No:
        return
    else:
        raise NotImplementedError  # lol


@dataclass(frozen=True)
class Command:
    key: str
    fn: Callable[[AppState], None]
    desc: str
    abbr: str | None = ""


commands = [
    Command(key="skip", fn=handle_skip_layer, desc="Skip"),
    Command(key="repeat", fn=handle_repeat_layer, desc="Repeat"),
    Command(key="swap", abbr=None, fn=handle_swap_layer, desc="Swap layers"),
    Command(key="multiply", fn=handle_multiplier_layer, desc="Multiply"),
    Command(key="add", fn=handle_addition_layer, desc="Add"),
    Command(key="quit", fn=handle_quit, desc="Quit"),
    Command(key="help", fn=print_help, desc="Help"),
    Command(key="print", fn=print_model, desc="Print model"),
    Command(key="diff", fn=print_diff, desc="Print difference to baseline model"),
    Command(key="listen", fn=handle_listen, desc="Listen to the current state"),
    Command(key="restore", abbr="x", fn=handle_restore, desc="Restore model"),
    Command(key="write", fn=handle_write, desc="Write file"),
    Command(key="export", fn=handle_export, desc="Export model to torchscript"),
]


def bend_loop(app: AppState) -> None:
    pt.shortcuts.clear()
    # TODO: splash screen?

    while True:
        try:
            user_input = get_input(
                msg="Enter command", escape_cancels=False, session=main_session
            )

        except (EOFError, KeyboardInterrupt, UserQuitError):
            sys.exit(0)

        try:

            for c in commands:
                if auto_complete(user_input, c.key, c.abbr):
                    c.fn(app)
                    break  # skip other commands that match
        except UserCancelledError:
            continue
        except (KeyboardInterrupt, EOFError, UserQuitError):
            sys.exit(0)
