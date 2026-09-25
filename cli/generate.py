import copy
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Self, cast

import sounddevice as sd
import torchaudio

from library.model import (
    LayerInfo,
    Net,
    NetTypeEnum,
    NNModel,
    get_all_layers,
    get_shape_preserving_layers,
    get_weighted_layers,
)
from library.layers import AdditionLayer, MultiplierLayer, RepeatingLayer, SkippingLayer
from cli.state import AppState


class PromptEnum(StrEnum):
    @classmethod
    def human_name(cls) -> str:
        name = cls.__name__
        chars = []
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                chars.append(" ")
            chars.append(char.lower())
        return "".join(chars)

    def formatted_choice(self) -> str:
        val = self.value
        shortcut = val[0]

        if val.startswith(shortcut):
            return f"[{val[0]}]{val[1:]}"
        else:
            return f"[{shortcut}]{val}"

    @classmethod
    def prompt_string(cls) -> str:
        options = " / ".join(member.formatted_choice() for member in cls)
        return f"{cls.human_name()} ({options})"

    @classmethod
    def parse(cls, msg: str) -> Self:
        try:
            return cls(msg)
        except ValueError:
            pass

        for member in cls:
            if msg == member.value[0]:
                return member

        prefix_matches = [member for member in cls if member.value.startswith(msg)]
        if len(prefix_matches) == 1:
            return prefix_matches[0]

        raise ValueError(
            f"Cannot parse '{msg}' into a valid option for {cls.human_name()}"
        )

    @classmethod
    def get_param(cls) -> Self:
        return cast(Self, get_param(cls.parse, cls.prompt_string()))


@dataclass(frozen=True)
class Command:
    key: str
    fn: Callable[[AppState], None]
    name: str
    abbr: str = ""


class UserCancelledError(Exception):
    pass


def get_input(msg: str = "") -> str:
    try:
        c = input(msg).strip().lower()
        if c == "q":
            raise UserCancelledError
    except (EOFError, KeyboardInterrupt):
        sys.exit()
    return c


ParamReturnType = int | float | Enum | str | Path


def get_param(fn: Callable[[str], ParamReturnType], thing: str) -> ParamReturnType:
    while True:
        user_input = ""
        try:
            user_input = get_input(f"Enter {thing}: ")
            return fn(user_input)
            break
        except ValueError:
            print(f"Invalid {thing}: {user_input}.")
            continue


def format_auto_complete(string: str, abbr: str = "") -> str:
    if len(string) < 2:
        return string
    if abbr != "":
        return f"[{abbr}]{string}"
    return f"[{string[0]}]{string[1:]}"


def print_menu(cmd: str) -> None:
    print(f"[{cmd}] ", end="")


def print_help(_: AppState) -> None:
    ln = max(len(c.key) for c in commands) + 2
    padding = 2

    print_menu("HELP")
    print("List of commands:")
    for c in commands:
        key_str = format_auto_complete(c.key, c.abbr)
        print(f"{key_str:<{ln + padding}} {c.name}")


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
    print_menu("PRINT")
    mode = PrintMode.get_param()

    if mode == PrintMode.All:
        for v in NetTypeEnum:
            print_layer_common(app.model, get_all_layers(app.model), v.value, True)
    else:
        print_layer_common(
            app.model, get_all_layers(app.model), NetTypeEnum(mode), True
        )


def print_diff(app: AppState) -> None:
    differing = []
    for mod, bl in zip(get_all_layers(app.model), get_all_layers(app.backup_model)):
        if type(mod.get_layer()) != type(bl.get_layer()):
            differing.append((mod, bl))

    if len(differing) == 0:
        print_menu("DIFF")
        print("Model has not been modified")
        return

    encoder_l = [l[0] for l in differing if l[0].net_type == NetTypeEnum.Encoder]
    if len(encoder_l):
        print_layer_common(app.model, encoder_l, NetTypeEnum.Encoder.value, True)

    decoder_l = [l[0] for l in differing if l[0].net_type == NetTypeEnum.Decoder]
    if len(decoder_l):
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


def get_net_type_and_index(
    model: NNModel,
    menu_name: str,
    layer_getter: Callable[[NNModel], list[LayerInfo]],
    layer_title: str,
) -> tuple[NetTypeEnum, int]:
    net_type = NetTypePromptEnum.get_param().to_net_type()

    layers = [l for l in layer_getter(model) if l.net_type == net_type]

    print_menu(menu_name)
    print(f"can be applied to {layer_title}:")
    print_layer_common(model, layers, layer_title, False)

    print_menu(menu_name)
    index = cast(int, get_param(int, "layer index"))

    if index not in [l.index for l in layers]:
        raise IndexError(f"[{menu_name}] Index out of bounds: {index}")

    return (net_type, index)


def handle_skip_layer(app: AppState) -> None:
    try:
        net_type, index = get_net_type_and_index(
            app.model, "SKIP", get_shape_preserving_layers, "shape preserving layers"
        )
    except IndexError as e:
        print_menu("SKIP")
        print(e)
        return

    net, layer_info = get_net_and_layer_info(app.model, net_type, index)
    net[index] = SkippingLayer(layer_info)


def handle_repeat_layer(app: AppState) -> None:
    try:
        net_type, index = get_net_type_and_index(
            app.model, "REPEAT", get_shape_preserving_layers, "shape preserving layers"
        )
    except IndexError as e:
        print_menu("REPEAT")
        print(e)
        return

    print_menu("REPEAT")
    num_repeats = get_param(int, "number of repetitions")

    net, layer_info = get_net_and_layer_info(app.model, net_type, index)
    net[index] = RepeatingLayer(layer_info, repeats=cast(int, num_repeats))


def handle_multiplier_layer(app: AppState) -> None:
    try:
        net_type, index = get_net_type_and_index(
            app.model, "MULT", get_weighted_layers, "layers with weights"
        )
    except IndexError as e:
        print_menu("MULT")
        print(e)
        return

    print_menu("MULT")
    mult = get_param(float, "multiplication factor")

    net, layer_info = get_net_and_layer_info(app.model, net_type, index)
    net[index] = MultiplierLayer(layer_info, weight_mul=cast(float, mult))


def handle_addition_layer(app: AppState) -> None:
    try:
        net_type, index = get_net_type_and_index(
            app.model, "ADD", get_weighted_layers, "layers with weights"
        )
    except IndexError as e:
        print_menu("ADD")
        print(e)
        return

    print_menu("ADD")
    add = get_param(float, "addition factor")

    net, layer_info = get_net_and_layer_info(app.model, net_type, index)
    net[index] = AdditionLayer(layer_info, weight_add=cast(float, add))


class ListeningMode(PromptEnum):
    Original = "original"
    Modified = "modified"
    Baseline = "baseline"


def listen(app: AppState) -> None:
    print_menu("LISTEN")
    input_index = 0
    if len(app.files) > 1:
        print("Select input file for playback:")
        for i, f in enumerate(app.files):
            print(f"{i})", f.path)

        print_menu("LISTEN")
        input_index = cast(int, get_param(int, "input file"))

    try:
        file = app.files[input_index]
    except IndexError:
        print_menu("LISTEN")
        print(f"Index out of bounds: {input_index}")
        return

    print_menu("LISTEN")
    mode = cast(
        ListeningMode,
        get_param(ListeningMode.parse, ListeningMode.prompt_string()),
    )

    if mode == ListeningMode.Modified:
        audio = app.model(file.wav)
        sr = app.model.get_sample_rate()
    elif mode == ListeningMode.Baseline:
        audio = app.backup_model(file.wav)
        sr = app.backup_model.get_sample_rate()
    else:
        audio, sr = file.wav

    try:
        sd.play(audio.numpy().T, samplerate=sr, blocking=True)
    except KeyboardInterrupt:
        pass


class RestoreMode(PromptEnum):
    All = "all"
    Index = "index"


def restore_model(app: AppState) -> None:
    print_menu("RESTORE")
    mode = cast(
        RestoreMode,
        get_param(RestoreMode.parse, RestoreMode.prompt_string()),
    )

    if mode == RestoreMode.All:
        app.model.reset()
        return

    try:
        net_type, index = get_net_type_and_index(
            app.model, "RESTORE", get_all_layers, "all layers"
        )
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
    Prepend = "prepend"
    Append = "append"
    Individual = "individual"


def write_file(app: AppState) -> None:
    mode = WriteFileMode.get_param()

    prep_or_app = ""
    if mode != WriteFileMode.Individual:
        prep_or_app = str(get_param(lambda x: x, f"file name {mode.value}"))

    parent = Path(app.output)
    for f in app.files:
        path = f.path
        stem, suffix = path.stem, path.suffix

        if mode == WriteFileMode.Prepend:
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


commands = [
    Command(key="skip", fn=handle_skip_layer, name="Skip"),
    Command(key="repeat", fn=handle_repeat_layer, name="Repeat"),
    Command(key="multiply", fn=handle_multiplier_layer, name="Multiply"),
    Command(key="add", fn=handle_addition_layer, name="Add"),
    Command(key="quit", fn=lambda _: sys.exit(), name="Quit"),
    Command(key="help", fn=print_help, name="Help"),
    Command(key="print", fn=print_model, name="Print model"),
    Command(key="diff", fn=print_diff, name="Print difference to baseline model"),
    Command(key="listen", fn=listen, name="Listen to the current state"),
    Command(key="restore", abbr="x", fn=restore_model, name="Restore model"),
    Command(key="write", fn=write_file, name="Write file"),
]


def auto_complete(user_string: str, key: str, abbr="") -> bool:
    ret = False
    if abbr != "":
        ret = abbr == user_string
    return ret or key.startswith(user_string)


def generate_loop(app: AppState) -> None:
    while True:
        try:
            user_input = get_input("[ROOT] Enter command ([h]elp / [q]uit)")
        except UserCancelledError:
            sys.exit()

        try:
            app.update_layer_lists()

            for c in commands:
                if auto_complete(user_input, c.key, c.abbr):
                    ret = c.fn(app)
                    if ret == True:
                        sys.exit()
                    break
        except UserCancelledError:
            continue
