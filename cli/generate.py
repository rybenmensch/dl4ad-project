import asyncio
import copy
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Self, cast

import sounddevice as sd
import torch
import torchaudio
from prompt_toolkit import prompt
from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.shortcuts import ProgressBar
from prompt_toolkit.shortcuts.progress_bar import formatters as pb_formatters
from prompt_toolkit.widgets import Label, RadioList

from cli.state import AppState, AudioTensor
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

    @classmethod
    def get_choice_menu(cls) -> Self:
        return cast(
            Self,
            choose(
                f"Select {cls.human_name()} (↑/↓, Enter, or shortcut key)",
                [
                    (member, member.formatted_choice(), member.value[0])
                    for member in cls
                ],
            ),
        )


@dataclass(frozen=True)
class Command:
    key: str
    fn: Callable[[AppState], None]
    name: str
    abbr: str = ""


class UserCancelledError(Exception):
    pass


def get_input(
    msg: str = "", *, quit_on_q: bool = False, escape_cancels: bool = False
) -> str:
    try:
        prompt_bindings = KeyBindings()
        prompt_bindings.add("c-l")(lambda event: event.app.renderer.clear())
        if escape_cancels:
            prompt_bindings.add("escape")(
                lambda event: event.app.exit(exception=UserCancelledError())
            )
        c = prompt(msg, key_bindings=prompt_bindings).strip().lower()
        if quit_on_q and c == "q":
            raise UserCancelledError
    except (EOFError, KeyboardInterrupt):
        sys.exit()
    return c


ParamReturnType = int | float | Enum | str | Path


def get_param(fn: Callable[[str], ParamReturnType], thing: str) -> ParamReturnType:
    while True:
        user_input = ""
        try:
            sys.stdout.write("\r\n")
            sys.stdout.flush()
            user_input = get_input(f"Enter {thing}:\n", escape_cancels=True)
            return fn(user_input)
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


def choose(
    message: str,
    options: list[tuple[object, str, str | None]],
    *,
    number_keys: dict[str, object] | None = None,
) -> object:
    """Show an arrow-key menu with optional direct letter and number selection."""
    radio = RadioList(
        [(value, label) for value, label, _ in options],
        show_numbers=False,
        select_on_focus=False,
        show_cursor=False,
        show_scrollbar=True,
        open_character="",
        select_character=">",
        close_character="",
    )
    bindings = KeyBindings()
    values = [value for value, _, _ in options]

    @bindings.add("enter", eager=True)
    def accept(event):
        radio._handle_enter()
        event.app.exit(result=radio.current_value)

    @bindings.add("up", eager=True)
    def move_up(event):
        radio._selected_index = max(0, radio._selected_index - 1)
        radio._handle_enter()
        if number_keys:
            reset_typed_number()

    @bindings.add("down", eager=True)
    def move_down(event):
        radio._selected_index = min(len(values) - 1, radio._selected_index + 1)
        radio._handle_enter()
        if number_keys:
            reset_typed_number()

    @bindings.add("escape", eager=True)
    def cancel(event):
        event.app.exit(exception=UserCancelledError())

    @bindings.add("c-c", eager=True)
    def interrupt(event):
        event.app.exit(exception=KeyboardInterrupt())

    @bindings.add("c-d", eager=True)
    def eof(event):
        event.app.exit(exception=EOFError())

    def choose_value(event, value: object) -> None:
        radio._selected_index = values.index(value)
        radio._handle_enter()
        event.app.exit(result=value)

    for _, _, shortcut in options:
        if shortcut is not None:

            @bindings.add(shortcut, eager=True)
            def select_shortcut(event, key=shortcut):
                for value, _, option_shortcut in options:
                    if option_shortcut == key:
                        choose_value(event, value)
                        return

    if number_keys:
        typed_number = ""
        reset_task = None

        def reset_typed_number() -> None:
            nonlocal typed_number, reset_task
            typed_number = ""
            if reset_task is not None:
                reset_task.cancel()
                reset_task = None

        def select_number(event, digit: str) -> None:
            nonlocal typed_number, reset_task
            candidate = typed_number + digit
            if any(index.startswith(candidate) for index in number_keys):
                typed_number = candidate
            elif any(index.startswith(digit) for index in number_keys):
                typed_number = digit
            else:
                typed_number = ""

            value = number_keys.get(typed_number)
            if value is not None and value in values:
                radio._selected_index = values.index(value)
                radio._handle_enter()

            if reset_task is not None:
                reset_task.cancel()

            async def clear_after_pause() -> None:
                nonlocal typed_number, reset_task
                await asyncio.sleep(1)
                typed_number = ""
                reset_task = None

            reset_task = event.app.create_background_task(clear_after_pause())

        for digit in "0123456789":

            @bindings.add(digit, eager=True)
            def select_layer_number(event, key=digit):
                select_number(event, key)

    app = Application(
        layout=Layout(HSplit([Label(message), radio])),
        key_bindings=bindings,
        full_screen=False,
    )
    sys.stdout.write("\r\n")
    sys.stdout.flush()
    result = app.run()
    sys.stdout.write("\r\n")
    sys.stdout.flush()
    return result


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


# def get_differing_models_from_net(model:)
#     pass


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

    @classmethod
    def human_name(cls) -> str:
        return "net type"


def get_net_type_and_index(
    model: NNModel,
    menu_name: str,
    layer_getter: Callable[[NNModel], list[LayerInfo]],
    layer_title: str,
) -> tuple[NetTypeEnum, int]:
    net_type = NetTypePromptEnum.get_choice_menu().to_net_type()

    layers = [l for l in layer_getter(model) if l.net_type == net_type]

    print_menu(menu_name)

    index = cast(
        int,
        choose(
            "Select layer (↑/↓, Enter; type its index to jump to it)",
            [
                (layer.index, f"({layer.index}) {layer.name} {layer.inout or ''}", None)
                for layer in layers
            ],
            number_keys={str(layer.index): layer.index for layer in layers},
        ),
    )

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
    print_menu("LISTEN")
    input_index = 0
    if len(app.files) > 1:
        input_index = cast(
            int,
            choose(
                "Select input file for playback (↑/↓, Enter)",
                [(i, f.path.name, None) for i, f in enumerate(app.files)],
            ),
        )

    try:
        file = app.files[input_index]
    except IndexError:
        print_menu("LISTEN")
        print(f"Index out of bounds: {input_index}")
        return

    mode = ListeningMode.get_choice_menu()
    at = mode.get_audio_tensor(app, file.wav)

    def make_progress_bar(length: str) -> ProgressBar:
        kb = KeyBindings()

        @kb.add("c-d", eager=True)
        def exit_on_eof(event):
            event.app.exit(exception=EOFError())

        custom_formatter = [
            pb_formatters.Label(),
            pb_formatters.Text(": "),
            pb_formatters.Bar(sym_a="#", sym_b="#", sym_c="."),
            pb_formatters.Text(" "),
            pb_formatters.TimeElapsed(),
            pb_formatters.Text(" / ", style="class:time-left"),
            pb_formatters.Text(length, style="class:time-left"),
        ]

        return ProgressBar(key_bindings=kb, formatters=custom_formatter)

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
    print_menu("RESTORE")
    mode = RestoreMode.get_choice_menu()

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
    Keep = "keep"
    Prepend = "prepend"
    Append = "append"
    Individual = "individual"


def write_file(app: AppState) -> None:
    mode = WriteFileMode.get_choice_menu()

    prep_or_app = ""
    if mode == WriteFileMode.Prepend or mode == WriteFileMode.Append:
        print_menu("WRITE")
        prep_or_app = str(get_param(lambda x: x, f"file name {mode.value}"))

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
            print_menu("WRITE")
            name = cast(
                Path, get_param(Path, f"new file name for modified file {path.name}")
            )
            name = name.stem
            name = name + suffix

        path = parent / name
        processed = app.model(f.wav)
        torchaudio.save(path, processed, app.model.get_sample_rate())
        print_menu("WRITE")
        print(f"Wrote file {path}")


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
    listen(app)
    exit()

    while True:
        try:
            user_input = get_input(
                "[ROOT] Enter command ([h]elp / [q]uit)", quit_on_q=True
            )
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
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)
