import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from model import LayerInfo, NetTypeEnum, get_shape_preserving_layers_from_net
from state import AppState

edit_list: list[Any] = []


def get_input(msg: str = "") -> str:
    try:
        c = input(msg).strip().lower()
    except (EOFError, KeyboardInterrupt):
        sys.exit()
    return c


def auto_complete(user_string: str, string: str) -> bool:
    return string.startswith(user_string)


def format_auto_complete(string: str) -> str:
    if len(string) < 2:
        return string
    return f"[{string[0]}]{string[1:]}"


def format_layer_info(layer: LayerInfo) -> str:
    inout = layer.inout or ""
    return f"({layer.index}) {layer.name} {inout}"


def parse_net_type(msg: str) -> NetTypeEnum:
    if auto_complete(msg, "encoder"):
        return NetTypeEnum.Encoder
    elif auto_complete(msg, "decoder"):
        return NetTypeEnum.Decoder
    else:
        raise ValueError()


def get_param(l, prefix, thing) -> int | float | NetTypeEnum:
    while True:
        try:
            return l(get_input(f"[{prefix}] Enter {thing}: "))
            break
        except ValueError:
            print(f"[{prefix}] Invalid {thing}.")
            continue


def get_net_type_and_index(name: str) -> tuple[NetTypeEnum, int]:
    net_type = get_param(parse_net_type, name, "net type ([e]ncoder/[d]ecoder)")
    index = get_param(int, name, "layer index")
    assert isinstance(net_type, NetTypeEnum)
    assert isinstance(index, int)
    return (net_type, index)


def print_help(_: AppState) -> None:
    ln = max(len(c.key) for c in commands) + 2
    padding = 2

    print("[ROOT] List of commands:")
    for c in commands:
        key_str = format_auto_complete(c.key)
        print(f"{key_str:<{ln + padding}} {c.name}")


def print_model(app: AppState) -> None:
    pass


def print_shape_preserving_layers(app: AppState) -> None:
    layers_name = (
        (
            get_shape_preserving_layers_from_net(app.model, n),
            app.model.from_layer.get_net_type(n).value,
        )
        for n in app.model.get_nets()
    )

    for layers, name in layers_name:
        print(name.upper())
        filler = " " * len(name)
        for layer in layers:
            print(filler, format_layer_info(layer))


def print_weighted_layers(app: AppState) -> None:
    pass


def handle_skip_layer(app: AppState) -> None:
    print("[SKIP] can only be applied to shape preserving layers:")
    print_shape_preserving_layers(app)
    net_type, index = get_net_type_and_index("SKIP")

    # actually do the thing here


def handle_repeat_layer(app: AppState) -> None:
    print("[REPEAT] can only be applied to shape preserving layers:")
    print_shape_preserving_layers(app)
    net_type, index = get_net_type_and_index("REPEAT")
    num_repeats = get_param(int, "REPEAT", "number of repetitions")


def handle_multiplier_layer(app: AppState) -> None:
    print("[MULT] can only be applied to layers with weights:")
    print_weighted_layers(app)
    net_type, index = get_net_type_and_index("MULT")
    mult = get_param(float, "MULT", "multiplication factor")


def handle_addition_layer(app: AppState) -> None:
    print("[ADD] can only be applied to layers with weights:")
    print_weighted_layers(app)
    net_type, index = get_net_type_and_index("ADD")
    add = get_param(float, "ADD", "addition factor")


@dataclass(frozen=True)
class Command:
    key: str
    func: Callable[[AppState], None | bool]
    name: str


commands = [
    Command(key="quit", func=lambda _: True, name="Quit"),
    Command(key="help", func=print_help, name="Help"),
    Command(key="skip", func=handle_skip_layer, name="Skip"),
    Command(key="repeat", func=handle_repeat_layer, name="Repeat"),
    Command(key="multiply", func=handle_multiplier_layer, name="Multiply"),
    Command(key="add", func=handle_addition_layer, name="Add"),
]


def generate_loop(app: AppState) -> None:
    print_shape_preserving_layers(app)
    while True:
        c = get_input("[ROOT] Enter command ([h]elp / [q]uit)")

        for b in commands:
            if auto_complete(c, b.key):
                ret = b.func(app)
                if ret == True:
                    sys.exit()
