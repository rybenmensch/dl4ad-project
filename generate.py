import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from model import (
    LayerInfo,
    Net,
    NetTypeEnum,
    NNModel,
    get_shape_preserving_layers_from_net,
    get_weighted_layers_from_net,
)
from modules import AdditionLayer, MultiplierLayer, RepeatingLayer, SkippingLayer
from state import AppState


@dataclass(frozen=True)
class Command:
    key: str
    fn: Callable[[AppState], None | bool]
    name: str
    abbr: str = ""


def get_input(msg: str = "") -> str:
    try:
        c = input(msg).strip().lower()
    except (EOFError, KeyboardInterrupt):
        sys.exit()
    return c


def auto_complete(user_string: str, key: str, abbr="") -> bool:
    ret = False
    if abbr != "":
        ret = abbr == user_string
    return ret or key.startswith(user_string)


def parse_net_type(msg: str) -> NetTypeEnum:
    if auto_complete(msg, "encoder"):
        return NetTypeEnum.Encoder
    elif auto_complete(msg, "decoder"):
        return NetTypeEnum.Decoder
    else:
        raise ValueError()


ParamReturnType = int | float | NetTypeEnum


def get_param(
    fn: Callable[[str], ParamReturnType], prefix: str, thing: str
) -> ParamReturnType:
    while True:
        try:
            return fn(get_input(f"[{prefix}] Enter {thing}: "))
            break
        except ValueError:
            print(f"[{prefix}] Invalid {thing}.")
            continue


def get_net_type_strings(model: NNModel) -> list[str]:
    return [t[1].value for t in model.get_nets_and_types()]


def get_net_type_and_index(model: NNModel, name: str) -> tuple[NetTypeEnum, int]:
    nstr = [format_auto_complete(s) for s in get_net_type_strings(model)]
    nstr = f"net type ({'/'.join(nstr)})"
    net_type = get_param(parse_net_type, name, nstr)
    index = get_param(int, name, "layer index")
    assert isinstance(net_type, NetTypeEnum)
    assert isinstance(index, int)
    return (net_type, index)


def format_auto_complete(string: str, abbr: str = "") -> str:
    if len(string) < 2:
        return string
    if abbr != "":
        return f"[{abbr}]{string}"
    return f"[{string[0]}]{string[1:]}"


def print_help(_: AppState) -> None:
    ln = max(len(c.key) for c in commands) + 2
    padding = 2

    print("[ROOT] List of commands:")
    for c in commands:
        key_str = format_auto_complete(c.key, c.abbr)
        print(f"{key_str:<{ln + padding}} {c.name}")


def print_layer_common(
    app: AppState, fn: Callable[[NNModel, Net], list[LayerInfo]], title: str
) -> None:
    layers_name = [
        (
            fn(app.model, n),
            app.model.from_layer.get_net_type(n).value,
        )
        for n in app.model.get_nets()
    ]

    max_name_len = max(max(len(l.name) for l in layers) for layers, _ in layers_name)

    print(title.upper())
    for layers, name in layers_name:
        print(name.upper())
        filler = " " * len(name)
        for layer in layers:
            inout = layer.inout or ""
            idx = f"({layer.index})"
            fmt = f"{idx:<4} {layer.name:<{max_name_len}} {inout}"
            print(filler, fmt)


def print_shape_preserving_layers(app: AppState) -> None:
    print_layer_common(
        app, get_shape_preserving_layers_from_net, "shape preserving layers"
    )


def print_weighted_layers(app: AppState) -> None:
    print_layer_common(app, get_weighted_layers_from_net, "weighted layers")


def print_model(app: AppState) -> None:
    print_layer_common(
        app,
        lambda model, net: [LayerInfo.from_layer(model, l) for l in net],
        "all layers",
    )


def get_net_and_layer_info(
    model: NNModel, net_type: NetTypeEnum, index: int
) -> tuple[Net, LayerInfo]:
    net = model.get_net(net_type)
    layer_info = LayerInfo.from_layer(model, net[index])
    return (net, layer_info)


def handle_skip_layer(app: AppState) -> None:
    print("[SKIP] can only be applied to shape preserving layers:")
    print_shape_preserving_layers(app)
    net_type, index = get_net_type_and_index(app.model, "SKIP")

    net, layer_info = get_net_and_layer_info(app.model, net_type, index)
    net[index] = SkippingLayer(layer_info)


def handle_repeat_layer(app: AppState) -> None:
    print("[REPEAT] can only be applied to shape preserving layers:")
    print_shape_preserving_layers(app)
    net_type, index = get_net_type_and_index(app.model, "REPEAT")
    num_repeats = get_param(int, "REPEAT", "number of repetitions")

    net, layer_info = get_net_and_layer_info(app.model, net_type, index)
    net[index] = RepeatingLayer(layer_info, repeats=cast(int, num_repeats))


def handle_multiplier_layer(app: AppState) -> None:
    print("[MULT] can only be applied to layers with weights:")
    print_weighted_layers(app)
    net_type, index = get_net_type_and_index(app.model, "MULT")
    mult = get_param(float, "MULT", "multiplication factor")

    net, layer_info = get_net_and_layer_info(app.model, net_type, index)
    layer = net[index]

    net[index] = MultiplierLayer(layer_info, weight_mul=cast(float, mult))


def handle_addition_layer(app: AppState) -> None:
    print("[ADD] can only be applied to layers with weights:")
    print_weighted_layers(app)
    net_type, index = get_net_type_and_index(app.model, "ADD")
    add = get_param(float, "ADD", "addition factor")

    net, layer_info = get_net_and_layer_info(app.model, net_type, index)
    net[index] = AdditionLayer(layer_info, weight_add=cast(float, add))


def write_file(app: AppState) -> None:
    print("write_file")


def restore_model(app: AppState) -> None:
    print("restore_model")


commands = [
    Command(key="skip", fn=handle_skip_layer, name="Skip"),
    Command(key="repeat", fn=handle_repeat_layer, name="Repeat"),
    Command(key="multiply", fn=handle_multiplier_layer, name="Multiply"),
    Command(key="add", fn=handle_addition_layer, name="Add"),
    Command(key="quit", fn=lambda _: True, name="Quit"),
    Command(key="help", fn=print_help, name="Help"),
    Command(key="print", fn=print_model, name="Print model"),
    Command(key="write", fn=write_file, name="Write file"),
    Command(key="restore", abbr="x", fn=restore_model, name="Restore model"),
]

# TODO: when getting the index, check if actually legal!
# TODO: if layer is already 'modified', what to do?


def generate_loop(app: AppState) -> None:
    while True:
        user_input = get_input("[ROOT] Enter command ([h]elp / [q]uit)")

        for c in commands:
            if auto_complete(user_input, c.key, c.abbr):
                ret = c.fn(app)
                if ret == True:
                    sys.exit()
