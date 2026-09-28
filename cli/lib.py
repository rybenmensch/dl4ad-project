import asyncio
import sys
from collections.abc import Callable
from enum import StrEnum
from typing import NamedTuple, Self, cast

import prompt_toolkit as pt

########################### FORMATTING AND PRINTING


def sel(further: str = "") -> str:
    if further != "":
        further = ", " + further
    return f"↑/↓, Enter{further}: select"


def usage(prev: str = "") -> str:
    if prev != "":
        prev = prev + "; "
    return "(" + prev + "<Esc>: abort; <C-c>, <C-d>: quit" + ")"


def format_auto_complete(string: str, abbr: str = "") -> str:
    if len(string) < 2:
        return string
    if abbr != "":
        return f"[{abbr}]{string}"
    return f"[{string[0]}]{string[1:]}"


def menu_string(cmd: str) -> str:
    return f"[{cmd}] "


def cr_and_flush() -> None:
    sys.stdout.write("\r\n")
    sys.stdout.flush()


########################### USER INPUT


class UserCancelledError(Exception):
    pass


def get_input(
    msg: str = "", *, quit_on_q: bool = False, escape_cancels: bool = False
) -> str:
    try:
        prompt_bindings = pt.key_binding.KeyBindings()
        prompt_bindings.add("c-l")(lambda event: event.app.renderer.clear())
        if escape_cancels:
            prompt_bindings.add("escape")(
                lambda event: event.app.exit(exception=UserCancelledError())
            )
        c = pt.prompt(msg, key_bindings=prompt_bindings).strip().lower()
        if quit_on_q and c == "q":
            raise UserCancelledError
    except (EOFError, KeyboardInterrupt):
        sys.exit()
    return c


def get_param[T](fn: Callable[[str], T], thing: str) -> T:
    while True:
        user_input = ""
        try:
            cr_and_flush()
            user_input = get_input(
                f"Enter {thing}:\n", escape_cancels=True, quit_on_q=False
            )
            return fn(user_input)
        except ValueError:
            print(f"Invalid {thing}: {user_input}.")
            continue


class ChoiceOption[T](NamedTuple):
    value: T
    label: str
    shortcut: str | None


def choose[T](
    message: str,
    options: list[ChoiceOption[T]],
    *,
    number_keys: dict[str, object] | None = None,
) -> T:
    """Show an arrow-key menu with optional direct letter and number selection."""
    radio = pt.widgets.RadioList(
        [(o.value, o.label) for o in options],
        show_numbers=False,
        select_on_focus=False,
        show_cursor=False,
        show_scrollbar=True,
        open_character="",
        select_character=">",
        close_character="",
    )
    bindings = pt.key_binding.KeyBindings()
    values = [o.value for o in options]

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

    def choose_value(event, value: T) -> None:
        radio._selected_index = values.index(value)
        radio._handle_enter()
        event.app.exit(result=value)

    for option in options:
        if option.shortcut is not None:

            @bindings.add(option.shortcut, eager=True)
            def select_shortcut(event, key=option.shortcut):
                for other in options:
                    if other.shortcut == key:
                        choose_value(event, other.value)
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

    app = pt.Application(
        layout=pt.Layout.Layout(pt.Layout.HSplit([pt.widgets.Label(message), radio])),
        key_bindings=bindings,
        full_screen=False,
    )

    cr_and_flush()
    result = app.run()
    cr_and_flush()
    return result


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
        usage_str = usage(sel(format_auto_complete("shortcut")))
        return cast(
            Self,
            choose(
                f"Select {cls.human_name()} {usage_str}",
                [
                    ChoiceOption(member, member.formatted_choice(), member.value[0])
                    for member in cls
                ],
            ),
        )


def auto_complete(user_string: str, key: str, abbr="") -> bool:
    ret = False
    if abbr != "":
        ret = abbr == user_string
    return ret or key.startswith(user_string)
