from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")

_ORDINAL_WORDS = {
    1: "first",
    2: "second",
    3: "third",
    4: "fourth",
    5: "fifth",
    6: "sixth",
    7: "seventh",
    8: "eighth",
    9: "ninth",
    10: "tenth",
    11: "eleventh",
    12: "twelfth",
    13: "thirteenth",
    14: "fourteenth",
    15: "fifteenth",
    16: "sixteenth",
    17: "seventeenth",
    18: "eighteenth",
    19: "nineteenth",
    20: "twentieth",
}


def _ordinal_word(value: int) -> str:
    if value in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[value]
    if 10 <= value % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    return f"{value}{suffix}"


def consecutive_streak_counts(sequence: Sequence[T]) -> list[int]:
    streaks: list[int] = []
    previous = object()
    current_streak = 0

    for item in sequence:
        if item == previous:
            current_streak += 1
        else:
            current_streak = 1
            previous = item
        streaks.append(current_streak)

    return streaks


def format_consecutive_streak_phrase(streak_count: int, action_phrase: str) -> str:
    return f"the {_ordinal_word(streak_count)} consecutive time {action_phrase}"


def format_patternlock_move_label(direction_name: str, streak_count: int) -> str:
    streak_phrase = format_consecutive_streak_phrase(
        streak_count, f"moving {direction_name}"
    )
    return f"move {direction_name}, {streak_phrase}"


def format_routestick_move_label(
    stick_side: str, circling_direction: str, streak_count: int
) -> str:
    base_label = (
        f"move to the nearest {stick_side} target by circling around the stick "
        f"{circling_direction}"
    )
    streak_phrase = format_consecutive_streak_phrase(
        streak_count, f"circling {circling_direction}"
    )
    return f"{base_label}, {streak_phrase}"
