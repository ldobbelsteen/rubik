"""Miscellaneous tools."""

import re
from datetime import datetime


def print_stamped(s: str):
    """Print with a timestamp."""
    print(f"[{datetime.now().isoformat(' ', 'seconds')}] {s}")


def natural_sorted(ls: list[str]):
    """Sort a list of strings 'naturally', such that strings with numbers in them
    are sorted increasingly instead of alphabetically.
    """

    def convert(s: str):
        return int(s) if s.isdigit() else s.lower()

    def alphanum_key(key: str):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(ls, key=alphanum_key)


def rotate_list(ls: list[int]):
    """Rotate a list by appending its last element to the front."""
    last = ls.pop()
    ls.insert(0, last)


def closest_factors(c: int):
    """Find the two closest factors of a number."""
    a, b, i = 1, c, 0
    while a < b:
        i += 1
        if c % i == 0:
            a = i
            b = c // a
    return b, a
