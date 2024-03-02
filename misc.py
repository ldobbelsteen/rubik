from datetime import datetime
import os
import re


def print_stamped(s: str):
    """Print with a timestamp."""
    print(f"[{datetime.now().isoformat(' ', 'seconds')}] {s}")


def create_parent_directory(file_path: str):
    """Create the parent directory of a file if it does not exist yet."""
    dir = os.path.dirname(file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)


def natural_sorted(ls: list[str]):
    """Sort a list of strings 'naturally', such that strings with numbers in them
    are sorted increasingly instead of alphabetically."""

    def convert(s: str):
        return int(s) if s.isdigit() else s.lower()

    def alphanum_key(key: str):
        return [convert(c) for c in re.split("([0-9]+)", key)]

    return sorted(ls, key=alphanum_key)
