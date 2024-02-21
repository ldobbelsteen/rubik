from datetime import datetime


def print_with_stamp(s: str):
    print(f"[{datetime.now().isoformat(' ', 'seconds')}] {s}")
