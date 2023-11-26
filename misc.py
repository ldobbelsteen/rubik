from datetime import datetime


def timestamped(s: str):
    return f"[{datetime.now().isoformat(' ', 'seconds')}] {s}"
