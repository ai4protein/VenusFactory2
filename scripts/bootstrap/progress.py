"""Small stdlib-only terminal UI, also readable in SSH/CI logs."""
from __future__ import annotations

import os
import shutil
import sys
import time


def size(value):
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.1f} {unit}'
        value /= 1024


class Progress:
    def __init__(self, stream=None, enabled=True):
        self._stream = stream
        self.enabled = enabled
        self.tty = self.stream.isatty() and os.getenv('TERM') != 'dumb'
        self.last = 0.0
        self.active = False

    @property
    def stream(self):
        return self._stream if self._stream is not None else sys.stderr

    def line(self, text):
        if not self.enabled:
            return
        if self.active:
            self.stream.write('\r\033[K')
        self.stream.write(str(text) + '\n')
        self.stream.flush()
        self.active = False
        self.last = 0

    def update(self, label, elapsed, done=None, total=None, force=False, initial=0):
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last < (0.15 if self.tty else 15):
            return
        self.last = now
        if done is None:
            spinner = '|/-\\'[int(elapsed * 4) % 4]
            detail = f'{spinner} {elapsed:.0f}s'
        else:
            speed = max(0, done - initial) / max(elapsed, 0.001)
            if total:
                fraction = min(done / total, 1)
                bars = int(fraction * 20)
                eta = max(0, total - done) / speed if speed else 0
                detail = f'[{"#" * bars}{"-" * (20 - bars)}] {fraction:.0%} {size(done)}/{size(total)}'
                if speed:
                    detail += f' {size(speed)}/s ETA {eta:.0f}s'
            else:
                detail = f'{size(done)} {size(speed)}/s'
        text = f'  {label}  {detail}'
        if self.tty:
            text = text[:max(20, shutil.get_terminal_size((80, 24)).columns - 1)]
        self.stream.write(('\r\033[K' if self.tty else '') + text + ('' if self.tty else '\n'))
        self.stream.flush()
        self.active = self.tty


ui = Progress()


def configure(enabled=True):
    global ui
    ui = Progress(enabled=enabled)
