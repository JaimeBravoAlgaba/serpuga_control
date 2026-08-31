"""Deterministic playback state for the interactive simulator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlaybackController:
    """Small UI-independent state machine for real-time playback.

    Keeping this separate from Matplotlib makes pause, rewind and single-step
    behaviour deterministic and easy to test.
    """

    frame_count: int
    index: int = 0
    playing: bool = True

    def __post_init__(self) -> None:
        if self.frame_count < 1:
            raise ValueError("frame_count must be at least one")
        if not 0 <= self.index < self.frame_count:
            raise ValueError("index must identify an existing frame")

    @property
    def last_index(self) -> int:
        return self.frame_count - 1

    @property
    def at_end(self) -> bool:
        return self.index == self.last_index

    @property
    def progress(self) -> float:
        if self.last_index == 0:
            return 1.0
        return self.index / self.last_index

    def toggle(self) -> None:
        """Pause/resume; pressing play at the end restarts the replay."""

        if self.playing:
            self.playing = False
            return
        if self.at_end:
            self.index = 0
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def seek(self, index: int, *, pause: bool = True) -> int:
        self.index = min(max(int(index), 0), self.last_index)
        if pause:
            self.playing = False
        return self.index

    def step_backward(self) -> int:
        return self.seek(self.index - 1)

    def step_forward(self) -> int:
        return self.seek(self.index + 1)

    def tick(self) -> bool:
        """Advance one real-time frame and report whether the view changed."""

        if not self.playing:
            return False
        if self.at_end:
            self.playing = False
            return False
        self.index += 1
        if self.at_end:
            self.playing = False
        return True
