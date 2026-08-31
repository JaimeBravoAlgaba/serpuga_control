from serpuga_control.playback import PlaybackController


def test_playback_advances_pauses_and_restarts() -> None:
    playback = PlaybackController(frame_count=3)

    assert playback.playing
    assert playback.tick()
    assert playback.index == 1
    assert playback.tick()
    assert playback.index == 2
    assert playback.at_end
    assert not playback.playing

    playback.toggle()
    assert playback.playing
    assert playback.index == 0


def test_manual_steps_pause_and_clamp_to_history() -> None:
    playback = PlaybackController(frame_count=4, index=2)

    assert playback.step_backward() == 1
    assert not playback.playing
    assert playback.step_backward() == 0
    assert playback.step_backward() == 0
    assert playback.step_forward() == 1
    assert playback.seek(99) == 3
    assert playback.progress == 1.0

