"""Windowing policy for long clips.

Only invoked for clips whose duration exceeds `max_duration_s + tolerance_s`
(see configs/config.yaml) -- the ~20 genuine outliers in the MATCH pool, not
the ~2,160 clips that merely round to 30.0-30.5s. Frame ranges are computed
here; no video is re-encoded, the Dataset class reads the (start_frame,
end_frame) slice lazily at decode time.
"""
from typing import List, TypedDict


class Window(TypedDict):
    window_index: int
    start_frame: int
    end_frame: int
    duration_s: float


def compute_windows(
    duration_s: float,
    fps: float,
    frame_count: int,
    window_s: float,
    min_remainder_s: float,
) -> List[Window]:
    """Split [0, frame_count) into non-overlapping ~window_s-second chunks.

    A trailing chunk shorter than `min_remainder_s` is dropped (its footage is
    not emitted as a window) rather than kept as an undersized window.
    """
    if fps <= 0 or frame_count <= 0:
        return [{"window_index": 0, "start_frame": 0, "end_frame": max(frame_count, 0), "duration_s": duration_s}]

    frames_per_window = max(1, round(window_s * fps))
    windows: List[Window] = []
    start = 0
    idx = 0
    while start < frame_count:
        end = min(start + frames_per_window, frame_count)
        seg_duration = (end - start) / fps
        if seg_duration < min_remainder_s and windows:
            break  # trailing remainder too short to stand alone
        windows.append({
            "window_index": idx,
            "start_frame": start,
            "end_frame": end,
            "duration_s": seg_duration,
        })
        idx += 1
        start = end
    return windows
