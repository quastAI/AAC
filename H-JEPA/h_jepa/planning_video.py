"""Example videos of planning runs, for looking at what the planner does.

One GIF per recorded task: the agent's frame next to the goal frame (nearest-neighbour upscaled), with the step,
the progress under the goal test and the achievements unlocked since the start. Long budgets are subsampled to at
most `max_frames` frames; a step that unlocks an achievement or ends the task is always kept, and the last frame is
held for a second. `World.evaluate_from_dataset(recorder=)` calls `capture` before every step and once at the end.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from eval_tasks import ACHIEVEMENT_NAMES

FRAME_MS = 100  # 10 frames per second
HOLD_MS = 1000  # the last frame stays for a second
GAP, HEADER, FOOTER = 4, 14, 30  # pixels


def _last(infos: dict, key: str, i: int):
    """The newest entry of env i (infos hold (envs, history, ...))."""
    value = infos.get(key)
    return None if value is None else np.asarray(value)[i, -1]


class VideoRecorder:
    """Frames of the envs whose entry in `tasks` is a task index (None: not recorded)."""

    def __init__(self, tasks: list[int | None], budget: int, *, scale: int = 4, max_frames: int = 600):
        self.tasks, self.scale = list(tasks), int(scale)
        self.every = max(1, math.ceil(int(budget) / int(max_frames)))
        self.frames: dict[int, list] = {i: [] for i, t in enumerate(self.tasks) if t is not None}
        self.goal: dict[int, np.ndarray] = {}
        self.start: dict[int, np.ndarray] = {}
        self.flags: dict[int, np.ndarray] = {}
        self.done: set[int] = set()

    def capture(self, infos: dict, step: int, final: bool = False) -> None:
        for i in self.frames:
            if i in self.done:
                continue
            goal = _last(infos, "goal", i)
            if goal is not None:
                self.goal[i] = goal  # constant per task unless the goal switches
            flags = _last(infos, "achievements", i)
            flags = np.zeros(len(ACHIEVEMENT_NAMES), bool) if flags is None else flags.astype(bool)
            self.start.setdefault(i, flags)
            unlocked = i in self.flags and bool((flags & ~self.flags[i]).any())
            self.flags[i] = flags
            terminated = bool(_last(infos, "terminated", i)) if "terminated" in infos else False
            if not (final or terminated or unlocked or step % self.every == 0):
                continue
            progress = _last(infos, "goal_progress", i)
            required = _last(infos, "goal_required", i)
            self.frames[i].append(self._frame(i, _last(infos, "pixels", i), step, progress, required, terminated))
            if terminated or final:
                self.done.add(i)

    def _frame(self, i, pixels, step, progress, required, success):
        from PIL import Image, ImageDraw

        s = self.scale
        size = 63 * s if pixels is None else pixels.shape[0] * s
        canvas = Image.new("RGB", (2 * size + 3 * GAP, HEADER + size + FOOTER), (24, 24, 24))
        for k, image in enumerate((pixels, self.goal.get(i))):
            if image is not None:
                panel = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize((size, size), Image.NEAREST)
                canvas.paste(panel, (GAP + k * (size + GAP), HEADER))
        draw = ImageDraw.Draw(canvas)
        draw.text((GAP, 2), "agent", fill=(220, 220, 220))
        draw.text((2 * GAP + size, 2), "goal", fill=(220, 220, 220))
        new = [ACHIEVEMENT_NAMES[j] for j in np.flatnonzero(self.flags[i] & ~self.start[i])]
        line = f"task {self.tasks[i]}  step {step}"
        if progress is not None and required is not None:
            line += f"  progress {round(float(progress) * int(required))}/{int(required)}"
        if success:
            line += "  SUCCESS"
        draw.text((GAP, HEADER + size + 3), line, fill=(120, 220, 120) if success else (220, 220, 220))
        draw.text((GAP, HEADER + size + 16), "unlocked: " + (", ".join(new[-4:]) or "-"), fill=(200, 200, 140))
        return canvas

    def write(self, directory: str | Path) -> list[Path]:
        """One GIF per recorded task, `task_<index>.gif`; returns the paths."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, frames in self.frames.items():
            if not frames:
                continue
            path = directory / f"task_{self.tasks[i]:03d}.gif"
            durations = [FRAME_MS] * (len(frames) - 1) + [HOLD_MS]  # Pillow merges identical frames, summing them
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
            paths.append(path)
        return paths


def recorder_for(cfg, start: int, end: int) -> VideoRecorder | None:
    """A recorder for the tasks start..end-1 of a run that fall among its first `eval.videos` tasks, or None."""
    n = int(cfg.eval.get("videos", 0) or 0)
    if start >= n:
        return None
    tasks = [t if t < n else None for t in range(start, end)]
    return VideoRecorder(
        tasks, int(cfg.eval.eval_budget), scale=int(cfg.eval.get("video_scale", 4)),
        max_frames=int(cfg.eval.get("video_max_frames", 600)),
    )  # fmt: skip
