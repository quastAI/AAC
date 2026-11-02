"""The state table of a file and the training clips: `clip` consecutive frames of one episode with their actions."""

from __future__ import annotations

import torch

import bc

SOURCES = (0, 3)  # B1 trains on every episode of the file: the final expert (0) and the early snapshots (3)


def load(path, sources=SOURCES, max_episodes=None) -> bc.StateTable:
    """The compact state and the executed actions (`action`, -1 at the end of an episode) of the given sources."""
    return bc.load_table(path, sources=sources, action_column="action", max_episodes=max_episodes)


def clip_starts(table: bc.StateTable, clip: int) -> torch.Tensor:
    """First frames of the clips: `clip` frames of one episode whose first `clip - 1` frames have an action (the last
    frame's action is not used). A frame with an action is not the last of its episode, so the frame after it is in the
    same episode. Episodes that end in death are not padded with their dead frame, unlike B1's data."""
    starts = bc.chunk_starts(table, clip - 1)
    return starts[starts + clip - 1 < len(table.action)]


def batch(table: bc.StateTable, starts: torch.Tensor, clip: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The states (B, clip, 1345) and the actions (B, clip) of the clips that begin at `starts`."""
    rows = starts[:, None] + torch.arange(clip)
    x = table.features(rows.flatten()).reshape(len(starts), clip, -1)
    return x, table.action[rows].long()
