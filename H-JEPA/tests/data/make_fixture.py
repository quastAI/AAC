"""Rebuild craftax_classic_episode.npz from a collector file (numpy and h5py only).

    python purejaxql/scripts/collect_craftax_episodes.py --out /tmp/episode.h5 --eps 1.0 \
        --num-episodes 1 --seed-start 0 --max-steps 60 --num-envs 1 \
        --no-require-diamond --no-stop-at-diamond
    python H-JEPA/tests/data/make_fixture.py /tmp/episode.h5

The test compares the planning environment frame by frame with this episode, so regenerate it
only when the episode contract changes (and say why in the commit).
"""

import sys
from pathlib import Path

import h5py
import numpy as np

OUT = Path(__file__).parent / "craftax_classic_episode.npz"


def main(path):
    with h5py.File(path, "r") as f:
        if len(f["ep_len"]) != 1:
            raise SystemExit(f"expected one episode, the file has {len(f['ep_len'])}")
        n = int(f["ep_len"][0])
        action = f["action"][:n, 0]
        np.savez_compressed(
            OUT,
            seed=np.int64(f["seed"][0]),
            actions=action[:-1].astype(np.int8),  # the last frame has no action (NaN)
            pixels=f["pixels"][:n],
            state=f["state"][:n],
            full_state=f["full_state"][:n],
            achievements=f["achievements"][:n],
            reward=f["reward"][:n],
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
