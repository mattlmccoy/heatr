"""Extract N evenly spaced frames from each GIF for visual verification."""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageSequence

REPO = Path(__file__).resolve().parents[2]
GIFS = REPO / "deck_gifs"
FRAMES = GIFS / "frames"
N = 6


def main() -> None:
    targets = sys.argv[1:] or sorted(p.name for p in GIFS.glob("*.gif"))
    for name in targets:
        p = GIFS / name
        im = Image.open(p)
        frames = [f.convert("RGB") for f in ImageSequence.Iterator(im)]
        n = len(frames)
        im.seek(0)
        total_ms = 0
        for f in ImageSequence.Iterator(im):
            total_ms += f.info.get("duration", 83)
        picks = [int(round(i * (n - 1) / (N - 1))) for i in range(N)]
        for i in picks:
            out = FRAMES / f"{p.stem}_f{i:03d}.png"
            frames[i].save(out)
        print(f"{name}: {n} stored frames, {total_ms / 1000:.1f} s, "
              f"{p.stat().st_size / 1e6:.2f} MB, extracted {picks}")


if __name__ == "__main__":
    main()
