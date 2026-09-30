"""Small caches used by rfam_gui_server.py.

- BackgroundValue: stale-while-revalidate holder for an expensive value (the run-card list).
- get_or_make_thumb / prewarm_thumbs: JPEG thumbnails cached on local disk, so browsing results
  does not re-read full-size images from the (slow, Dropbox-backed) outputs tree.

Thumbnails live in $HEATR_THUMB_CACHE, default ~/.cache/heatr/thumbs.
"""
from __future__ import annotations

import hashlib
import io
import os
import threading
import time
from pathlib import Path
from typing import Callable, Generic, Iterable, TypeVar

T = TypeVar("T")

THUMB_DIR = Path(os.environ.get("HEATR_THUMB_CACHE", Path.home() / ".cache" / "heatr" / "thumbs"))
_THUMB_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}


class BackgroundValue(Generic[T]):
    """Return the last computed value immediately; rebuild in a background thread once stale.

    The first get() blocks until a value exists. After that, get() never blocks: if the value is
    older than ttl_s a single refresh thread is started and the stale value is returned meanwhile.
    """

    def __init__(self, fn: Callable[[], T], ttl_s: float = 45.0) -> None:
        self._fn = fn
        self._ttl_s = ttl_s
        self._value: T | None = None
        self._stamp = 0.0
        self._has_value = False
        self._lock = threading.Lock()
        self._refreshing = False

    def _compute(self) -> T:
        value = self._fn()
        with self._lock:
            self._value = value
            self._stamp = time.monotonic()
            self._has_value = True
        return value

    def _refresh_bg(self) -> None:
        try:
            self._compute()
        except Exception:
            pass  # keep serving the stale value; the next get() retries
        finally:
            with self._lock:
                self._refreshing = False

    def get(self) -> T:
        with self._lock:
            has_value, value, stamp = self._has_value, self._value, self._stamp
            stale = has_value and (time.monotonic() - stamp) > self._ttl_s
            start_bg = stale and not self._refreshing
            if start_bg:
                self._refreshing = True
        if not has_value:
            return self._compute()
        if start_bg:
            threading.Thread(target=self._refresh_bg, daemon=True).start()
        return value  # type: ignore[return-value]

    def invalidate(self) -> None:
        """Force the next get() to trigger a rebuild (still returns the stale value if one exists)."""
        with self._lock:
            self._stamp = 0.0


def _thumb_path(src: Path, max_px: int) -> Path:
    st = src.stat()
    key = f"{src.resolve()}|{st.st_mtime_ns}|{st.st_size}|{max_px}"
    return THUMB_DIR / (hashlib.sha1(key.encode("utf-8")).hexdigest() + ".jpg")


def get_or_make_thumb(src: Path, max_px: int = 320) -> bytes | None:
    """JPEG thumbnail bytes for an image, cached on local disk. None if it can't be thumbnailed."""
    try:
        src = Path(src)
        if src.suffix.lower() not in _THUMB_EXTS:
            return None
        cached = _thumb_path(src, max_px)
        if cached.exists():
            return cached.read_bytes()
        from PIL import Image  # Pillow ships with matplotlib

        with Image.open(src) as im:
            im.thumbnail((max_px, max_px))
            if im.mode not in ("RGB", "L"):
                bg = Image.new("RGB", im.size, (255, 255, 255))
                rgba = im.convert("RGBA")
                bg.paste(rgba, mask=rgba.split()[-1])
                im = bg
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85, optimize=True)
        data = buf.getvalue()
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, cached)
        return data
    except Exception:
        return None


def prewarm_thumbs(paths: Iterable[Path], max_px: int = 320) -> int:
    """Generate thumbnails for the given images; returns how many are now cached."""
    n = 0
    for p in paths:
        if get_or_make_thumb(Path(p), max_px=max_px) is not None:
            n += 1
    return n
