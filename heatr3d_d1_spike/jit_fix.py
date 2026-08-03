"""D1 FINDING (deployability): FFCx's C JIT cannot compile when the Python
prefix path contains SPACES -- which this repo's path does
("/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/...").

Mechanism: conda's python bakes its install prefix into sysconfig's CFLAGS /
CPPFLAGS / LDFLAGS / LDSHARED strings (e.g. "-isystem <prefix>/include").
distutils splits those strings on WHITESPACE into argv, so the prefix is torn
into fragments and clang reports:

    clang: error: no such file or directory: 'Dropbox/Matthew'

Every dolfinx form assembly goes through this JIT, so NOTHING runs without a
fix. This is a genuine D1 scoring datum (dolfinx deployability on a lab
machine whose project lives in a Dropbox path), not a code bug of ours.

Fix applied here: create a space-free SYMLINK to the environment prefix and
rewrite the sysconfig strings to use it. The symlink is a 0-byte pointer under
~/.cache (a space-free home path); no software is installed outside the spike
directory. Set D1_JIT_LINK to choose another space-free location.

Import this module BEFORE dolfinx in every spike script:
    import jit_fix; jit_fix.apply()
"""
from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path

_VARS = ("CFLAGS", "CPPFLAGS", "LDFLAGS", "LDSHARED", "LDCXXSHARED",
         "BLDSHARED", "PY_CFLAGS", "PY_CPPFLAGS", "PY_LDFLAGS",
         "CONFIGURE_CFLAGS", "CONFIGURE_CPPFLAGS", "CONFIGURE_LDFLAGS")

_applied: dict | None = None


def link_path() -> Path:
    env = os.environ.get("D1_JIT_LINK")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "heatr3d_d1_spike" / "env"


def apply() -> dict:
    """Make the C JIT usable from a space-containing prefix. Idempotent."""
    global _applied
    if _applied is not None:
        return _applied
    prefix = sys.prefix
    if " " not in prefix:
        _applied = {"needed": False, "prefix": prefix}
        return _applied

    link = link_path()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if os.readlink(link) != prefix:
            link.unlink()
    if not link.exists() and not link.is_symlink():
        link.symlink_to(prefix, target_is_directory=True)
    if " " in str(link):
        raise RuntimeError(f"D1_JIT_LINK must be space-free, got {link!r}")

    patched = []
    cfg = sysconfig.get_config_vars()            # forces initialization
    dicts = [cfg]
    try:                                          # setuptools keeps its own copy
        from setuptools._distutils import sysconfig as _dsysconfig
        d = getattr(_dsysconfig, "_config_vars", None)
        if isinstance(d, dict) and d is not cfg:
            dicts.append(d)
    except Exception:                             # pragma: no cover
        pass
    for d in dicts:
        for k in _VARS:
            v = d.get(k)
            if isinstance(v, str) and prefix in v:
                d[k] = v.replace(prefix, str(link))
                patched.append(k)
    _applied = {"needed": True, "prefix": prefix, "link": str(link),
                "patched_vars": sorted(set(patched))}
    return _applied
