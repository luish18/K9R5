#
# One design point of the SoC, injected from outside.
#
# Every RTL-controllable number in system.py and memsys.py is a default here.
# A sweep overrides a subset of them by pointing HES_DESIGN at a JSON file:
#
#     HES_DESIGN=/path/to/design.json  make hetero OP=...
#
# The GVSoC boards, pipeline/gen_system_header.py and
# pipeline/hetero_platform/engines.py all import hetero.system / hetero.memsys,
# so overriding at this level is the one mechanism that reaches every consumer
# -- including the ones that are not GVSoC and so cannot be reached by gapy's
# --config-opt. The environment is inherited across the subprocesses
# run_hetero.py spawns (generate.py, gvsoc, the compiler), so a design set once
# by the sweep driver applies to the whole pipeline of one cell.
#
# With no HES_DESIGN set, every get() returns the default and the system is
# bit-for-bit the one this repository committed. `gen_system_header.py --check`
# against the committed runtime/mesh/ is the regression test for that.
#
# NOTE: this analysis assumes GVSoC's *legacy* runner, which resolves every
# component parameter from the generated JSON config. That is what this
# checkout uses: USE_GVRUN is unset, and engine/python/gvsoc/runner.py gates the
# legacy stack on its absence. If anyone ever sets USE_GVRUN, the compiled
# platform-tree path comes back and parameters baked into it stop following a
# design file -- re-verify before trusting a sweep in that configuration.
#

import json
import os
import sys
from pathlib import Path

# The loaded design, or {} once we have looked and found nothing. None means
# "not looked yet", which is what lets use() run before the first get().
_SPEC = None

# Keys that have been read, with the value handed out. The sweep driver dumps
# this to record what a cell actually ran with, rather than what it asked for.
_USED = {}


def use(path):
    """Load a design file explicitly, instead of from $HES_DESIGN.

    For the generated per-variant target modules the build matrix writes: they
    call this before importing hetero.soc, so that one gvsoc build can carry
    several compile-time variants under different target names.

    Raises if hetero.system or hetero.memsys has already been imported, because
    their module-level constants are evaluated at import time -- a design
    applied after that point would be silently ignored.
    """
    global _SPEC
    for mod in ('hetero.system', 'hetero.memsys'):
        if mod in sys.modules:
            raise RuntimeError(
                f"hetero.design.use({path!r}) called after {mod} was already "
                "imported -- its constants are evaluated at import time, so the "
                "design would be ignored. Call use() before importing hetero.soc.")
    _SPEC = _load(path)


def _load(path):
    spec = json.loads(Path(path).read_text())
    if not isinstance(spec, dict):
        raise RuntimeError(f"design file {path} must hold a JSON object")
    return spec


def get(key, default):
    """The design's value for `key`, or `default` if it does not set one."""
    global _SPEC
    if _SPEC is None:
        path = os.environ.get('HES_DESIGN')
        _SPEC = _load(path) if path else {}
    value = _SPEC.get(key, default)
    _USED[key] = value
    return value


def used():
    """Every key read so far and the value handed out.

    Only complete once the importing module has finished, so callers should
    import hetero.system and hetero.memsys before reading it.
    """
    return dict(_USED)


def unknown_keys():
    """Keys the design file sets that no consumer asked for.

    A typo in a design file would otherwise be silent -- the sweep would run the
    default and report it under the wrong label. The sweep driver checks this
    after importing system and memsys, and refuses to run a cell if it is
    non-empty.
    """
    if _SPEC is None:
        return []
    return sorted(set(_SPEC) - set(_USED))
