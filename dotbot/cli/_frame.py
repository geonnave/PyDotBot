# SPDX-FileCopyrightText: 2026-present Inria
# SPDX-License-Identifier: BSD-3-Clause

"""Which coordinate frame this session's calibrations live in.

A frame names a site's coordinate system, so it is read across namespaces -
`swarm lh2-calibration collect` writes into it, `swarm lh2-calibration push`
and `run controller --calibration` look an id up under it - and resolves as
a top-level config key next to `conn` and `swarm_id`. The package default is
deliberately neutral: a real frame is named by the config, never by PyDotBot.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

FRAME_ENV = "DOTBOT_FRAME"


def resolve_frame(
    config: Any = None,
    deployment: Any = None,
    flag: Optional[str] = None,
    environ: Mapping[str, str] = os.environ,
) -> tuple[str, str]:
    """The frame name and the layer it came from.

    `--frame` > `DOTBOT_FRAME` > the selected deployment's `frame` > the
    top-level `frame` > the package default.
    """
    from dotbot.calibration.lighthouse2 import FRAME_NAME_DEFAULT

    if flag:
        return flag, "the command line"
    raw = environ.get(FRAME_ENV)
    if raw:
        return raw, FRAME_ENV
    for layer in (deployment, config):
        value = getattr(layer, "frame", None)
        if value:
            return value, "the config file"
    return FRAME_NAME_DEFAULT, "the default"


def frame_from_context(ctx: Any, flag: Optional[str] = None) -> tuple[str, str]:
    """`resolve_frame` against the config the root group stashed on `ctx.obj`."""
    obj = ctx.obj or {}
    return resolve_frame(
        config=obj.get("config"), deployment=obj.get("deployment"), flag=flag
    )
