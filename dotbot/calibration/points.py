# SPDX-FileCopyrightText: 2026-present Inria
# SPDX-License-Identifier: BSD-3-Clause

"""Resolve what the operator typed into frame coordinates.

One function turns a `--points` specification into the millimetre
coordinates stored as a placement's `points_mm`. Nothing downstream ever
sees the specification again: the solver reads (point index, frame mm)
pairs, and the free-text `at` note keeps the provenance.
"""

from __future__ import annotations

from dotbot.bounds import Bounds, BoundsRegistry
from dotbot.robots import ROBOT_DEFAULT, robot_geometry

# The corners of a rectangle, in the order a placement stores them.
CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")

# Which edges each corner touches, so a walled edge can inset it.
_CORNER_EDGES = {
    "top-left": ("top", "left"),
    "top-right": ("top", "right"),
    "bottom-left": ("bottom", "left"),
    "bottom-right": ("bottom", "right"),
}


def corner_mark(
    bounds: Bounds, corner: str, robot: str = ROBOT_DEFAULT
) -> tuple[float, float]:
    """Where a robot's photodiode lands when placed at `corner`, at heading 0.

    A corner whose edges are both open resolves to the exact frame corner. A
    walled edge moves the mark inward by the robot's clearance to that edge,
    because the body cannot occupy the wall.
    """
    if corner not in _CORNER_EDGES:
        raise ValueError(
            f"unknown corner {corner!r}; expected one of {', '.join(CORNERS)}"
        )
    geometry = robot_geometry(robot)
    vertical, horizontal = _CORNER_EDGES[corner]
    x = bounds.x if horizontal == "left" else bounds.x_max
    y = bounds.y if vertical == "top" else bounds.y_max
    if horizontal in bounds.walls:
        inset = geometry.wall_inset_mm(horizontal)
        x = x + inset if horizontal == "left" else x - inset
    if vertical in bounds.walls:
        inset = geometry.wall_inset_mm(vertical)
        y = y + inset if vertical == "top" else y - inset
    return (float(x), float(y))


def resolve_points(
    spec: str,
    registry: BoundsRegistry | None = None,
    robot: str = ROBOT_DEFAULT,
) -> list[tuple[float, float]]:
    """The frame coordinates one `--points` specification stands for.

    Four forms:

    - `x,y` - one literal point in frame millimetres.
    - `<bounds>` - one point at that bounds' centre.
    - `<bounds>:<corner>` - one corner mark, `<corner>` from CORNERS.
    - `<bounds>:corners` - all four corner marks, in CORNERS order.
    """
    registry = registry or BoundsRegistry()
    spec = spec.strip()
    if not spec:
        raise ValueError("empty points specification")

    if ":" in spec:
        name, _, corner = spec.partition(":")
        bounds = registry.resolve(name)
        if corner == "corners":
            return [corner_mark(bounds, c, robot) for c in CORNERS]
        return [corner_mark(bounds, corner, robot)]

    if "," in spec:
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 2:
            raise ValueError(f"points {spec!r}: a literal point is x,y in frame mm")
        try:
            return [(float(parts[0]), float(parts[1]))]
        except ValueError as exc:
            raise ValueError(f"points {spec!r}: x,y must be numbers") from exc

    return [registry.resolve(spec).centre]


def resolve_placement_points(
    specs: list[str] | tuple[str, ...],
    registry: BoundsRegistry | None = None,
    robot: str = ROBOT_DEFAULT,
) -> list[tuple[float, float]]:
    """Every point of one placement, concatenated in the order given."""
    registry = registry or BoundsRegistry()
    points: list[tuple[float, float]] = []
    for spec in specs:
        points.extend(resolve_points(spec, registry, robot))
    return points
