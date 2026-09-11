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


def corner_mark(
    bounds: Bounds, corner: str, robot: str = ROBOT_DEFAULT
) -> tuple[float, float]:
    """Where a robot's photodiode lands when placed at `corner`.

    The robot sits inside the rectangle with its PCB edges on the
    rectangle's edge lines and its nose toward the nearest top or bottom
    edge, so the mark is the corner inset by the photodiode's distance to
    those two edges.
    """
    if corner not in CORNERS:
        raise ValueError(
            f"unknown corner {corner!r}; expected one of {', '.join(CORNERS)}"
        )
    dx, dy = robot_geometry(robot).photodiode_inset(corner)
    x = bounds.x if corner.endswith("left") else bounds.x_max
    y = bounds.y if corner.startswith("top") else bounds.y_max
    return (x + dx, y + dy)


def resolve_points(
    spec: str,
    registry: BoundsRegistry | None = None,
    robot: str = ROBOT_DEFAULT,
) -> list[tuple[float, float]]:
    """The frame coordinates one `--points` specification stands for.

    Four forms, where `<bounds>` is a name, a `+`-joined composite, or a
    literal `x,y,w,h` rectangle in millimetres:

    - `x,y` - one literal point in frame millimetres, the photodiode's own
      position, taken exactly as typed.
    - `<bounds>` - one point at that rectangle's centre.
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
        if len(parts) == 4:
            return [registry.resolve(spec).centre]
        if len(parts) != 2:
            raise ValueError(
                f"points {spec!r}: two numbers are a point x,y, four are a "
                f"rectangle x,y,w,h, both in frame mm"
            )
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
