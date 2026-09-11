# SPDX-FileCopyrightText: 2026-present Inria
# SPDX-License-Identifier: BSD-3-Clause

"""Named bounds: the rectangles of the frame in use right now.

A bounds is session configuration, a view into the frame that carries no
homography: changing it never touches a calibration file. Named bounds come
from the `[bounds.<name>]` tables of a dotbot config file, with the package
defaults below as the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BOUNDS_DEFAULT = "arena"


@dataclass(frozen=True)
class Bounds:
    """One rectangle in frame millimetres.

    Edges are named as the console draws the frame: x grows right, y grows
    down, so `top` is the low-y edge.
    """

    x: int
    y: int
    w: int
    h: int
    name: str = ""

    @property
    def x_max(self) -> int:
        return self.x + self.w

    @property
    def y_max(self) -> int:
        return self.y + self.h

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    def as_dict(self) -> dict[str, int]:
        """The four numbers, the shape every consumer of bounds receives."""
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


# The C405 layout.
NAMED_BOUNDS_DEFAULT: dict[str, Bounds] = {
    "arena": Bounds(0, 0, 2000, 2000, "arena"),
    "annex": Bounds(0, 2000, 2000, 2000, "annex"),
    "arena+annex": Bounds(0, 0, 2000, 4000, "arena+annex"),
    "wing": Bounds(2000, 2610, 1330, 1390, "wing"),
    "dev-corner": Bounds(1000, 0, 1000, 1000, "dev-corner"),
}


@dataclass
class BoundsRegistry:
    """The named bounds available this session, and the resolver over them."""

    named: dict[str, Bounds] = field(
        default_factory=lambda: dict(NAMED_BOUNDS_DEFAULT)
    )

    def resolve(self, spec: str) -> Bounds:
        """One bounds from a name, a `+`-joined composite, or `x,y,w,h` in mm.

        A composite is the bounding box of its parts.
        """
        spec = spec.strip()
        if not spec:
            raise ValueError("empty bounds specification")
        if spec in self.named:
            return self.named[spec]
        if "," in spec:
            parts = [p.strip() for p in spec.split(",")]
            if len(parts) != 4:
                raise ValueError(
                    f"bounds {spec!r}: a literal rectangle is x,y,w,h in mm"
                )
            try:
                x, y, w, h = (int(p) for p in parts)
            except ValueError as exc:
                raise ValueError(
                    f"bounds {spec!r}: x,y,w,h must be whole millimetres"
                ) from exc
            return Bounds(x, y, w, h, spec)
        if "+" in spec:
            return self._composite(spec)
        known = ", ".join(sorted(self.named)) or "(none defined)"
        raise ValueError(f"unknown bounds {spec!r}; defined bounds: {known}")

    def resolve_all(self, specs: list[str] | tuple[str, ...]) -> list[Bounds]:
        """The active set: one rectangle per specification, in order."""
        return [self.resolve(spec) for spec in specs]

    def _composite(self, spec: str) -> Bounds:
        parts = [self.resolve(p) for p in spec.split("+")]
        x = min(b.x for b in parts)
        y = min(b.y for b in parts)
        x_max = max(b.x_max for b in parts)
        y_max = max(b.y_max for b in parts)
        return Bounds(x, y, x_max - x, y_max - y, spec)


def union(bounds: list[Bounds]) -> Bounds:
    """The bounding box of an active set, for a renderer that needs one box."""
    if not bounds:
        return NAMED_BOUNDS_DEFAULT[BOUNDS_DEFAULT]
    x = min(b.x for b in bounds)
    y = min(b.y for b in bounds)
    x_max = max(b.x_max for b in bounds)
    y_max = max(b.y_max for b in bounds)
    return Bounds(x, y, x_max - x, y_max - y)
