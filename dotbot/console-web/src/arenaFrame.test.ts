import { describe, expect, it } from "vitest";

import { arenaToFraction, fractionToArena, headingToGlyphRotation } from "./arenaFrame";
import type { Bounds } from "./types";

const ARENA: Bounds = { x: 0, y: 0, w: 1000, h: 800 };
// The 2 x 2 m square below the arena in the C405 layout.
const ANNEX: Bounds = { x: 0, y: 2000, w: 2000, h: 2000 };

describe("arena frame", () => {
  it("puts the origin at the top-left of the bounds box", () => {
    expect(arenaToFraction({ x: 0, y: 0 }, ARENA)).toEqual({ fx: 0, fy: 0 });
  });

  it("grows y downward, so max-y draws at the bottom", () => {
    expect(arenaToFraction({ x: 1000, y: 800 }, ARENA)).toEqual({ fx: 1, fy: 1 });
    expect(arenaToFraction({ x: 250, y: 200 }, ARENA)).toEqual({ fx: 0.25, fy: 0.25 });
  });

  it("subtracts the bounds origin, so a frame position draws inside the box", () => {
    expect(arenaToFraction({ x: 1500, y: 2500 }, ANNEX)).toEqual({ fx: 0.75, fy: 0.25 });
  });

  it("round-trips a click back to the frame position it was drawn from", () => {
    const p = { x: 612, y: 149 };
    const { fx, fy } = arenaToFraction(p, ARENA);
    expect(fractionToArena(fx, fy, ARENA)).toEqual(p);
  });

  it("round-trips through an offset bounds too", () => {
    const p = { x: 1500, y: 2500 };
    const { fx, fy } = arenaToFraction(p, ANNEX);
    expect(fractionToArena(fx, fy, ANNEX)).toEqual(p);
  });

  it("faces a zero-heading bot at the bottom of the map", () => {
    expect(headingToGlyphRotation(0)).toBe(180);
  });

  it("turns clockwise on screen for a positive heading", () => {
    // Heading 90 is -x in the frame, which is a left-pointing glyph: three
    // quarter turns clockwise from nose-up.
    expect(headingToGlyphRotation(90)).toBe(270);
  });
});
