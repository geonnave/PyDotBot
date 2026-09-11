import type { Bounds, LH2Position } from "./types";

// The frame, and the single place the console states it.
//
// Origin is the frame's anchor, x grows right, y grows down. Every position
// the console receives is in frame millimetres; the bounds say which part of
// the frame is drawn, so the box origin is subtracted before scaling.

/** Frame mm to a fraction of the bounds box, 0..1 from its top-left corner. */
export function arenaToFraction(p: LH2Position, b: Bounds): { fx: number; fy: number } {
  return { fx: (p.x - b.x) / b.w, fy: (p.y - b.y) / b.h };
}

/** A fraction of the bounds box back to frame mm. */
export function fractionToArena(fx: number, fy: number, b: Bounds): LH2Position {
  return { x: b.x + fx * b.w, y: b.y + fy * b.h };
}

/**
 * Heading in degrees to the CSS rotation for a nose-up glyph. Heading 0 is +y,
 * which points at the bottom of the map, so the glyph turns half a circle
 * before the heading itself applies.
 */
export function headingToGlyphRotation(heading: number): number {
  return 180 + heading;
}
