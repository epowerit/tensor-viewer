import { describe, expect, it } from "vitest";
import { INITIAL_CAMERA, rotate, type Point3 } from "./volume";
import { thumbnailFrame } from "./volumeProjection";

describe("compact tensor framing", () => {
  it.each([
    [8, 4, 2],
    [1, 1, 1],
    [5, 5, 5],
    [16, 1, 1],
  ])(
    "keeps geometry and silhouette labels in the frame for %j",
    (...extents) => {
      const frame = thumbnailFrame(extents, INITIAL_CAMERA, 28);
      for (const sx of [-1, 1])
        for (const sy of [-1, 1])
          for (const sz of [-1, 1]) {
            const point = rotate(
              extents.map(
                (extent, i) => (extent / 2 + 0.7) * [sx, sy, sz][i],
              ) as Point3,
              INITIAL_CAMERA,
            );
            expect(Math.abs(point[0]) * 28 + 9).toBeLessThan(frame.width / 2);
            expect(Math.abs(point[1]) * 28 + 10).toBeLessThan(frame.height / 2);
          }
    },
  );

  it("reclaims the vertical rotation margin for a long thin tensor", () => {
    const extents = [16, 1, 1];
    const frame = thumbnailFrame(extents, INITIAL_CAMERA, 28);
    const rotatableHeight = Math.hypot(...extents) * 28 + 72;
    expect(frame.height).toBeLessThan(rotatableHeight * 0.5);
  });

  it("does not reserve index labels around a scalar", () => {
    const scalar = thumbnailFrame([1, 1, 1], INITIAL_CAMERA, 28, false);
    const indexed = thumbnailFrame([1, 1, 1], INITIAL_CAMERA, 28);
    expect(scalar.width).toBeLessThan(indexed.width);
    expect(scalar.height).toBeLessThan(indexed.height);
  });
});
