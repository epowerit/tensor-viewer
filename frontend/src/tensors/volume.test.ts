import { describe, expect, it } from "vitest";
import {
  INITIAL_CAMERA,
  rotate,
  sampleAxis,
  volumeLayout,
  voxelFaces,
} from "./volume";
import { product, ravel } from "./coordinates";

describe("indexed volume geometry", () => {
  it("preserves first two and final indices with exact omitted ranges", () => {
    expect(sampleAxis(3).entries.map((e) => e.index)).toEqual([0, 1, 2]);
    expect(sampleAxis(768).entries.map((e) => e.index)).toEqual([0, 1, 767]);
    expect(sampleAxis(768).gaps.map((g) => [g.first, g.last])).toEqual([
      [2, 766],
    ]);
    expect(sampleAxis(768, 500).entries.map((e) => e.index)).toEqual([
      0, 1, 500, 767,
    ]);
    expect(sampleAxis(768, 500).gaps.map((g) => [g.first, g.last])).toEqual([
      [2, 499],
      [501, 766],
    ]);
  });
  it("maps [batch,tokens,features] to a single actual volume", () => {
    const shape = [32, 16, 768],
      volume = volumeLayout(shape, [0, 0, 0]);
    expect(volume.spatialAxes).toEqual([2, 1, 0]);
    expect(volume.outerAxis).toBeNull();
    const cells = volume.blocks[0].voxels;
    expect(cells).toHaveLength(27);
    expect(cells.at(-1)?.coords).toEqual([31, 15, 767]);
    expect(cells.at(-1)?.flat).toBe(product(shape) - 1);
  });
  it("creates equally detailed [C,H,W] volumes for separate batches", () => {
    const shape = [32, 3, 28, 28],
      volume = volumeLayout(shape, [0, 0, 0, 0]);
    expect(volume.outerAxis).toBe(0);
    expect(volume.blocks.map((b) => b.index)).toEqual([0, 1, 31]);
    expect(volume.groups.gaps.map((g) => [g.first, g.last])).toEqual([[2, 30]]);
    expect(volume.blocks.map((b) => b.voxels.length)).toEqual([27, 27, 27]);
    expect(volume.blocks[0].voxels.map((v) => v.center)).toEqual(
      volume.blocks[2].voxels.map((v) => v.center),
    );
    for (const block of volume.blocks)
      for (const cell of block.voxels) {
        expect(cell.coords[0]).toBe(block.index);
        expect(cell.flat).toBe(ravel(cell.coords, shape));
      }
  });
  it("bounds huge rank-six previews while retaining the selected interior cell", () => {
    const shape = [2, 3, 32, 1024, 1024, 1024],
      coords = [1, 2, 15, 511, 511, 511];
    const volume = volumeLayout(shape, coords);
    const cells = volume.blocks.flatMap((b) => b.voxels);
    expect(cells).toHaveLength(256);
    expect(cells.some((v) => v.flat === ravel(coords, shape))).toBe(true);
    expect(cells.every((v) => v.coords[0] === 1 && v.coords[1] === 2)).toBe(
      true,
    );
  });
  it("exposes interior layers and handles empty/scalar tensors without invented cells", () => {
    const volume = volumeLayout([4, 5, 6], [2, 3, 4], 0);
    expect(volume.blocks[0].voxels.every((v) => v.coords[0] === 2)).toBe(true);
    expect(
      volumeLayout([0, 3, 3], [0, 0, 0]).blocks.flatMap((b) => b.voxels),
    ).toHaveLength(0);
    expect(volumeLayout([], []).blocks[0].voxels).toEqual([
      { flat: 0, coords: [], center: [0, 0, 0] },
    ]);
  });
  it("rotates geometric positions without changing length and culls rear faces", () => {
    const point: [number, number, number] = [1, 2, 3];
    expect(Math.hypot(...rotate(point, INITIAL_CAMERA))).toBeCloseTo(
      Math.hypot(...point),
    );
    expect(voxelFaces([0, 0, 0], { yaw: 0, pitch: 0 })).toHaveLength(1);
    expect(voxelFaces([0, 0, 0], INITIAL_CAMERA)).toHaveLength(3);
  });
});
