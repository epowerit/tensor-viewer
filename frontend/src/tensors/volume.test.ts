import { describe, expect, it } from "vitest";
import {
  INITIAL_CAMERA,
  THUMBNAIL_CELL_LIMIT,
  VOLUME_CELL_LIMIT,
  rotate,
  sampleAxis,
  turnCamera,
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
  it.each([4, 8, 16])(
    "shows every index of a small axis of size %s without artificial spacing",
    (size) => {
      const axis = sampleAxis(size);
      expect(axis.entries).toEqual(
        Array.from({ length: size }, (_, index) => ({
          index,
          position: index,
        })),
      );
      expect(axis.gaps).toEqual([]);
      expect(axis.extent).toBe(size);
    },
  );
  it.each([
    [4, 4, 4, 4],
    [2, 3, 8, 8],
    [8, 8, 8],
    [2, 16, 16],
  ])("fully renders a small tensor %j", (...shape) => {
    const volume = volumeLayout(
      shape,
      shape.map(() => 0),
    );
    const cells = volume.blocks.flatMap((block) => block.voxels);
    expect(volume.hasGaps).toBe(false);
    expect(cells.map((cell) => cell.flat).sort((a, b) => a - b)).toEqual(
      Array.from({ length: product(shape) }, (_, i) => i),
    );
  });
  it("maps [batch,tokens,features] to a single actual volume", () => {
    const shape = [32, 16, 768],
      volume = volumeLayout(shape, [0, 0, 0]);
    expect(volume.spatialAxes).toEqual([2, 1, 0]);
    expect(volume.outerAxis).toBeNull();
    const cells = volume.blocks[0].voxels;
    expect(cells).toHaveLength(144);
    expect(volume.samples[1].entries).toHaveLength(16);
    expect(volume.samples[1].gaps).toEqual([]);
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
  it("condenses only the necessary axes of dense previews and expands them when a slice is isolated", () => {
    const shape = [16, 16, 16],
      coordinates = [8, 8, 8];
    const dense = volumeLayout(shape, coordinates);
    expect(dense.samples.filter((axis) => axis.gaps.length)).toHaveLength(1);
    expect(dense.blocks.flatMap((b) => b.voxels)).toHaveLength(
      VOLUME_CELL_LIMIT,
    );
    const slice = volumeLayout(shape, coordinates, 0);
    expect(slice.hasGaps).toBe(false);
    expect(slice.blocks[0].voxels).toHaveLength(256);
  });
  it("bounds both rendering modes while retaining selected coordinates in mixed small and huge shapes", () => {
    for (const shape of [
      [16, 16, 16, 16],
      [1024, 16, 16],
      [2, 3, 8, 8],
      [32, 3, 28, 28],
      [4, 8, 16, 768],
    ]) {
      const coords = shape.map((n) => Math.floor(n / 2));
      for (const limit of [THUMBNAIL_CELL_LIMIT, VOLUME_CELL_LIMIT]) {
        const volume = volumeLayout(shape, coords, undefined, limit);
        const cells = volume.blocks.flatMap((b) => b.voxels);
        expect(cells.length).toBeLessThanOrEqual(limit);
        expect(new Set(cells.map((c) => c.flat)).size).toBe(cells.length);
        expect(cells.some((c) => c.flat === ravel(coords, shape))).toBe(true);
        expect(cells.some((c) => c.flat === product(shape) - 1)).toBe(true);
      }
    }
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

describe("screen-space turntable controls", () => {
  it("moves the near surface in the requested screen direction", () => {
    const front = { yaw: 0, pitch: 0 };
    const near: [number, number, number] = [0, 0, 1];
    expect(rotate(near, turnCamera(front, 20, 0))[0]).toBeGreaterThan(0);
    expect(rotate(near, turnCamera(front, -20, 0))[0]).toBeLessThan(0);
    // Screen y increases downward.
    expect(rotate(near, turnCamera(front, 0, 15))[1]).toBeGreaterThan(0);
    expect(rotate(near, turnCamera(front, 0, -15))[1]).toBeLessThan(0);
  });

  it.each([-170, -32, 90, 170])(
    "keeps controls consistent after turning to yaw %s",
    (yaw) => {
      const camera = { yaw, pitch: 24 };
      const y = (yaw * Math.PI) / 180,
        p = (camera.pitch * Math.PI) / 180;
      // The sphere point facing the viewer at this orientation.
      const near: [number, number, number] = [
        -Math.cos(p) * Math.sin(y),
        Math.sin(p),
        Math.cos(p) * Math.cos(y),
      ];
      expect(rotate(near, camera)[2]).toBeCloseTo(1);
      expect(rotate(near, turnCamera(camera, 10, 0))[0]).toBeGreaterThan(0);
      expect(rotate(near, turnCamera(camera, 0, 10))[1]).toBeGreaterThan(0);
    },
  );

  it("prevents pole flips and preserves geometry across full revolutions", () => {
    expect(turnCamera(INITIAL_CAMERA, 0, -1000).pitch).toBe(80);
    expect(turnCamera(INITIAL_CAMERA, 0, 1000).pitch).toBe(-80);
    const point: [number, number, number] = [1, 2, 3];
    const expected = rotate(point, INITIAL_CAMERA);
    for (const turns of [-1000, -1, 1, 1000]) {
      const actual = rotate(point, turnCamera(INITIAL_CAMERA, turns * 360, 0));
      actual.forEach((value, i) => expect(value).toBeCloseTo(expected[i]));
    }
  });
});
