import { describe, expect, it } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import type { Tensor } from "../api/client";
import { TensorVolume } from "./TensorVolume";
import {
  INITIAL_CAMERA,
  THUMBNAIL_CELL_LIMIT,
  VOLUME_CELL_LIMIT,
  moveVolumeSelection,
  rotate,
  sampleAxis,
  turnCamera,
  volumeLayout,
  voxelFaces,
} from "./volume";
import { product, ravel, unravel } from "./coordinates";

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
  it("adds narrow visual gutters without moving cell centers or changing face directions", () => {
    const center: [number, number, number] = [3, -2, 5];
    const projectedCenter = rotate(center, INITIAL_CAMERA);
    const original = voxelFaces(center, INITIAL_CAMERA);
    const inset = voxelFaces(center, INITIAL_CAMERA, 0.96);
    expect(center).toEqual([3, -2, 5]);
    expect(inset.map((face) => face.visibility)).toEqual(
      original.map((face) => face.visibility),
    );
    for (let face = 0; face < original.length; face++) {
      for (let corner = 0; corner < original[face].points.length; corner++) {
        for (let axis = 0; axis < 3; axis++) {
          expect(inset[face].points[corner][axis]).toBeCloseTo(
            projectedCenter[axis] +
              (original[face].points[corner][axis] - projectedCenter[axis]) *
                0.96,
          );
        }
      }
    }
    const front = { yaw: 0, pitch: 0 };
    const first = voxelFaces([0, 0, 0], front, 0.96)[0].points;
    for (const axis of [0, 1]) {
      const neighborCenter: [number, number, number] = [0, 0, 0];
      neighborCenter[axis] = 1;
      const neighbor = voxelFaces(neighborCenter, front, 0.96)[0].points;
      expect(
        Math.min(...neighbor.map((point) => point[axis])) -
          Math.max(...first.map((point) => point[axis])),
      ).toBeCloseTo(0.04);
    }
  });
});

describe("exact selected planes in the volume renderer", () => {
  it("keeps every undisplayed high-rank coordinate fixed", () => {
    const shape = [2, 3, 4, 5, 6, 7];
    const coordinates = [1, 2, 3, 4, 5, 6];
    const slice = volumeLayout(
      shape,
      coordinates,
      undefined,
      VOLUME_CELL_LIMIT,
      {
        row: 0,
        column: 4,
      },
    );
    expect(slice.spatialAxes).toEqual([4, 0, null]);
    expect(slice.outerAxis).toBeNull();
    expect(slice.blocks).toHaveLength(1);
    expect(slice.blocks[0].index).toBeNull();
    expect(slice.blocks[0].voxels).toHaveLength(12);
    for (const cell of slice.blocks[0].voxels) {
      expect(cell.coords.filter((_, axis) => axis !== 0 && axis !== 4)).toEqual(
        [2, 3, 4, 6],
      );
      expect(cell.flat).toBe(ravel(cell.coords, shape));
      expect(cell.center[2]).toBe(0);
    }
    expect(coordinates).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("swaps display axes without permuting tensor coordinates or cell identity", () => {
    const shape = [3, 4, 5];
    const coordinates = [1, 2, 3];
    const a = volumeLayout(shape, coordinates, undefined, VOLUME_CELL_LIMIT, {
      row: 0,
      column: 2,
    });
    const b = volumeLayout(shape, coordinates, undefined, VOLUME_CELL_LIMIT, {
      row: 2,
      column: 0,
    });
    for (const cell of a.blocks[0].voxels) {
      const swapped = b.blocks[0].voxels.find(
        (other) => other.flat === cell.flat,
      )!;
      expect(swapped.coords).toEqual(cell.coords);
      expect(swapped.center).toEqual([cell.center[1], cell.center[0], 0]);
    }
  });

  it("bounds huge planes and retains selected interior indices and precise gaps", () => {
    const shape = [1024, 1_000_000, 768, 32];
    const coordinates = [511, 543_210, 600, 16];
    const slice = volumeLayout(
      shape,
      coordinates,
      undefined,
      VOLUME_CELL_LIMIT,
      {
        row: 1,
        column: 3,
      },
    );
    expect(slice.blocks[0].voxels).toHaveLength(16);
    expect(slice.samples[0].entries.map((entry) => entry.index)).toEqual([
      0, 1, 16, 31,
    ]);
    expect(slice.samples[1].gaps.map((gap) => [gap.first, gap.last])).toEqual([
      [2, 543_209],
      [543_211, 999_998],
    ]);
    expect(
      slice.blocks[0].voxels.some(
        (cell) => cell.flat === ravel(coordinates, shape),
      ),
    ).toBe(true);
    expect(
      slice.blocks[0].voxels.every(
        (cell) => cell.coords[0] === 511 && cell.coords[2] === 600,
      ),
    ).toBe(true);
  });

  it("ignores volume isolation and only condenses small planes to meet their budget", () => {
    const shape = [32, 16, 16];
    const coords = [20, 8, 8];
    const plane = { row: 1, column: 2 };
    const complete = volumeLayout(shape, coords, 1, VOLUME_CELL_LIMIT, plane);
    expect(complete.hasGaps).toBe(false);
    expect(complete.blocks[0].voxels).toHaveLength(256);
    const bounded = volumeLayout(shape, coords, 2, 64, plane);
    expect(bounded.blocks[0].voxels).toHaveLength(64);
    expect(
      bounded.blocks[0].voxels.some(
        (cell) => cell.flat === ravel(coords, shape),
      ),
    ).toBe(true);
  });

  it("handles empty, scalar, singleton and null plane axes without phantom cells", () => {
    const view = (
      shape: number[],
      coords: number[],
      row: number | null,
      column: number | null,
    ) =>
      volumeLayout(shape, coords, undefined, VOLUME_CELL_LIMIT, {
        row,
        column,
      });
    for (const shape of [
      [0, 3, 4],
      [2, 0, 4],
      [2, 3, 0],
    ]) {
      expect(
        view(shape, [0, 0, 0], 1, 2).blocks.flatMap((block) => block.voxels),
      ).toEqual([]);
    }
    expect(view([], [], null, null).blocks[0].voxels).toEqual([
      { flat: 0, coords: [], center: [0, 0, 0] },
    ]);
    for (const [row, column] of [
      [null, 0],
      [0, null],
    ]) {
      expect(
        view([3], [1], row, column).blocks[0].voxels.map((cell) => cell.flat),
      ).toEqual([0, 1, 2]);
    }
    expect(view([8, 1, 16], [6, 0, 9], 1, 2).blocks[0].voxels).toHaveLength(16);
    expect(view([3, 4], [2, 3], null, null).blocks[0].voxels).toEqual([
      { flat: 11, coords: [2, 3], center: [0, 0, 0] },
    ]);
  });

  it("navigates chosen axes while preserving fixed coordinates and refuses depth keys in slices", () => {
    const shape = [4, 5, 6];
    const index = ravel([2, 3, 4], shape);
    const plane = { row: 2, column: 0 };
    for (const [key, expected] of [
      ["ArrowRight", [3, 3, 4]],
      ["ArrowLeft", [1, 3, 4]],
      ["ArrowDown", [2, 3, 5]],
      ["ArrowUp", [2, 3, 3]],
    ] as const) {
      expect(
        unravel(moveVolumeSelection(index, shape, key, plane)!, shape),
      ).toEqual(expected);
    }
    expect(moveVolumeSelection(index, shape, "PageDown", plane)).toBeNull();
    expect(moveVolumeSelection(index, shape, "PageUp", plane)).toBeNull();
    expect(
      unravel(moveVolumeSelection(index, shape, "PageUp")!, shape),
    ).toEqual([1, 3, 4]);
    expect(
      moveVolumeSelection(0, [], "ArrowRight", { row: null, column: null }),
    ).toBe(0);
    expect(
      moveVolumeSelection(1, [3], "ArrowDown", { row: null, column: 0 }),
    ).toBe(1);
    expect(
      moveVolumeSelection(2, [3], "ArrowRight", { row: null, column: 0 }),
    ).toBe(2);
    expect(
      moveVolumeSelection(0, [0, 4], "ArrowRight", { row: 0, column: 1 }),
    ).toBe(0);
  });

  it("renders front-facing cells with exact selected values and explicit slice context", () => {
    const tensor: Tensor = {
      id: "slice",
      name: "x",
      role: "input",
      shape: [2, 2, 2],
      axes: ["batch", "rows", "columns"],
      strides: [4, 2, 1],
      storage_offset: 0,
      storage_id: "s0",
      dtype: "torch.float32",
      contiguous: true,
      numel: 8,
      values: [0, 1, 2, 3, 4, 5, 6, 7],
      minimum: 0,
      maximum: 7,
    };
    const html = renderToStaticMarkup(
      createElement(TensorVolume, {
        tensor,
        selected: 5,
        plane: { row: 1, column: 2 },
        onSelect: () => {},
      }),
    );
    expect(html).toContain("x selected 2D slice of 3-dimensional tensor");
    expect(html).toContain(
      "<title>Selected 2D slice. Each square is one indexed element;",
    );
    expect(html).toContain("x cell [1, 0, 1] = 5");
    expect(html.match(/data-volume-index=/g)).toHaveLength(4);
    expect(html.match(/fill="url\(/g)).toHaveLength(4);
    expect(html).not.toContain("x cell [0,");
    expect(html).not.toContain("Rotate left");
    expect(html).not.toContain("Complete tensor");
    expect(html).toContain("Columns");
    expect(html).toContain("Rows");
  });

  it.each([
    { shape: [], count: 1, context: "a scalar" },
    { shape: [3], count: 3, context: "a vector" },
    { shape: [0, 3, 4], count: 0, context: "3-dimensional tensor" },
  ])(
    "labels $shape slices honestly without invented geometry",
    ({ shape, count, context }) => {
      const tensor: Tensor = {
        id: "edge",
        name: "x",
        role: "input",
        shape,
        axes: [],
        strides: [],
        storage_offset: 0,
        storage_id: "s0",
        dtype: "torch.float32",
        contiguous: true,
        numel: count,
        values: count ? Array.from({ length: count }, (_, i) => i + 42) : [],
        minimum: count ? 42 : null,
        maximum: count ? 42 + count - 1 : null,
      };
      const html = renderToStaticMarkup(
        createElement(TensorVolume, {
          tensor,
          plane: {
            row: shape.length > 1 ? shape.length - 2 : null,
            column: shape.length ? shape.length - 1 : null,
          },
        }),
      );
      expect(html).toContain(`selected 2D slice of ${context}`);
      expect(html.match(/data-volume-index=/g)?.length ?? 0).toBe(count);
      expect(html).not.toContain("NaN");
      expect(html).not.toContain("Complete tensor");
      if (!count) {
        expect(html).toContain("No cells in the selected 2D slice.");
        expect(html).not.toContain("Select a visible cell");
      }
    },
  );
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
