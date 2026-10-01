import { describe, expect, it } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import type { Tensor } from "../api/client";
import { TensorCard } from "./TensorCard";
import { ravel } from "./coordinates";
import {
  changePlane,
  coordinatesFor,
  defaultPlane,
  hiddenAxes,
  mappedOutputs,
  moveInPlane,
  planeCoordinates,
  planeSize,
  parseCoordinate,
  storagePosition,
} from "./plane";

describe("tensor viewing planes", () => {
  it("jumps directly to large logical coordinates and rejects invalid axis indices", () => {
    expect(parseCoordinate("[4095, 15, 15]", [4096, 16, 16])).toEqual({
      index: 1048575,
      error: "",
    });
    expect(parseCoordinate("3 2 1", [4, 3, 2])).toEqual({
      index: 23,
      error: "",
    });
    expect(parseCoordinate("[]", [])).toEqual({ index: 0, error: "" });
    for (const value of [
      "4096, 0, 0",
      "-1, 0, 0",
      "1.5, 0, 0",
      "Infinity, 0, 0",
      "1, 0",
      "",
    ]) {
      const result = parseCoordinate(value, [4096, 16, 16]);
      expect(result.index).toBeNull();
      expect(result.error).not.toBe("");
    }
    expect(parseCoordinate("0, 0", [0, 8]).index).toBeNull();
    expect(
      parseCoordinate("999999999, 999999999", [1e9, 1e9]).index,
    ).toBeNull();
  });
  it("displays arbitrary axes without permuting logical coordinates or values", () => {
    const shape = [2, 3, 4, 5];
    const plane = { row: 0, column: 2 };
    expect(hiddenAxes(shape, plane)).toEqual([1, 3]);
    expect(planeSize(shape, plane)).toEqual({ rows: 2, columns: 4 });
    const fixed = [0, 2, 0, 4];
    const indices = [];
    for (let row = 0; row < 2; row++)
      for (let column = 0; column < 4; column++) {
        const coords = planeCoordinates(shape, plane, fixed, row, column);
        indices.push(ravel(coords, shape));
      }
    expect(indices).toEqual([44, 49, 54, 59, 104, 109, 114, 119]);
    expect(fixed).toEqual([0, 2, 0, 4]);
  });

  it("swaps occupied viewing axes and keeps an independent axis change valid", () => {
    expect(changePlane({ row: 1, column: 2 }, "row", 2)).toEqual({
      row: 2,
      column: 1,
    });
    expect(changePlane({ row: 1, column: 2 }, "column", 0)).toEqual({
      row: 1,
      column: 0,
    });
  });

  it("uses true strides and offsets rather than the logical flat index", () => {
    expect(ravel([2, 1], [3, 2])).toBe(5);
    expect(storagePosition([2, 1], [1, 3], 7)).toBe(12);
    expect(storagePosition([2, 1], [2, 1], 0)).toBe(5);
    expect(storagePosition([1, 1], [1, 1], 0)).toBe(
      storagePosition([2, 0], [1, 1], 0),
    );
    expect(storagePosition([], [], 9)).toBe(9);
  });

  it("moves across page boundaries in the chosen axes without changing a fixed slice", () => {
    const shape = [2, 10, 12];
    const plane = { row: 2, column: 1 };
    const index = ravel([1, 7, 8], shape);
    expect(
      coordinatesFor(moveInPlane(index, shape, plane, "ArrowRight"), shape),
    ).toEqual([1, 8, 8]);
    expect(
      coordinatesFor(moveInPlane(index, shape, plane, "ArrowDown"), shape),
    ).toEqual([1, 7, 9]);
    expect(
      coordinatesFor(moveInPlane(index, shape, plane, "Home"), shape),
    ).toEqual([1, 0, 8]);
    expect(
      coordinatesFor(moveInPlane(index, shape, plane, "End"), shape),
    ).toEqual([1, 9, 8]);
    const end = ravel([1, 9, 11], shape);
    expect(moveInPlane(end, shape, plane, "ArrowRight")).toBe(end);
    expect(moveInPlane(end, shape, plane, "ArrowDown")).toBe(end);
  });

  it("handles scalars, vectors, and empty leading axes without invalid coordinates", () => {
    expect(defaultPlane(0)).toEqual({ row: null, column: null });
    expect(planeCoordinates([], defaultPlane(0), [], 0, 0)).toEqual([]);
    expect(planeSize([12], defaultPlane(1))).toEqual({ rows: 1, columns: 12 });
    expect(moveInPlane(5, [12], defaultPlane(1), "ArrowUp")).toBe(5);
    expect(coordinatesFor(0, [0, 4, 5])).toEqual([0, 0, 0]);
    expect(moveInPlane(0, [0, 4, 5], defaultPlane(3), "ArrowRight")).toBe(0);
  });

  it("keeps all memberships in overlapping windows and identifies excluded elements", () => {
    expect(mappedOutputs([0, 1, 2, 1, 2, 3, 2, 3, 4], 2)).toEqual([2, 4, 6]);
    expect(mappedOutputs([0, 1, 3, 4], 2)).toEqual([]);
    expect(mappedOutputs(null, 2)).toEqual([]);
  });
});

describe("rendering tensor edge cases", () => {
  const tensor: Tensor = {
    id: "test",
    name: "x",
    role: "input",
    shape: [],
    axes: [],
    strides: [],
    storage_offset: 3,
    storage_id: "storage0",
    dtype: "torch.float32",
    contiguous: true,
    numel: 1,
    values: [42],
    minimum: 42,
    maximum: 42,
  };
  const draw = (value: Tensor) =>
    renderToStaticMarkup(
      createElement(TensorCard, {
        tensor: value,
        label: "Input",
        showValues: true,
        gridFrame: { rows: 4, columns: 5 },
      }),
    );
  it("draws no phantom cells when a non-visible axis is empty", () => {
    const html = draw({
      ...tensor,
      shape: [0, 4, 5],
      axes: ["batch", "rows", "columns"],
      strides: [20, 5, 1],
      numel: 0,
      values: [],
    });
    expect(html).toContain("Empty tensor");
    expect(html).not.toContain("data-cell-index=");
    expect(html).not.toContain("NaN");
  });
  it("renders a scalar as one selectable element with its recorded storage offset", () => {
    const html = draw(tensor);
    expect(html.match(/data-cell-index=/g)).toHaveLength(1);
    expect(html).toContain("Input element scalar value 42");
    expect(html).toContain("storage position 3");
  });
  it("keeps large tensors bounded to 64 cells and never creates one option per slice", () => {
    const html = draw({
      ...tensor,
      shape: [4096, 16, 16],
      axes: ["batch", "rows", "columns"],
      strides: [256, 16, 1],
      numel: 1048576,
      values: Array.from({ length: 1048576 }, (_, i) => i),
      minimum: 0,
      maximum: 1048575,
    });
    expect(html.match(/data-cell-index=/g)).toHaveLength(64);
    expect(html.match(/<option/g)).toHaveLength(8);
    expect(html).toContain("1,048,576 elements");
    expect(html).toContain("Go to cell");
    expect(html.length).toBeLessThan(80000);
  });
});
