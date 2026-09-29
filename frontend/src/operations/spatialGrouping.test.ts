import { describe, expect, it } from "vitest";
import type { ModuleCall, Operation, Run, Tensor } from "../api/client";
import { ravel, unravel } from "../tensors/coordinates";
import {
  findSpatialGrouping,
  groupingNeighborhood,
  groupingSelection,
  spatialGrouping,
} from "./spatialGrouping";

function fixture(kind = "partition", grid = [2, 3, 4, 6], size = 2) {
  const [b, c, h, w] = grid,
    r = h / size,
    n = w / size;
  let shapes: number[][], orders: (number[] | null)[], moduleType: string;
  if (kind === "restore") {
    shapes = [
      [b * r * n, size * size, c],
      [b, r, n, size, size, c],
      [b, r, size, n, size, c],
      [b, h, w, c],
      grid,
    ];
    orders = [null, [0, 1, 3, 2, 4, 5], null, [0, 3, 1, 2]];
    moduleType = "WindowReverse";
  } else {
    shapes = [
      grid,
      [b, h, w, c],
      [b, r, size, n, size, c],
      [b, r, n, size, size, c],
      kind === "merge" ? [b, r, n, 4 * c] : [b * r * n, size * size, c],
    ];
    orders = [
      [0, 2, 3, 1],
      null,
      kind === "merge" ? [0, 1, 3, 4, 2, 5] : [0, 1, 3, 2, 4, 5],
      null,
    ];
    moduleType = kind === "merge" ? "PatchGrouping" : "WindowPartition";
  }
  const tensors: Record<string, Tensor> = Object.fromEntries(
    shapes.map((shape, i) => [
      `t${i}`,
      {
        id: `t${i}`,
        name: `t${i}`,
        shape,
        numel: shape.reduce((a, b) => a * b, 1),
        dtype: "torch.float32",
        value_source: "shape",
        values: [],
        strides: [],
        axes: [],
        storage_id: `storage${i}`,
        storage_offset: 0,
        contiguous: true,
        minimum: null,
        maximum: null,
        role: "intermediate",
      } satisfies Tensor,
    ]),
  );
  const operations = orders.map(
    (order, i) =>
      ({
        id: `op${i}`,
        index: i,
        kind: order ? "permute" : "reshape",
        inputs: [`t${i}`],
        outputs: [`t${i + 1}`],
        status: "ok",
        arguments: {},
        lesson: {
          interaction: "mapping",
          mapping_rule: order ? "permutation" : "identity",
          axis_order: order,
        },
      }) as Operation,
  );
  const call: ModuleCall = {
    id: "call",
    parent_id: null,
    module_type: moduleType,
    path: "stage_0",
    start_index: 0,
    end_index: 4,
    inputs: ["t0"],
    outputs: ["t4"],
  };
  const run = { trace: { tensors, operations, module_calls: [call] } } as Run;
  return { run, call, p: spatialGrouping(run, call)! };
}

describe("verified spatial grouping", () => {
  it("matches every window token to its original image cell and reverses selection", () => {
    const { p } = fixture();
    expect(p).not.toBeNull();
    for (let b = 0; b < 2; b++)
      for (let y = 0; y < 4; y++)
        for (let x = 0; x < 6; x++)
          for (let c = 0; c < 3; c++) {
            const source = ravel([b, c, y, x], [2, 3, 4, 6]);
            const target = ravel(
              [
                b * 6 + Math.floor(y / 2) * 3 + Math.floor(x / 2),
                (y % 2) * 2 + (x % 2),
                c,
              ],
              [12, 4, 3],
            );
            expect(groupingSelection(p, target).indices[0]).toBe(source);
            expect(groupingSelection(p, source, "input").indices[4]).toBe(
              target,
            );
          }
    expect(
      groupingSelection(p, ravel([10, 2, 1], [12, 4, 3])).coordinates[0],
    ).toEqual([1, 1, 3, 2]);
  });
  it("restores windows without confusing image batch and window-batch", () => {
    const { p } = fixture("restore");
    const selected = groupingSelection(p, ravel([1, 1, 3, 2], [2, 3, 4, 6]));
    expect(selected.coordinates[0]).toEqual([10, 2, 1]);
    expect(groupingSelection(p, selected.indices[0], "input").indices).toEqual(
      selected.indices,
    );
    expect(groupingNeighborhood(p, selected.indices[4]).cells).toHaveLength(4);
  });
  it("keeps the top-left, bottom-left, top-right, bottom-right feature order", () => {
    const { p } = fixture("merge");
    const selected = groupingSelection(p, ravel([1, 1, 1, 7], [2, 2, 3, 12]));
    expect(selected.coordinates[0]).toEqual([1, 1, 2, 3]);
    const neighborhood = groupingNeighborhood(p, selected.indices[4]);
    expect(neighborhood.cells.map((cell) => cell.indices[4])).toEqual([
      121, 127, 124, 130,
    ]);
    expect(neighborhood.cells.map((cell) => cell.coordinates[0])).toEqual([
      [1, 1, 2, 2],
      [1, 1, 2, 3],
      [1, 1, 3, 2],
      [1, 1, 3, 3],
    ]);
  });
  it("uses the recorded layout operations rather than only a matching class name", () => {
    const { run, call } = fixture();
    expect(findSpatialGrouping(run, run.trace.operations[2])?.call.id).toBe(
      call.id,
    );
    run.trace.operations[2].lesson.axis_order = [0, 1, 3, 4, 2, 5];
    expect(spatialGrouping(run, call)).toBeNull();
    run.trace.operations[2].lesson.axis_order = [0, 1, 3, 2, 4, 5];
    run.trace.operations[1].inputs = ["t0"];
    expect(spatialGrouping(run, call)).toBeNull();
  });
  it("rejects partial runs, writes, extra operations, missing tensors, and unrelated modules", () => {
    const { run, call } = fixture();
    expect(spatialGrouping(run, { ...call, end_index: 3 })).toBeNull();
    expect(spatialGrouping(run, { ...call, outputs: [] })).toBeNull();
    expect(
      spatialGrouping(run, { ...call, module_type: "SomeOtherModule" }),
    ).toBeNull();
    run.trace.operations[1].status = "error";
    expect(spatialGrouping(run, call)).toBeNull();
    run.trace.operations[1].status = "ok";
    run.trace.operations[1].mutations = [
      { before: "t1", after: "t2", kind: "metadata" },
    ];
    expect(spatialGrouping(run, call)).toBeNull();
    run.trace.operations[1].mutations = [];
    delete run.trace.tensors.t1;
    expect(spatialGrouping(run, call)).toBeNull();
  });
  it("stays bounded on billion-element grids and large windows", () => {
    const { p } = fixture("partition", [1024, 768, 128, 128], 32);
    const last = p.tensors[4].numel - 1;
    expect(groupingSelection(p, last).coordinates[0]).toEqual([
      1023, 767, 127, 127,
    ]);
    const neighborhood = groupingNeighborhood(p, last);
    expect(neighborhood.cells).toHaveLength(64);
    expect(neighborhood.cells.at(-1)!.indices[4]).toBe(last);
    expect(neighborhood.cells[0].coordinates[0]).toEqual([1023, 767, 120, 120]);
    expect(
      neighborhood.cells.every((cell) => cell.coordinates[4][0] === 16383),
    ).toBe(true);
  });
  it("handles size-one windows and noncontiguous logical order", () => {
    const { p } = fixture("partition", [2, 3, 4, 6], 1);
    p.tensors[0].strides = [72, 24, 1, 4];
    p.tensors[0].storage_offset = 15;
    expect(groupingNeighborhood(p, 100).cells).toHaveLength(1);
    const coords = unravel(100, [48, 1, 3]);
    expect(groupingSelection(p, 100).coordinates[0]).toEqual([
      Math.floor(coords[0] / 24),
      coords[2],
      Math.floor((coords[0] % 24) / 6),
      coords[0] % 6,
    ]);
  });
});
