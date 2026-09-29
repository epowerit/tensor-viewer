import type { ModuleCall, Operation, Run, Tensor } from "../api/client";
import { ravel, unravel } from "../tensors/coordinates";
import { outputIndices, sourceIndex } from "../tensors/relationships";

export type SpatialGrouping = {
  kind: "partition" | "restore" | "merge";
  size: number;
  call: ModuleCall;
  tensors: Tensor[];
  operations: Operation[];
};
const equal = (a: number[], b: number[]) =>
  a.length === b.length && a.every((n, i) => n === b[i]);

/** A class name alone is never evidence: verify the complete recorded layout chain. */
export function spatialGrouping(
  run: Run,
  call: ModuleCall,
): SpatialGrouping | null {
  const kind = (
    {
      WindowPartition: "partition",
      WindowReverse: "restore",
      PatchGrouping: "merge",
    } as const
  )[call.module_type as "WindowPartition" | "WindowReverse" | "PatchGrouping"];
  if (
    !kind ||
    call.outputs.length !== 1 ||
    call.inputs.length !== 1 ||
    call.end_index - call.start_index !== 4
  )
    return null;
  const operations = run.trace.operations.slice(
    call.start_index,
    call.end_index,
  );
  if (operations.length !== 4) return null;
  const tensors = [
    run.trace.tensors[call.inputs[0]],
    ...operations.map((op) => run.trace.tensors[op.outputs[0]]),
  ];
  if (tensors.some((t) => !t || !t.numel) || call.outputs[0] !== tensors[4].id)
    return null;
  const grid = kind === "restore" ? tensors[4] : tensors[0];
  if (grid.shape.length !== 4) return null;
  const [batch, features, height, width] = grid.shape;
  const size =
    kind === "merge"
      ? 2
      : kind === "partition"
        ? tensors[2].shape[2]
        : tensors[1].shape[3];
  if (!Number.isInteger(size) || size < 1 || height % size || width % size)
    return null;
  const rows = height / size,
    columns = width / size;
  let shapes: number[][], orders: (number[] | null)[];
  if (kind === "restore") {
    shapes = [
      [batch * rows * columns, size * size, features],
      [batch, rows, columns, size, size, features],
      [batch, rows, size, columns, size, features],
      [batch, height, width, features],
      grid.shape,
    ];
    orders = [null, [0, 1, 3, 2, 4, 5], null, [0, 3, 1, 2]];
  } else {
    shapes = [
      grid.shape,
      [batch, height, width, features],
      [batch, rows, size, columns, size, features],
      [batch, rows, columns, size, size, features],
      kind === "merge"
        ? [batch, rows, columns, 4 * features]
        : [batch * rows * columns, size * size, features],
    ];
    orders = [
      [0, 2, 3, 1],
      null,
      kind === "merge" ? [0, 1, 3, 4, 2, 5] : [0, 1, 3, 2, 4, 5],
      null,
    ];
  }
  if (
    tensors.some(
      (t, i) =>
        t.dtype !== grid.dtype ||
        t.numel !== grid.numel ||
        !equal(t.shape, shapes[i]),
    )
  )
    return null;
  for (const [i, op] of operations.entries()) {
    const order = orders[i];
    if (
      op.status !== "ok" ||
      op.mutations?.length ||
      op.inputs.length !== 1 ||
      op.outputs.length !== 1 ||
      op.inputs[0] !== tensors[i].id ||
      op.lesson.interaction !== "mapping"
    )
      return null;
    if (
      order
        ? op.kind !== "permute" ||
          op.lesson.mapping_rule !== "permutation" ||
          !equal(op.lesson.axis_order ?? [], order)
        : !["reshape", "view"].includes(op.kind) ||
          op.lesson.mapping_rule !== "identity"
    )
      return null;
  }
  return { kind, size, call, tensors, operations };
}

export function findSpatialGrouping(run: Run, operation: Operation) {
  for (const call of run.trace.module_calls ?? []) {
    if (
      operation.index >= call.start_index &&
      operation.index < call.end_index
    ) {
      const grouping = spatialGrouping(run, call);
      if (grouping) return grouping;
    }
  }
  return null;
}

/** Each path is a verified bijection; this needs at most six coordinates per step. */
export function groupingSelection(
  p: SpatialGrouping,
  index: number,
  from: "input" | "output" = "output",
) {
  const indices = Array<number>(5);
  if (from === "output") {
    indices[4] = index;
    for (let i = 3; i >= 0; i--)
      indices[i] = sourceIndex(
        p.operations[i],
        p.tensors[i],
        p.tensors[i + 1],
        indices[i + 1],
      )!;
  } else {
    indices[0] = index;
    for (let i = 0; i < 4; i++)
      indices[i + 1] = outputIndices(
        p.operations[i],
        p.tensors[i],
        p.tensors[i + 1],
        indices[i],
      )[0];
  }
  return {
    indices,
    coordinates: indices.map((n, i) => unravel(n, p.tensors[i].shape)),
  };
}

/** Highlight one channel in a bounded page of the selected neighborhood. */
export function groupingNeighborhood(p: SpatialGrouping, output: number) {
  const selected = groupingSelection(p, output);
  const gridIndex = p.kind === "restore" ? 4 : 0;
  const grid = p.tensors[gridIndex];
  const [batch, feature, y, x] = selected.coordinates[gridIndex];
  const row = Math.floor(y / p.size),
    column = Math.floor(x / p.size);
  const startY = Math.floor((y % p.size) / 8) * 8,
    startX = Math.floor((x % p.size) / 8) * 8;
  const rows = Math.min(8, p.size - startY),
    columns = Math.min(8, p.size - startX);
  const cells = Array.from({ length: rows * columns }, (_, i) => {
    const gridCell = ravel(
      [
        batch,
        feature,
        row * p.size + startY + Math.floor(i / columns),
        column * p.size + startX + (i % columns),
      ],
      grid.shape,
    );
    return groupingSelection(
      p,
      gridCell,
      p.kind === "restore" ? "output" : "input",
    );
  });
  return {
    batch,
    feature,
    row,
    column,
    localRow: y % p.size,
    localColumn: x % p.size,
    cells,
  };
}
