import type { Operation, Run, Tensor } from "../api/client";
import { ravel } from "../tensors/coordinates";

export type PatchJourney = {
  image: Tensor;
  weight: Tensor;
  bias?: Tensor;
  projected: Tensor;
  flattened?: Tensor;
  tokens?: Tensor;
  projection: Operation;
  flatten?: Operation;
  transpose?: Operation;
  patch: [number, number];
};

const equal = (a: number[], b: number[]) =>
  a.length === b.length && a.every((v, i) => v === b[i]);

/** Follow recorded dependencies and exact layout rules, never just matching shapes. */
export function findPatchJourney(
  run: Run,
  selected: Operation,
): PatchJourney | null {
  const { operations, tensors } = run.trace;
  for (const projection of operations) {
    const patch = projection.lesson.patch_size;
    if (
      projection.status !== "ok" ||
      projection.lesson.interaction !== "patch_projection" ||
      patch?.length !== 2
    )
      continue;
    const image = tensors[projection.inputs[0]],
      weight = tensors[projection.inputs[1]],
      projected = tensors[projection.outputs[0]];
    if (!image || !weight || !projected || projected.shape.length !== 4)
      continue;
    const [batch, features, rows, columns] = projected.shape;
    const base: PatchJourney = {
      image,
      weight,
      projected,
      bias: tensors[projection.inputs[2]],
      projection,
      patch: [patch[0], patch[1]],
    };
    let fallback: PatchJourney | null =
      selected.id === projection.id ? base : null;
    const candidates = operations.filter(
      (op) =>
        op.status === "ok" &&
        op.inputs[0] === projected.id &&
        op.outputs.length === 1 &&
        ["flatten", "view", "reshape"].includes(op.kind) &&
        op.lesson.mapping_rule === "identity" &&
        equal(tensors[op.outputs[0]].shape, [batch, features, rows * columns]),
    );
    for (const flatten of candidates) {
      const flattened = tensors[flatten.outputs[0]];
      const transposes = operations.filter(
        (op) =>
          op.status === "ok" &&
          op.inputs[0] === flattened.id &&
          op.outputs.length === 1 &&
          op.lesson.mapping_rule === "permutation" &&
          equal(op.lesson.axis_order ?? [], [0, 2, 1]) &&
          equal(tensors[op.outputs[0]].shape, [
            batch,
            rows * columns,
            features,
          ]),
      );
      for (const transpose of transposes) {
        if ([projection.id, flatten.id, transpose.id].includes(selected.id))
          return {
            ...base,
            flatten,
            flattened,
            transpose,
            tokens: tensors[transpose.outputs[0]],
          };
      }
      if ([projection.id, flatten.id].includes(selected.id))
        fallback = { ...base, flatten, flattened };
    }
    if (fallback) return fallback;
  }
  return null;
}

export function patchPosition(journey: PatchJourney, token: number) {
  const columns = journey.projected.shape[3];
  const row = Math.floor(token / columns),
    column = token % columns;
  return {
    row,
    column,
    y: row * journey.patch[0],
    x: column * journey.patch[1],
  };
}

export function patchCell(
  journey: PatchJourney,
  batch: number,
  token: number,
  feature: number,
  channel: number,
  dy: number,
  dx: number,
) {
  const p = patchPosition(journey, token);
  const inputCoordinates = [batch, channel, p.y + dy, p.x + dx];
  const weightCoordinates = [feature, channel, dy, dx];
  return {
    input: ravel(inputCoordinates, journey.image.shape),
    inputCoordinates,
    weight: ravel(weightCoordinates, journey.weight.shape),
    weightCoordinates,
    projected: ravel(
      [batch, feature, p.row, p.column],
      journey.projected.shape,
    ),
    token: journey.tokens
      ? ravel([batch, token, feature], journey.tokens.shape)
      : null,
    flattened: journey.flattened
      ? ravel([batch, feature, token], journey.flattened.shape)
      : null,
    patchVectorIndex: (channel * journey.patch[0] + dy) * journey.patch[1] + dx,
  };
}

/** Only the selected page is allocated, independently of the full image size. */
export function patchGridWindow(
  journey: PatchJourney,
  token: number,
  pageSize = 8,
) {
  const p = patchPosition(journey, token);
  const [, , rows, columns] = journey.projected.shape;
  const startRow = Math.floor(p.row / pageSize) * pageSize,
    startColumn = Math.floor(p.column / pageSize) * pageSize;
  const visibleRows = Math.min(pageSize, rows - startRow),
    visibleColumns = Math.min(pageSize, columns - startColumn);
  return {
    startRow,
    startColumn,
    rows: visibleRows,
    columns: visibleColumns,
    tokens: Array.from(
      { length: visibleRows * visibleColumns },
      (_, i) =>
        (startRow + Math.floor(i / visibleColumns)) * columns +
        startColumn +
        (i % visibleColumns),
    ),
  };
}
