import type { Operation } from "../api/client";

/** Returned values and observed side effects are distinct in the saved trace. */
export function producedTensorIds(op: Operation): string[] {
  const mutations = op.mutations ?? [];
  return [
    ...new Set([
      ...op.outputs,
      ...mutations.filter((m) => m.kind !== "alias").map((m) => m.after),
      ...mutations.filter((m) => m.kind === "alias").map((m) => m.after),
    ]),
  ];
}

export function mutationInputs(op: Operation): string[] {
  return [...new Set((op.mutations ?? []).map((m) => m.before))].filter(
    (id) => !op.inputs.includes(id),
  );
}

export function tensorProducer(
  operations: Operation[],
  tensorId: string,
  before = operations.length,
) {
  return operations
    .slice(0, before)
    .reverse()
    .find((op) => producedTensorIds(op).includes(tensorId));
}
