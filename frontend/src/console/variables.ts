import type { Run, Tensor } from "../api/client";

export type Variable = {
  name: string;
  /** The latest recorded state carrying this name. */
  tensor: Tensor;
  /** Journey node that produced the latest state. */
  nodeId: string;
  line: number | null;
  /** How many recorded states have had this name. */
  states: number;
  /** Other current variables backed by the same storage. */
  sharedWith: string[];
  /** Named only after the operation that produced it. */
  anonymous: boolean;
  /** Not computed yet at the playback position; `tensor` is its first state. */
  pending: boolean;
  /** Written by the step at the playback position. */
  fresh: boolean;
  /** Every recorded state with this name, over the whole run, in order. */
  history: { tensor: Tensor; nodeId: string }[];
  /** Which of `history` is shown (-1 while pending). */
  shown: number;
};

/**
 * Named tensors in order of first use, as they stand after the operation at
 * index `through` (the playback position), or at the end of the run. Names
 * first assigned later are listed as pending, like locals not yet set.
 */
export function variables(trace: Run["trace"], through = Infinity): Variable[] {
  const found = new Map<string, Variable>();
  const histories = new Map<string, Variable["history"]>();
  const record = (
    id: string,
    nodeId: string,
    line: number | null,
    kind?: string,
    // Inputs hold values before any step and are never a step's result.
    index = Number.NEGATIVE_INFINITY,
  ) => {
    const tensor = trace.tensors[id];
    if (!tensor || tensor.role === "parameter") return;
    const history = histories.get(tensor.name) ?? [];
    history.push({ tensor, nodeId });
    histories.set(tensor.name, history);
    const previous = found.get(tensor.name);
    if (index > through) {
      if (!previous)
        found.set(tensor.name, {
          name: tensor.name,
          tensor,
          nodeId,
          line,
          states: 0,
          sharedWith: [],
          anonymous: tensor.name === kind,
          pending: true,
          fresh: false,
          history,
          shown: -1,
        });
      return;
    }
    found.set(tensor.name, {
      name: tensor.name,
      tensor,
      nodeId,
      line,
      states: (previous?.states ?? 0) + 1,
      sharedWith: [],
      anonymous: tensor.name === kind,
      pending: false,
      fresh: index === through,
      history,
      shown: history.length - 1,
    });
  };
  trace.input_ids.forEach((id) => record(id, `input-${id}`, null));
  for (const op of trace.operations) {
    const line = op.source && !op.source.file ? op.source.line : null;
    op.outputs.forEach((id) => record(id, op.id, line, op.kind, op.index));
    op.mutations?.forEach((mutation) => {
      // A write gives an existing name a new state without reassigning it.
      if (!op.outputs.includes(mutation.after))
        record(mutation.after, op.id, line, undefined, op.index);
    });
  }
  // Defined names first, in order of first use; pending names after them.
  const all = [...found.values()];
  const list = [
    ...all.filter((item) => !item.pending),
    ...all.filter((item) => item.pending),
  ];
  for (const item of list)
    item.sharedWith = list
      .filter(
        (other) =>
          other !== item &&
          !other.pending &&
          !item.pending &&
          other.tensor.storage_id === item.tensor.storage_id,
      )
      .map((other) => other.name);
  return list;
}
