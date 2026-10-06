import type { ModuleCall, Run } from "../api/client";

/**
 * The model's stack of repeated blocks, as the backend finds it: the longest
 * run of its top-level calls of one module type at numbered paths.
 */
export function layerStack(trace: Run["trace"]): ModuleCall[] {
  const calls = trace.module_calls ?? [];
  const root = calls.find((call) => !call.parent_id);
  if (!root) return [];
  let best: ModuleCall[] = [];
  let run: ModuleCall[] = [];
  for (const call of calls.filter((each) => each.parent_id === root.id)) {
    const numbered = /\.\d+$/.test(call.path) || /^\d+$/.test(call.path);
    if (run.length && numbered && call.module_type === run.at(-1)!.module_type)
      run.push(call);
    else run = numbered ? [call] : [];
    if (run.length > best.length) best = [...run];
  }
  return best.length >= 2 ? best : [];
}

/** The state entering the stack and each block's result, by name. */
export function layerStates(
  trace: Run["trace"],
): { name: string; tensorId: string }[] {
  const stack = layerStack(trace);
  if (!stack.length || !stack[0].inputs?.length) return [];
  if (stack.some((call) => !call.outputs?.length)) return [];
  return [
    { name: `before ${stack[0].path}`, tensorId: stack[0].inputs[0] },
    ...stack.map((call) => ({ name: call.path, tensorId: call.outputs![0] })),
  ];
}

/**
 * The layer state a step belongs to, by name: the block whose call holds it
 * (its steps run from start_index up to, not including, end_index),
 * or the state entering the stack for a step before the first block. A step
 * after the last block, or a model without a stack, has none.
 */
export function layerOfStep(
  trace: Run["trace"],
  stepId: string | null,
): string | null {
  if (!stepId) return null;
  const step = trace.operations.find((op) => op.id === stepId);
  const stack = layerStack(trace);
  if (!step || !stack.length) return null;
  if (step.index < stack[0].start_index) return `before ${stack[0].path}`;
  return (
    stack.find(
      (call) => call.start_index <= step.index && step.index < call.end_index,
    )?.path ?? null
  );
}
