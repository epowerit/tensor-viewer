import type { Operation, Tensor } from "../api/client";

/** Steps that combine their operands cell by cell, broadcasting as needed. */
const ELEMENTWISE =
  /^(add|sub|rsub|mul|div|true_divide|floor_divide|remainder|pow|maximum|minimum|where|masked_fill|eq|ne|gt|ge|lt|le|logical_and|logical_or|atan2|hypot|lerp|addcmul|addcdiv|clamp)$/;

/** How one operand of a step was stretched to the result's shape. */
export type Reuse = {
  /** Each of the operand's cells is used this many times. */
  factor: number;
  operand: Tensor;
  /** The result's axes the operand did not have, or had at size 1. */
  axes: string[];
};

/**
 * The most reused operand of an elementwise step: broadcasting lines shapes
 * up from the last axis, and an axis an operand lacks (or holds at size 1)
 * repeats each of its cells along the result's axis. A single number reused
 * everywhere is left out; it is a constant, not a layout.
 */
export function broadcastReuse(
  op: Operation,
  tensors: Record<string, Tensor>,
): Reuse | null {
  const kind = op.kind.replace(/^__|__$|_$/g, "");
  if (!ELEMENTWISE.test(kind)) return null;
  const out = tensors[op.outputs[0]];
  if (!out?.numel) return null;
  let best: Reuse | null = null;
  for (const id of op.inputs) {
    const operand = tensors[id];
    if (!operand || operand.numel <= 1 || operand.numel >= out.numel) continue;
    const offset = out.shape.length - operand.shape.length;
    if (offset < 0) continue;
    const axes: string[] = [];
    let fits = true;
    out.shape.forEach((size, axis) => {
      const own = axis < offset ? 1 : operand.shape[axis - offset];
      if (own === size) return;
      if (own !== 1) fits = false;
      else axes.push(out.axes[axis] || `axis ${axis}`);
    });
    if (!fits || !axes.length) continue;
    const factor = out.numel / operand.numel;
    if (!best || factor > best.factor) best = { factor, operand, axes };
  }
  return best;
}

/** "×13 bias over tokens", "×2 mask over batch, heads". */
export const reuseText = (reuse: Reuse) =>
  `×${reuse.factor} ${reuse.operand.name} over ${reuse.axes.join(", ")}`;
