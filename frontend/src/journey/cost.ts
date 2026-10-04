import type { Operation, Tensor } from "../api/client";
import { tensorBytes } from "../tensors/memory";

/** Steps that only rearrange or select values: no arithmetic. */
const LAYOUT =
  /^(reshape|view|permute|transpose|t|contiguous|flatten|unflatten|squeeze|unsqueeze|expand|expand_as|chunk|split|unbind|narrow|select|__getitem__|getitem|index|cat|concat|stack|repeat|clone|to|type|float|long|int|embedding|arange|zeros|ones|zeros_like|ones_like|full|empty)$/;
/** Arithmetic per element beyond one, roughly, for the common heavier steps. */
const PER_ELEMENT: Record<string, number> = {
  softmax: 5,
  log_softmax: 5,
  layer_norm: 8,
  group_norm: 8,
  batch_norm: 4,
  rms_norm: 5,
  gelu: 8,
  silu: 4,
  sigmoid: 4,
  tanh: 4,
  exp: 2,
  sqrt: 2,
  rsqrt: 2,
  scaled_dot_product_attention: 0,
};

/**
 * The floating-point operations a step does, estimated from its shapes:
 * matrix products and linear layers count a multiply and an add for every
 * term of every result (2·M·N·K), convolutions likewise over their kernel
 * window, normalisations and activations a few per element, and steps that
 * only move values around none. An estimate, to compare steps, not a
 * measured count.
 */
export function stepFlops(
  op: Operation,
  tensors: Record<string, Tensor>,
): number {
  const out = tensors[op.outputs[0]];
  if (!out) return 0;
  const kind = op.kind.toLowerCase();
  if (LAYOUT.test(kind)) return 0;
  const first = tensors[op.inputs[0]];
  const second = tensors[op.inputs[1]];
  if (/^(matmul|bmm|mm|__matmul__|einsum|baddbmm|addmm)$/.test(kind)) {
    // The contracted size: the first operand's last axis.
    const k = first?.shape.at(-1) ?? 1;
    return 2 * out.numel * k;
  }
  if (kind === "linear") {
    const k = first?.shape.at(-1) ?? 1;
    return 2 * out.numel * k + (op.inputs.length > 2 ? out.numel : 0);
  }
  if (/^conv(\d)d$|^conv_transpose\dd$/.test(kind) && second) {
    // Every output value sums over its kernel window and input channels.
    const window = second.shape.slice(1).reduce((a, b) => a * b, 1);
    return 2 * out.numel * window;
  }
  if (kind === "scaled_dot_product_attention" && first && second) {
    // Scores (Q·Kᵀ), their softmax, and weights times values.
    const queries = first.shape.at(-2) ?? 1;
    const keys = second.shape.at(-2) ?? 1;
    const depth = first.shape.at(-1) ?? 1;
    const heads = first.numel / (queries * depth);
    return heads * queries * keys * (4 * depth + 5);
  }
  return out.numel * (1 + (PER_ELEMENT[kind] ?? 0));
}

/**
 * The new memory a step's results take, in bytes. A view of an input (a
 * reshape, a transpose) shares that input's storage and takes none.
 */
export function stepBytes(
  op: Operation,
  tensors: Record<string, Tensor>,
): number {
  const shared = new Set(
    op.inputs.flatMap((id) => tensors[id]?.storage_id ?? []),
  );
  return op.outputs.reduce((total, id) => {
    const tensor = tensors[id];
    if (!tensor || (tensor.storage_id && shared.has(tensor.storage_id)))
      return total;
    return total + (tensorBytes(tensor) ?? 0);
  }, 0);
}

const units = (value: number, base: number, names: string[]) => {
  let at = 0;
  while (value >= base && at < names.length - 1) {
    value /= base;
    at++;
  }
  const text =
    value >= 100 || at === 0 ? Math.round(value) : value.toPrecision(2);
  return `${text} ${names[at]}`;
};

/** "1.2 MFLOP", "0 FLOP". */
export const flopsText = (flops: number) =>
  units(flops, 1000, ["FLOP", "kFLOP", "MFLOP", "GFLOP", "TFLOP"]);
/** "48 KB". */
export const bytesText = (bytes: number) =>
  units(bytes, 1024, ["B", "KB", "MB", "GB"]);
