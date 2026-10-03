import type { AxisWiringData } from "./AxisWiring";
import type { Operation, Run, Tensor } from "../api/client";
import { broadcastAxes, tensorAddition } from "../operations/addition";
import { tensorAssembly } from "../operations/assembly";
import { tensorConvolution } from "../operations/convolution";
import { linearProjection } from "../operations/linear";
import { tensorRelation } from "../operations/relations";
import { mutationInputs, producedTensorIds } from "../tensors/provenance";
import { kindName } from "../operations/kindName";

export type OperationSemantics = {
  title: string;
  summary: string;
  /** For a layout capsule: its axes as wiring, input above, result below. */
  wiring?: AxisWiringData;
  /** Entries preserve operand order, including repeated uses of one tensor. */
  inputs: { tensorId: string; label: string }[];
  outputs: { tensorId: string; label: string }[];
};

const shape = (tensor: Tensor) => `[${tensor.shape.join(" × ")}]`;
const equal = (a: number[], b: number[]) =>
  a.length === b.length && a.every((size, axis) => size === b[axis]);
const count = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;
const axisList = (axes: number[]) =>
  `${axes.length === 1 ? "axis" : "axes"} ${axes.join(", ")}`;
const roles: Record<string, string> = {
  input: "Input",
  other: "Other operand",
  condition: "Condition",
  mask: "Mask",
  value: "Value",
  weight: "Weights",
  bias: "Bias",
  running_mean: "Running mean",
  running_var: "Running variance",
};

/** Validate vector/matrix promotion and batch broadcasting without visiting cells. */
function matrixProduct(inputs: (Tensor | undefined)[], output?: Tensor) {
  if (inputs.length !== 2 || !inputs[0] || !inputs[1] || !output) return null;
  const [left, right] = inputs;
  if (!left.shape.length || !right.shape.length) return null;
  const terms = left.shape.at(-1)!;
  if (terms !== right.shape.at(right.shape.length === 1 ? -1 : -2)) return null;
  const a = left.shape.slice(0, -2),
    b = right.shape.slice(0, -2);
  const rank = Math.max(a.length, b.length);
  const batch: number[] = [];
  for (let i = rank; i > 0; i--) {
    const x = a.at(-i) ?? 1,
      y = b.at(-i) ?? 1;
    if (x !== y && x !== 1 && y !== 1) return null;
    batch.push(x === 1 ? y : x);
  }
  const expected = [
    ...batch,
    ...(left.shape.length > 1 ? [left.shape.at(-2)!] : []),
    ...(right.shape.length > 1 ? [right.shape.at(-1)!] : []),
  ];
  return equal(expected, output.shape) ? terms : null;
}

/**
 * Scene copy and operand roles come from the recorded operation and validated
 * geometry. No names such as "tokens" or "images" are guessed from dimensions.
 * Work is bounded by recorded metadata, not by the number of tensor elements.
 */
export function operationSemantics(
  run: Run,
  op: Operation,
): OperationSemantics {
  const inputs = op.inputs.map((tensorId, i) => ({
    tensorId,
    label: op.inputs.length === 1 ? "Input" : `Input ${i + 1}`,
  }));
  const produced = producedTensorIds(op);
  const outputs = produced.map((tensorId, i) => ({
    tensorId,
    label: produced.length === 1 ? "Result" : `Result ${i + 1}`,
  }));
  const result = (title: string, summary: string): OperationSemantics => ({
    title,
    summary,
    inputs,
    outputs,
  });
  const inputTensors = op.inputs.map((id) => run.trace.tensors[id]);
  const outputTensors = op.outputs.map((id) => run.trace.tensors[id]);
  const input = inputTensors[0],
    output = outputTensors[0];
  inputTensors.forEach((tensor, i) => {
    if (tensor?.role === "parameter") inputs[i].label = "Parameter";
  });
  if (op.status !== "ok")
    return result(
      "Operation stopped",
      `${kindName(op.kind)} did not complete. Inspect its recorded inputs and error.`,
    );
  if (op.mutations?.length) {
    inputs.push(
      ...mutationInputs(op).map((tensorId) => ({
        tensorId,
        label: "Before update",
      })),
    );
    outputs.forEach((item) => {
      const mutation = op.mutations!.find((m) => m.after === item.tensorId);
      if (mutation)
        item.label =
          mutation.kind === "alias"
            ? "Shared view"
            : mutation.kind === "metadata"
              ? "Updated layout"
              : "Updated tensor";
    });
    return result(
      "Update tensor state",
      `${kindName(op.kind)} records ${count(op.mutations.length, "tensor update")}. Inspect the before and after states.`,
    );
  }
  if (inputTensors.some((t) => !t) || outputTensors.some((t) => !t))
    return result(
      op.kind,
      "Some tensor snapshots are unavailable in this recording.",
    );

  const assembly = tensorAssembly(run, op);
  if (assembly) {
    if (assembly.joining) {
      inputs.forEach((item, i) => (item.label = `Part ${i + 1}`));
      outputs[0].label = "Joined tensor";
      return result(
        assembly.inserted ? "Stack tensors" : "Join tensors",
        `${count(assembly.parts.length, "tensor")} join along ${assembly.inserted ? "new " : ""}axis ${assembly.axis} to form ${shape(assembly.whole)}.`,
      );
    }
    outputs.forEach((item, i) => (item.label = `Part ${i + 1}`));
    return result(
      "Separate tensor parts",
      `${shape(assembly.whole)} separates into ${count(assembly.parts.length, "tensor")} along axis ${assembly.axis}${assembly.inserted ? "; that axis is removed from each part" : ""}.`,
    );
  }

  const linear = linearProjection(run, op);
  if (linear) {
    ["Input", "Weights", "Bias"].forEach((label, i) => {
      if (inputs[i]) inputs[i].label = label;
    });
    return result(
      "Project features",
      `${linear.features} input features become ${linear.outputs} output features at every leading coordinate${linear.bias ? ", with bias" : ""}.`,
    );
  }
  const convolution = tensorConvolution(run, op);
  if (convolution) {
    ["Input", "Kernel", "Bias"].forEach((label, i) => {
      if (inputs[i]) inputs[i].label = label;
    });
    return result(
      `Slide a ${convolution.dimensions}D kernel`,
      `${convolution.kernel.join(" × ")} kernel · stride ${convolution.stride.join(" × ")} · ${count(convolution.groups, "channel group")}. Each result combines ${count(convolution.terms, "input × weight term")}${convolution.bias ? " and bias" : ""}.`,
    );
  }
  if (["matmul", "mm", "bmm"].includes(op.kind) && outputs.length === 1) {
    const terms = matrixProduct(inputTensors, output);
    if (terms !== null) {
      inputs[0].label = "Left operand";
      inputs[1].label = "Right operand";
      return result(
        "Multiply and accumulate",
        `Each result cell combines ${count(terms, "matching pair")} from the two operands. Result ${shape(output)}.`,
      );
    }
  }

  // Relations other than these can contain per-cell maps. Summaries never scan them.
  const rawRule = op.lesson.relation?.rule;
  const relation =
    outputs.length === 1 &&
    ["reduce", "elementwise", "prefix", "channel_affine", "einsum"].includes(
      String(rawRule),
    )
      ? tensorRelation(op, inputTensors, output)
      : null;
  if (relation?.rule === "reduce") {
    const source = inputTensors[relation.operand]!;
    return result(
      "Reduce a group of values",
      `${op.kind} combines values along ${axisList(relation.axes)}: ${shape(source)} → ${shape(output)}${relation.keepdim ? ". Reduced axes remain at size 1" : ""}.`,
    );
  }
  if (relation?.rule === "prefix")
    return result(
      "Accumulate along an axis",
      `${op.kind} accumulates along axis ${relation.axis}; the shape remains ${shape(output)}.`,
    );
  if (relation?.rule === "channel_affine") {
    relation.roles.forEach((role, i) => {
      inputs[i].label = roles[role] ?? inputs[i].label;
    });
    return result(
      "Normalize by channel",
      `Each position uses the recorded statistics for its coordinate on axis ${relation.axis}. Result ${shape(output)}.`,
    );
  }
  if (relation?.rule === "einsum")
    return result(
      "Combine indexed dimensions",
      `${relation.inputs.join(", ")} → ${relation.output || "scalar"}. Result ${shape(output)}.`,
    );

  const addition = tensorAddition(run, op);
  if (relation?.rule === "elementwise" || addition) {
    if (relation?.rule === "elementwise")
      relation.roles.forEach((role, i) => {
        inputs[i].label = roles[role] ?? inputs[i].label;
      });
    else {
      inputs[0].label = "Left operand";
      inputs[1].label = "Right operand";
    }
    const expanded = inputTensors.flatMap((tensor, i) => {
      const axes = broadcastAxes(tensor, output);
      return axes.length
        ? [`${inputs[i].label} reuses values along ${axisList(axes)}`]
        : [];
    });
    return result(
      op.lesson.title || "Compute each position",
      expanded.length
        ? `${expanded.join("; ")}. Result ${shape(output)}.`
        : op.lesson.summary || `Compute each position in ${shape(output)}.`,
    );
  }

  if (
    inputs.length === 1 &&
    outputs.length === 1 &&
    input &&
    output &&
    input.dtype === output.dtype &&
    input.numel === output.numel &&
    op.lesson.interaction === "mapping"
  ) {
    const order = op.lesson.axis_order;
    if (
      order &&
      order.length === input.shape.length &&
      new Set(order).size === order.length &&
      order.every(
        (axis) =>
          Number.isInteger(axis) && axis >= 0 && axis < input.shape.length,
      ) &&
      equal(
        order.map((axis) => input.shape[axis]),
        output.shape,
      )
    )
      return result(
        "Reorder the axes",
        `Output axes follow input axes [${order.join(", ")}]: ${shape(input)} → ${shape(output)}.`,
      );
    if (op.lesson.mapping_rule === "identity")
      return result(
        op.lesson.title || "Reshape the tensor",
        `Same ${count(input.numel, "element")} in the same logical order: ${shape(input)} → ${shape(output)}.`,
      );
  }
  if (outputs.length > 1)
    return result(
      op.lesson.title || op.kind,
      `${kindName(op.kind)} returns ${count(outputs.length, "recorded tensor")}. Each result has its own shape and values.`,
    );
  return result(
    op.lesson.title || op.kind,
    op.lesson.summary ||
      `${count(inputs.length, "recorded tensor input")} → ${count(outputs.length, "tensor result")}.`,
  );
}
