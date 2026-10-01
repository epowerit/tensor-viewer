import type { Operation, Run, Tensor } from "../api/client";
import { locatePart, tensorAssembly } from "../operations/assembly";
import { layoutMorph, relationMorph } from "../operations/layoutMorph";
import { tensorRelation } from "../operations/relations";
import { product } from "../tensors/coordinates";

/** Entire, verified mappings only: never imply that a sample is the full tensor. */
export const CELL_MOTION_LIMIT = 128;
export type CellMover = {
  sourceTensorId: string;
  sourceIndex: number;
  targetTensorId: string;
  targetIndex: number;
};
export type CellMotionPlan = {
  mode: "rearrange" | "select" | "collapse" | "assemble" | "partition" | "none";
  movers: CellMover[];
  caption: string;
};

const still = (caption: string): CellMotionPlan => ({
  mode: "none",
  movers: [],
  caption,
});
const unsupported = () =>
  still(
    "Showing the recorded tensors; cell motion is not available for this operation.",
  );
const large = () =>
  still(
    "Showing the full tensor shapes; open a tensor to inspect its cells. Whole-tensor motion is limited to small tensors.",
  );

/**
 * A scene-level map, in logical row-major indices rather than storage addresses.
 * No arithmetic is inferred: reductions show contributors meeting, while copy
 * rules show exact origins. Assembly includes every returned tensor, not just
 * whichever output happens to be visible on its node.
 */
export function cellMotionPlan(run: Run, op: Operation): CellMotionPlan {
  if (op.status !== "ok") return still("This operation did not complete.");
  if (op.mutations?.length || op.arguments.out != null)
    return still(
      "Showing the recorded write; cell motion is unavailable for mutations.",
    );
  if (!op.inputs.length || !op.outputs.length) return unsupported();
  // Bound metadata work as well as the eventual number of SVG movers. This also
  // excludes long sequences of empty operands without traversing them all.
  if (op.inputs.length + op.outputs.length > CELL_MOTION_LIMIT) return large();
  const tensors: Tensor[] = [];
  let cells = 0;
  for (const id of [...op.inputs, ...op.outputs]) {
    const tensor = run.trace.tensors[id];
    if (
      !tensor ||
      tensor.shape.length > 64 ||
      tensor.shape.some((n) => !Number.isSafeInteger(n) || n < 0) ||
      !Number.isSafeInteger(tensor.numel) ||
      tensor.numel < 0 ||
      product(tensor.shape) !== tensor.numel
    )
      return unsupported();
    cells += tensor.numel;
    if (tensor.numel > CELL_MOTION_LIMIT || cells > CELL_MOTION_LIMIT * 2)
      return large();
    tensors.push(tensor);
  }
  const inputs = tensors.slice(0, op.inputs.length);
  const outputs = tensors.slice(op.inputs.length);
  const assembly = tensorAssembly(run, op);
  if (assembly) {
    const movers = Array.from({ length: assembly.whole.numel }, (_, index) => {
      const local = locatePart(assembly, index);
      const part = assembly.parts[local.part];
      return assembly.joining
        ? {
            sourceTensorId: part.id,
            sourceIndex: local.index,
            targetTensorId: assembly.whole.id,
            targetIndex: index,
          }
        : {
            sourceTensorId: assembly.whole.id,
            sourceIndex: index,
            targetTensorId: part.id,
            targetIndex: local.index,
          };
    });
    return {
      mode: assembly.joining ? "assemble" : "partition",
      movers,
      caption: assembly.joining
        ? `Cells from each operand enter their recorded positions along axis ${assembly.axis}.`
        : `Cells separate into all ${assembly.parts.length} returned tensors along axis ${assembly.axis}.`,
    };
  }
  // Current relation/layout records describe one output. Applying that rule to
  // a tuple's first member would silently omit the other results.
  if (outputs.length !== 1) return unsupported();
  const output = outputs[0];
  const relation = tensorRelation(op, inputs, output);
  const supportedRelation =
    relation && ["index", "tile", "table", "reduce"].includes(relation.rule);
  if (relation && !supportedRelation) return unsupported();
  const input = relation ? inputs[relation.operand] : inputs[0];
  if (!input || !output.numel || !input.numel) return unsupported();
  // A selection copies values. A reduction can deliberately change dtype,
  // such as summing integer input into a larger integer accumulator.
  if (relation?.rule !== "reduce" && input.dtype !== output.dtype)
    return unsupported();
  const morph = relation
    ? relationMorph(op, relation, input, output)
    : layoutMorph(op, input, output);
  if (!morph || morph.movers.length > CELL_MOTION_LIMIT) return unsupported();
  const movers: CellMover[] = [];
  for (const mover of morph.movers) {
    if (
      !Number.isSafeInteger(mover.source) ||
      !Number.isSafeInteger(mover.target) ||
      mover.source < 0 ||
      mover.target < 0 ||
      mover.source >= input.numel ||
      mover.target >= output.numel
    )
      return unsupported();
    movers.push({
      sourceTensorId: input.id,
      sourceIndex: mover.source,
      targetTensorId: output.id,
      targetIndex: mover.target,
    });
  }
  return { mode: morph.kind, movers, caption: morph.caption };
}
