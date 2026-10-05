import type { Knockout, Run, Tensor } from "../api/client";

/** An axis's own name, or null for a numbered one. */
export const axisName = (tensor: Tensor, axis: number) => {
  const name = tensor.axes[axis];
  return name && !/^axis \d+$/.test(name) ? name : null;
};

/** The tensor a knockout replaces, in a run whose steps line up with it. */
export function knockedTensor(
  trace: Run["trace"],
  knockout: Knockout,
): Tensor | null {
  const op = trace.operations[knockout.step];
  const id = op?.outputs[knockout.output ?? 0];
  return (id && trace.tensors[id]) || null;
}

/**
 * A knockout in a few words, as Python would index it: `doubled = 0`,
 * `scores[:, 1] = mean over heads`, `h[:, :, 3] from the run before`.
 */
export function knockoutText(trace: Run["trace"], knockout: Knockout): string {
  const tensor = knockedTensor(trace, knockout);
  const name = tensor?.name ?? `step ${knockout.step + 1}`;
  const axis = knockout.axis ?? null;
  const target =
    axis === null
      ? name
      : `${name}[${":, ".repeat(axis)}${knockout.index ?? 0}]`;
  const over = axis !== null && tensor ? axisName(tensor, axis) : null;
  switch (knockout.mode) {
    case "mean":
      return axis === null
        ? `${target} = its mean`
        : `${target} = mean over ${over ?? `axis ${axis}`}`;
    case "patch":
      return `${target} from the run before`;
    default:
      return `${target} = 0`;
  }
}
