import type { Tensor } from "../api/client";

/** Resolve an explicitly chosen tensor before validating its local cell index. */
export function tensorSelection(
  tensors: Pick<Tensor, "id" | "numel">[],
  tensorId?: string,
  cell?: number,
) {
  const choice = Math.max(
    0,
    tensors.findIndex((tensor) => tensor.id === tensorId),
  );
  const count = tensors[choice]?.numel ?? 0;
  const index =
    cell !== undefined &&
    Number.isSafeInteger(cell) &&
    cell >= 0 &&
    cell < count
      ? cell
      : 0;
  return { choice, index };
}
