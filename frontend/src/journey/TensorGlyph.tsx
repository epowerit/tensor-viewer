import type { Tensor } from "../api/client";
import { TensorVolume } from "../tensors/TensorVolume";

/** Shares the indexed-cell geometry of the expanded inspector. */
export function TensorGlyph({
  tensor,
  onSelect,
}: {
  tensor: Tensor;
  onSelect?: (index: number) => void;
}) {
  return <TensorVolume tensor={tensor} compact onSelect={onSelect} />;
}
