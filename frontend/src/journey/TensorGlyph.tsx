import type { Tensor } from "../api/client";
import { TensorVolume, type TensorLight } from "../tensors/TensorVolume";

/** Shares the indexed-cell geometry of the expanded inspector. */
export function TensorGlyph({
  tensor,
  selected,
  highlights,
  keyboardNavigation,
  onSelect,
  light,
}: {
  tensor: Tensor;
  light?: TensorLight;
  selected?: number;
  highlights?: readonly number[];
  keyboardNavigation?: boolean;
  onSelect?: (index: number) => void;
}) {
  return (
    <TensorVolume
      tensor={tensor}
      compact
      selected={selected}
      highlights={highlights}
      keyboardNavigation={keyboardNavigation}
      onSelect={onSelect}
      light={light}
    />
  );
}
