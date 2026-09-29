import type { Draft, Tensor } from "../api/client";
import { product } from "../tensors/coordinates";

export function blankProject(name: string): Draft {
  return {
    name,
    code: "from torch import nn\n\nclass ComposedModel(nn.Module):\n    def forward(self, x):\n        return x\n",
    class_name: "ComposedModel",
    constructor: {},
    capture_mode: "values",
    blueprint: { has_input: false, components: [] },
    input: {
      shape: [2, 4, 8],
      axis_names: ["batch", "tokens", "features"],
      generator: "arange",
      dtype: "float32",
      seed: 7,
      random_stream: "input",
    },
  };
}

/** Inferred shape previews explicitly have no values or recorded storage. */
export function previewTensor(
  id: string,
  name: string,
  shape: number[],
  axes: string[],
  dtype: string = "float32",
): Tensor {
  let stride = 1;
  const strides = shape.map(() => 0);
  for (let i = shape.length - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return {
    id,
    name,
    shape,
    axes,
    dtype,
    strides,
    storage_id: "preview",
    storage_offset: 0,
    contiguous: true,
    numel: product(shape),
    values: [],
    value_source: "shape",
    minimum: null,
    maximum: null,
    role: "input",
  };
}
