import { nearestWindowSample } from "../tensors/windowGeometry";
export { nearestWindowSample as nearestConvolutionSample } from "../tensors/windowGeometry";
import type { Operation, Run, Tensor } from "../api/client";
import { product, ravel, unravel } from "../tensors/coordinates";

export type Convolution = {
  input: Tensor;
  weight: Tensor;
  bias?: Tensor;
  output: Tensor;
  dimensions: number;
  channelAxis: number;
  groups: number;
  channels: number;
  features: number;
  kernel: number[];
  stride: number[];
  dilation: number[];
  before: number[];
  after: number[];
  terms: number;
};
const integer = (n: unknown, min: number): n is number =>
  typeof n === "number" && Number.isSafeInteger(n) && n >= min;
function spatial(value: unknown, dimensions: number, minimum: number) {
  const values =
    typeof value === "number" ? Array(dimensions).fill(value) : value;
  return Array.isArray(values) &&
    values.length === dimensions &&
    values.every((n) => integer(n, minimum))
    ? (values as number[])
    : null;
}

/** Prove geometry from captured operands; never infer neighborhoods from shape alone. */
export function tensorConvolution(run: Run, op: Operation): Convolution | null {
  if (
    !["conv1d", "conv2d"].includes(op.kind) ||
    op.status !== "ok" ||
    op.mutations?.length ||
    op.outputs.length !== 1 ||
    ![2, 3].includes(op.inputs.length)
  )
    return null;
  const [input, weight, bias] = op.inputs.map((id) => run.trace.tensors[id]);
  const output = run.trace.tensors[op.outputs[0]];
  const dimensions = op.kind === "conv1d" ? 1 : 2;
  if (
    !input ||
    !weight ||
    !output ||
    (op.inputs.length === 3 && !bias) ||
    ![dimensions + 1, dimensions + 2].includes(input.shape.length) ||
    weight.shape.length !== dimensions + 2 ||
    output.shape.length !== input.shape.length ||
    !["float16", "bfloat16", "float32", "float64"].includes(input.dtype) ||
    [input, weight, output, ...(bias ? [bias] : [])].some(
      (t) =>
        t.dtype !== input.dtype ||
        !integer(t.numel, 1) ||
        t.shape.some((n) => !integer(n, 1)) ||
        product(t.shape) !== t.numel,
    )
  )
    return null;
  // Older Conv1D traces used positional arg3–arg6; Conv2D already named them.
  const arg = (key: string, position: number, fallback: unknown) =>
    Object.hasOwn(op.arguments, key)
      ? op.arguments[key]
      : (op.arguments[`arg${position}`] ?? fallback);
  const channelAxis = input.shape.length - dimensions - 1;
  const groups = arg("groups", 6, 1);
  if (!integer(groups, 1)) return null;
  const channels = input.shape[channelAxis] / groups;
  const features = weight.shape[0] / groups;
  if (
    !integer(channels, 1) ||
    !integer(features, 1) ||
    weight.shape[1] !== channels ||
    (bias && (bias.shape.length !== 1 || bias.shape[0] !== weight.shape[0]))
  )
    return null;
  const kernel = weight.shape.slice(2);
  const stride = spatial(arg("stride", 3, 1), dimensions, 1);
  const dilation = spatial(arg("dilation", 5, 1), dimensions, 1);
  if (!stride || !dilation) return null;
  const effective = kernel.map((k, i) => (k - 1) * dilation[i] + 1);
  if (effective.some((n) => !integer(n, 1))) return null;
  const padding = arg("padding", 4, 0);
  let before: number[], after: number[];
  if (padding === "valid") before = after = Array(dimensions).fill(0);
  else if (padding === "same" && stride.every((s) => s === 1)) {
    before = effective.map((k) => Math.floor((k - 1) / 2));
    after = effective.map((k, i) => k - 1 - before[i]);
  } else {
    const pad = spatial(padding, dimensions, 0);
    if (!pad) return null;
    before = after = pad;
  }
  const sizes = input.shape
    .slice(-dimensions)
    .map((n, i) => n + before[i] + after[i]);
  if (sizes.some((n) => !integer(n, 1))) return null;
  const expected = [
    ...input.shape.slice(0, channelAxis),
    weight.shape[0],
    ...sizes.map((n, i) => Math.floor((n - effective[i]) / stride[i]) + 1),
  ];
  if (expected.some((n, i) => output.shape[i] !== n)) return null;
  return {
    input,
    weight,
    bias,
    output,
    dimensions,
    channelAxis,
    groups,
    channels,
    features,
    kernel,
    stride,
    dilation,
    before,
    after,
    terms: product(weight.shape.slice(1)),
  };
}

export function convolutionSelection(
  p: Convolution,
  output: number,
  term: number,
) {
  const coordinates = unravel(output, p.output.shape);
  const feature = coordinates[p.channelAxis];
  const group = Math.floor(feature / p.features);
  const kernelCoordinates = unravel(term, [p.channels, ...p.kernel]);
  const channel = group * p.channels + kernelCoordinates[0];
  const position = coordinates
    .slice(-p.dimensions)
    .map(
      (o, i) =>
        o * p.stride[i] -
        p.before[i] +
        kernelCoordinates[i + 1] * p.dilation[i],
    );
  const inputCoordinates = [
    ...coordinates.slice(0, p.channelAxis),
    channel,
    ...position,
  ];
  const weightCoordinates = [feature, ...kernelCoordinates];
  const padding = position.some(
    (v, i) => v < 0 || v >= p.input.shape[p.channelAxis + 1 + i],
  );
  return {
    output,
    term,
    coordinates,
    feature,
    group,
    channel,
    inputCoordinates,
    weightCoordinates,
    padding,
    input: padding ? null : ravel(inputCoordinates, p.input.shape),
    weight: ravel(weightCoordinates, p.weight.shape),
  };
}
export function convolutionWindow(
  p: Convolution,
  output: number,
  term: number,
) {
  const start = Math.floor(term / 8) * 8;
  return Array.from({ length: Math.min(8, p.terms - start) }, (_, i) =>
    convolutionSelection(p, output, start + i),
  );
}
export function convolutionWeightSelection(
  p: Convolution,
  weight: number,
  output: number,
) {
  const coordinates = unravel(weight, p.weight.shape);
  const target = unravel(output, p.output.shape);
  target[p.channelAxis] = coordinates[0];
  return {
    output: ravel(target, p.output.shape),
    term: ravel(coordinates.slice(1), p.weight.shape.slice(1)),
  };
}

export function convolutionInputSelection(
  p: Convolution,
  input: number,
  output: number,
) {
  const source = unravel(input, p.input.shape),
    target = unravel(output, p.output.shape);
  const channel = source[p.channelAxis];
  const feature =
    Math.floor(channel / p.channels) * p.features +
    (target[p.channelAxis] % p.features);
  const position: number[] = [],
    kernel: number[] = [];
  for (let i = 0; i < p.dimensions; i++) {
    const axis = p.channelAxis + 1 + i;
    const sample = nearestWindowSample(
      source[axis],
      p.before[i],
      p.stride[i],
      p.dilation[i],
      p.kernel[i],
      p.output.shape[axis],
      target[axis],
    );
    if (!sample) return null;
    position.push(sample.output);
    kernel.push(sample.kernel);
  }
  return {
    output: ravel(
      [...source.slice(0, p.channelAxis), feature, ...position],
      p.output.shape,
    ),
    term: ravel([channel % p.channels, ...kernel], [p.channels, ...p.kernel]),
  };
}
