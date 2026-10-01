import type { Draft, Tensor } from "../api/client";

export const MAX_TOKENS = 64;
export const DEFAULT_SENTENCE = "the cat sat on the mat";

/** The same rule as the backend: lower-cased words and single symbols. */
export function tokenize(text: string) {
  // Python's Unicode whitespace differs from JavaScript's \s (notably NEL
  // and BOM). Unicode mode also keeps an astral symbol in a single token.
  const tokens = (
    text.match(
      /[A-Za-z0-9']+|[^\t-\r\u001c-\u0020\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000A-Za-z0-9']/gu,
    ) ?? []
  ).map((token) => token.toLowerCase());
  const vocabulary = [...new Set(tokens)].sort((a, b) => {
    // Python sorts Unicode code points, rather than UTF-16 code units.
    const left = Array.from(a, (c) => c.codePointAt(0)!);
    const right = Array.from(b, (c) => c.codePointAt(0)!);
    for (let i = 0; i < Math.min(left.length, right.length); i++)
      if (left[i] !== right[i]) return left[i] - right[i];
    return left.length - right.length;
  });
  return {
    tokens,
    vocabulary,
    ids: tokens.map((token) => vocabulary.indexOf(token)),
  };
}

type Input = Draft["input"];

/** Replace the sentence, keeping shape and dtype consistent with its tokens. */
export function withSentence(input: Input, text: string): Input {
  const count = Math.max(1, tokenize(text).tokens.length);
  return {
    ...input,
    generator: "text",
    text,
    uploaded: null,
    dtype: "int64",
    shape: [1, count],
    axis_names: ["batch", "tokens"],
  };
}

/** Switch the value source, adjusting whatever the new source fixes. */
export function withGenerator(
  input: Input,
  generator: Input["generator"],
): Input {
  if (generator === "text")
    return withSentence(input, input.text ?? DEFAULT_SENTENCE);
  const next: Input = { ...input, generator, uploaded: null, text: null };
  // Leaving a sentence: token ids are integers, most generators are not.
  if (input.generator === "text" && generator !== "arange")
    next.dtype = "float32";
  if (generator === "random") {
    next.random_stream = "input";
    if (next.dtype === "int64") next.dtype = "float32";
  }
  if (generator === "image") {
    if (next.dtype === "int64") next.dtype = "float32";
    if (next.shape.length < 3 || input.generator === "text") {
      next.shape = [1, 3, 16, 16];
      next.axis_names = ["batch", "channels", "height", "width"];
    } else if (
      !next.axis_names.length ||
      next.axis_names.length !== next.shape.length
    ) {
      // Name only the axes the picture defines.
      next.axis_names = next.shape.map((_, axis) => {
        const fromEnd = next.shape.length - axis;
        return fromEnd === 1
          ? "width"
          : fromEnd === 2
            ? "height"
            : fromEnd === 3
              ? "channels"
              : fromEnd === 4
                ? "batch"
                : `axis ${axis}`;
      });
    }
  }
  return next;
}

export type PixelPlan = {
  /** Flat indices per pixel, row-major: one for gray, three for color. */
  pixels: number[][];
  height: number;
  width: number;
  color: boolean;
  /** Fixed leading coordinates, including the channel when gray. */
  prefix: number[];
};

const named = (tensor: Tensor, fromEnd: number, pattern: RegExp) =>
  pattern.test(tensor.axes.at(-fromEnd) ?? "");

/**
 * Draw a tensor as a picture only when its axes are explicitly named height
 * and width. Sizes alone never make a tensor an image.
 */
export function pixelPlan(tensor: Tensor, index: number): PixelPlan | null {
  const rank = tensor.shape.length;
  if (
    rank < 2 ||
    !tensor.numel ||
    tensor.dtype === "bool" ||
    !named(tensor, 2, /^height$/i) ||
    !named(tensor, 1, /^width$/i)
  )
    return null;
  const height = tensor.shape[rank - 2],
    width = tensor.shape[rank - 1];
  if (height < 2 || width < 2 || height * width > 1024) return null;
  const coords: number[] = Array(rank).fill(0);
  let rest = Math.max(0, Math.min(index, tensor.numel - 1));
  for (let axis = rank - 1; axis >= 0; axis--) {
    coords[axis] = rest % tensor.shape[axis];
    rest = Math.floor(rest / tensor.shape[axis]);
  }
  const color =
    rank >= 3 &&
    tensor.shape[rank - 3] === 3 &&
    named(tensor, 3, /^channels?$/i);
  const prefix = coords.slice(0, color ? rank - 3 : rank - 2);
  const plane = height * width;
  let base = 0;
  prefix.forEach((coordinate, axis) => {
    base = base * tensor.shape[axis] + coordinate;
  });
  const pixels = Array.from({ length: plane }, (_, p) =>
    color
      ? [0, 1, 2].map((channel) => (base * 3 + channel) * plane + p)
      : [base * plane + p],
  );
  return { pixels, height, width, color, prefix };
}

/** Map recorded values onto 0–255 using the displayed plane's own range. */
export function pixelColors(
  plan: PixelPlan,
  valueAt: (index: number) => number | string | undefined,
): { colors: (string | null)[]; minimum: number; maximum: number } | null {
  let minimum = Infinity,
    maximum = -Infinity;
  const values = plan.pixels.map((indices) =>
    indices.map((index) => {
      const value = valueAt(index);
      if (typeof value !== "number" || !Number.isFinite(value)) return null;
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
      return value;
    }),
  );
  if (minimum === Infinity) return null;
  // Scale before subtracting so finite opposite extremes do not overflow.
  const scale = Math.max(Math.abs(minimum), Math.abs(maximum)) || 1;
  const low = minimum / scale;
  const span = maximum / scale - low || 1;
  const level = (value: number) =>
    Math.max(
      0,
      Math.min(255, Math.round(((value / scale - low) / span) * 255)),
    );
  return {
    minimum,
    maximum,
    colors: values.map((channels) => {
      if (channels.some((value) => value === null)) return null;
      const [r, g, b] = plan.color
        ? (channels as number[]).map(level)
        : Array(3).fill(level(channels[0] as number));
      return `rgb(${r}, ${g}, ${b})`;
    }),
  };
}
