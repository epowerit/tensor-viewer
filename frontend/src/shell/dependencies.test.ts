import { expect, test } from "vitest";
import type { Dependencies, Run } from "../api/client";
import {
  allowed,
  askableAxes,
  causalEvidence,
  defaultAsk,
  dependencyNotes,
  expectedPattern,
  inputAxisFor,
  isTriangularMap,
  isTriangularMask,
  readList,
  violationsOf,
  type DependencyAsk,
} from "./dependencies";

type Spec = {
  shape: number[];
  dtype?: string;
  values?: (number | string)[];
  role?: string;
  axes?: string[];
  name?: string;
};

function tensor(id: string, spec: Spec) {
  return {
    id,
    name: spec.name ?? id,
    shape: spec.shape,
    numel: spec.shape.reduce((a, b) => a * b, 1),
    dtype: spec.dtype ?? "float32",
    values: spec.values ?? [],
    value_source: "inline",
    role: spec.role ?? "intermediate",
    storage_id: `s-${id}`,
    storage_offset: 0,
    strides: [],
    contiguous: true,
    axes: spec.axes ?? spec.shape.map((_, i) => `axis ${i}`),
  };
}

function run(
  tensors: Record<string, Spec>,
  steps: { kind: string; inputs: string[]; outputs: string[] }[],
  options: { buffers?: string[]; inputs?: string[] } = {},
): Run {
  return {
    project: { entry_path: "model.py", input: {} },
    trace: {
      input_ids: options.inputs ?? ["tokens"],
      output_ids: [steps.at(-1)?.outputs[0]],
      buffer_names: options.buffers ?? [],
      tensors: Object.fromEntries(
        Object.entries(tensors).map(([id, spec]) => [id, tensor(id, spec)]),
      ),
      operations: steps.map((step, index) => ({
        id: `op${index}`,
        index,
        status: "ok",
        function: step.kind,
        arguments: {},
        mutations: [],
        module: "",
        source: { line: 10 + index, text: "" },
        ...step,
      })),
    },
  } as unknown as Run;
}

const square = <T>(n: number, at: (row: number, column: number) => T) =>
  Array.from({ length: n * n }, (_, i) => at(Math.floor(i / n), i % n));
const tril = square(4, (r, c) => (c <= r ? 1 : 0));
const causalWeights = square(4, (r, c) => (c <= r ? 1 / (r + 1) : 0));

/** A small GPT: embeddings, causal attention, and a softmax over words. */
function gpt(options: { mask?: boolean; weights?: number[] } = {}) {
  return run(
    {
      tokens: {
        shape: [1, 4],
        dtype: "int64",
        role: "input",
        axes: ["batch", "tokens"],
      },
      causal: {
        shape: [4, 4],
        dtype: "bool",
        role: "parameter",
        name: "blocks.0.causal",
        values: tril,
      },
      table: { shape: [5, 8], role: "parameter", name: "token.weight" },
      embedding: { shape: [1, 4, 8], axes: ["batch", "tokens", "features"] },
      weights: {
        shape: [1, 4, 4],
        axes: ["batch", "queries", "keys"],
        values: options.weights ?? causalWeights,
      },
      mixed: { shape: [1, 4, 8], axes: ["batch", "tokens", "features"] },
      probabilities: {
        shape: [1, 4, 5],
        axes: ["batch", "tokens", "vocabulary"],
      },
    },
    [
      {
        kind: "embedding",
        inputs: ["tokens", "table"],
        outputs: ["embedding"],
      },
      { kind: "softmax", inputs: ["embedding"], outputs: ["weights"] },
      { kind: "matmul", inputs: ["weights", "embedding"], outputs: ["mixed"] },
      { kind: "softmax", inputs: ["mixed"], outputs: ["probabilities"] },
    ],
    { buffers: options.mask === false ? [] : ["blocks.0.causal"] },
  );
}

test("a mask that sets the future apart is triangular, whatever its values", () => {
  const t = (values: (number | string)[]) =>
    tensor("m", { shape: [4, 4], values }) as never;
  expect(isTriangularMask(t(tril))).toBe(true);
  // ~tril, as masked_fill takes it, and an additive mask of −∞ and 0.
  expect(isTriangularMask(t(tril.map((v) => 1 - v)))).toBe(true);
  expect(isTriangularMask(t(square(4, (r, c) => (c > r ? "-inf" : 0))))).toBe(
    true,
  );
  expect(isTriangularMask(t(square(4, () => 1)))).toBe(false);
  // Off by one: each position also sees the next.
  expect(isTriangularMask(t(square(4, (r, c) => (c <= r + 1 ? 1 : 0))))).toBe(
    false,
  );
  expect(
    isTriangularMask(
      tensor("m", { shape: [2, 3], values: [1, 0, 0, 1, 1, 0] }) as never,
    ),
  ).toBe(false);
});

test("attention weights are triangular when no query weighs a later key", () => {
  const t = (values: number[]) =>
    tensor("w", { shape: [1, 4, 4], values }) as never;
  expect(isTriangularMap(t(causalWeights))).toBe(true);
  expect(isTriangularMap(t(square(4, () => 0.25)))).toBe(false);
  // Each reading only itself proves nothing about the past.
  expect(isTriangularMap(t(square(4, (r, c) => (r === c ? 1 : 0))))).toBe(
    false,
  );
});

test("causal evidence comes from mask buffers, then attention maps", () => {
  const found = causalEvidence(gpt().trace);
  expect(found.map((each) => each.tensorId)).toEqual(["causal", "weights"]);
  expect(found[0].label).toBe("the causal mask blocks.0.causal");
  expect(found[0].size).toBeNull();
  expect(found[1].size).toBe(4);
  expect(found[1].label).toMatch(/^the attention weights weights at step 2/);
  // Without the mask and with full attention there is none.
  expect(
    causalEvidence(gpt({ mask: false, weights: square(4, () => 0.25) }).trace),
  ).toEqual([]);
});

test("a result's axis is matched to the input axis by name, lineage or size", () => {
  const { trace } = gpt();
  const probabilities = trace.tensors.probabilities;
  const none = () => [];
  expect(inputAxisFor(trace, probabilities, 0, none)).toBe(0);
  expect(inputAxisFor(trace, probabilities, 1, none)).toBe(1);
  expect(inputAxisFor(trace, probabilities, 2, none)).toBeNull();
  const unnamed = { ...probabilities, axes: ["axis 0", "axis 1", "axis 2"] };
  // Size 4 is the input's only axis of that size; size 1 is too.
  expect(inputAxisFor(trace, unnamed, 1, none)).toBe(1);
  expect(
    inputAxisFor(trace, unnamed, 1, () => [
      { terms: [] },
      { terms: [{ label: "tokens.batch" }] },
    ]),
  ).toBe(0);
});

test("a batch keeps to the diagonal, causal tokens to the lower triangle", () => {
  const { trace } = gpt();
  const probabilities = trace.tensors.probabilities;
  expect(expectedPattern(trace, probabilities, 0, 0)?.pattern).toBe("diagonal");
  const causal = expectedPattern(trace, probabilities, 1, 1);
  expect(causal?.pattern).toBe("causal");
  expect(causal?.why).toBe(
    "the causal mask blocks.0.causal lets each position read only itself and those before it",
  );
  // Without any sign of causal attention, tokens may read every token.
  const open = gpt({ mask: false, weights: square(4, () => 0.25) }).trace;
  expect(expectedPattern(open, open.tensors.probabilities, 1, 1)).toBeNull();
  // Attention maps alone count, for the positions they cover.
  const maps = gpt({ mask: false }).trace;
  expect(expectedPattern(maps, maps.tensors.probabilities, 1, 1)?.why).toMatch(
    /^the attention weights weights/,
  );
  const longer = {
    ...maps.tensors.probabilities,
    shape: [1, 6, 5],
    axes: ["batch", "tokens", "vocabulary"],
  };
  expect(expectedPattern(maps, longer, 1, 1)).toBeNull();
});

test("violations are the cells read outside the pattern", () => {
  const lower = [0, 1, 2].map((r) => [0, 1, 2].map((c) => c <= r));
  const all = [0, 1, 2].map(() => [true, true, true]);
  expect(violationsOf(lower, "causal")).toEqual([]);
  expect(violationsOf(all, "causal")).toEqual([
    [0, 1],
    [0, 2],
    [1, 2],
  ]);
  expect(violationsOf(lower, "diagonal")).toEqual([
    [1, 0],
    [2, 0],
    [2, 1],
  ]);
  expect(violationsOf(all, null)).toEqual([]);
  expect(allowed("causal", 2, 2)).toBe(true);
  expect(allowed("causal", 2, 3)).toBe(false);
  expect(allowed("diagonal", 1, 1)).toBe(true);
});

test("the first question is the latest result on its causal axis", () => {
  const { trace } = gpt();
  expect(defaultAsk(trace)).toMatchObject({
    tensor_id: "probabilities",
    axis: 1,
    input_axis: 1,
    expect: "causal",
  });
  // The batch axis is offered too, with its own pattern.
  expect(
    askableAxes(trace, trace.tensors.probabilities).map((ask) => [
      ask.axis,
      ask.expect,
    ]),
  ).toEqual([
    [0, "diagonal"],
    [1, "causal"],
  ]);
  // Parameters and inputs are not results.
  expect(askableAxes(trace, trace.tensors.table)).toEqual([]);
});

const answer = (reads: boolean[][], extra: Partial<Dependencies> = {}) =>
  ({
    tensor_id: "probabilities",
    axis: 1,
    input_id: "tokens",
    input_axis: 1,
    kind: "embedding",
    size: reads.length,
    threshold: 1e-6,
    strength: reads.map((row) => row.map((read) => (read ? 1 : 0))),
    reads,
    zero: reads.map((row) => row.map((read) => !read)),
    expect: "causal",
    violations: [],
    breach: null,
    ...extra,
  }) as Dependencies;

const causalAsk: DependencyAsk = {
  tensor_id: "probabilities",
  axis: 1,
  input_axis: 1,
  expect: "causal",
  why: "the causal mask blocks.0.causal lets each position read only itself and those before it",
};

test("a breach becomes a note pinned to the first step that reads it", () => {
  const r = gpt();
  const lower = [0, 1, 2, 3].map((o) => [0, 1, 2, 3].map((i) => i <= o));
  expect(dependencyNotes(r, causalAsk, answer(lower))).toEqual([]);
  const all = [0, 1, 2, 3].map(() => [true, true, true, true]);
  const [note] = dependencyNotes(
    r,
    causalAsk,
    answer(all, {
      breach: {
        output: 0,
        input: 1,
        operation_id: "op2",
        step: 2,
        tensor_id: "mixed",
        axis: 1,
        after: "op0",
      },
    }),
    ["once", "upon", "a", "time"],
  );
  expect(note.title).toBe("Position 0 (“once”) reads position 1 (“upon”)");
  expect(note.severity).toBe("warning");
  expect(note.node).toBe("op2");
  expect(note.line).toBe(12);
  expect(note.file).toBe("model.py");
  expect(note.detail).toBe(
    "probabilities [1, 4, 5] should read each position along tokens only from itself and the positions before it, since the causal mask blocks.0.causal lets each position read only itself and those before it. Gradients show 6 cells read a later position: 0 reads 1, 2, 3; 1 reads 2, 3; 2 reads 3. The first step whose result reads it is step 3, matmul (line 12); step 1, embedding, before it still kept the pattern.",
  );
});

test("examples that read each other are named as examples", () => {
  const r = gpt();
  const [note] = dependencyNotes(
    r,
    { ...causalAsk, axis: 0, input_axis: 0, expect: "diagonal", why: null },
    answer(
      [
        [true, true],
        [false, true],
      ],
      { axis: 0, input_axis: 0, expect: "diagonal" },
    ),
    ["once", "upon"],
  );
  expect(note.title).toBe("Example 0 reads example 1");
  expect(note.node).toBeNull();
  expect(note.detail).toMatch(
    /^probabilities \[1, 4, 5\] should read each example along batch only from itself\. Gradients show 1 cell reads another example: 0 reads 1\.$/,
  );
});

test("a list of reads names the first few positions", () => {
  expect(readList([[0, 1]])).toBe("0 reads 1");
  expect(
    readList([
      [0, 1],
      [0, 2],
      [0, 3],
      [0, 4],
      [0, 5],
      [1, 2],
      [2, 3],
      [3, 4],
    ]),
  ).toBe("0 reads 1, 2, 3, 4, …; 1 reads 2; 2 reads 3; …");
});
