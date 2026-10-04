import { expect, test } from "vitest";
import type { Run, Tensor } from "../api/client";
import { deadUnits, tensorInsights } from "./insights";
import { collectProblems } from "./problems";

type Spec = {
  shape: number[];
  dtype?: string;
  values?: (number | string)[];
  role?: string;
  storage?: string;
  axes?: string[];
  name?: string;
  minimum?: number;
  maximum?: number;
};
type Step = {
  kind: string;
  inputs: string[];
  outputs: string[];
  arguments?: object;
  mutations?: object[];
  status?: string;
  lesson?: object;
  loops?: object[];
};

function run(
  tensors: Record<string, Spec>,
  steps: Step[],
  options: { inputs?: string[]; outputs?: string[]; error?: object } = {},
): Run {
  return {
    project: { entry_path: "model.py" },
    trace: {
      input_ids: options.inputs ?? ["x"],
      output_ids: options.outputs ?? [steps.at(-1)?.outputs[0] ?? "x"],
      error: options.error ?? null,
      tensors: Object.fromEntries(
        Object.entries(tensors).map(([id, spec]) => [
          id,
          {
            id,
            name: spec.name ?? id,
            shape: spec.shape,
            numel: spec.shape.reduce((a, b) => a * b, 1),
            dtype: spec.dtype ?? "float32",
            values: spec.values ?? [],
            role: spec.role ?? "intermediate",
            storage_id: spec.storage ?? `s-${id}`,
            axes: spec.axes ?? spec.shape.map((_, i) => `axis ${i}`),
            minimum: spec.minimum ?? null,
            maximum: spec.maximum ?? null,
          },
        ]),
      ),
      operations: steps.map((step, index) => ({
        id: `op${index}`,
        index,
        status: step.status ?? "ok",
        arguments: step.arguments ?? {},
        mutations: step.mutations ?? [],
        source: { line: index + 1 },
        ...step,
      })),
    },
  } as unknown as Run;
}
const rules = (found: ReturnType<typeof tensorInsights>) =>
  found.map((item) => item.id.split("-").slice(1, -1).join("-"));

test("an outer combination is flagged; an ordinary broadcast is not", () => {
  const outer = run(
    {
      x: { shape: [3, 1], role: "input" },
      b: { shape: [3] },
      y: { shape: [3, 3] },
    },
    [{ kind: "sub", inputs: ["x", "b"], outputs: ["y"] }],
  );
  const found = tensorInsights(outer);
  expect(rules(found)).toEqual(["outer-broadcast"]);
  expect(found[0]).toMatchObject({ severity: "warning", node: "op0", line: 1 });
  expect(found[0].detail).toContain("x [3, 1] and b [3] combine into [3, 3]");
  // A bias added to every row stretches only one operand.
  const bias = run(
    {
      x: { shape: [4, 3], role: "input" },
      b: { shape: [3] },
      y: { shape: [4, 3] },
    },
    [{ kind: "add", inputs: ["x", "b"], outputs: ["y"] }],
  );
  expect(tensorInsights(bias)).toEqual([]);
  // Size-1 axes on both sides spell out the table on purpose.
  const table = run(
    {
      rows: { shape: [4, 1], role: "input" },
      columns: { shape: [1, 6] },
      y: { shape: [4, 6] },
    },
    [{ kind: "add", inputs: ["rows", "columns"], outputs: ["y"] }],
  );
  expect(tensorInsights(table)).toEqual([]);
});

test("an infinity written on purpose is not a non-finite warning", () => {
  const masked = (filled: (number | string)[]) =>
    run(
      {
        scores: { shape: [2], values: [1, 2], role: "input" },
        mask: { shape: [2], values: [0, 1], dtype: "bool" },
        y: { shape: [2], values: filled },
      },
      [
        {
          kind: "masked_fill",
          inputs: ["scores", "mask"],
          outputs: ["y"],
          arguments: { value: "-inf" },
        },
      ],
    );
  expect(tensorInsights(masked([1, "-inf"]))).toEqual([]);
  expect(rules(tensorInsights(masked(["nan", "-inf"])))).toEqual([
    "non-finite",
  ]);
});

test("the first non-finite value is located, not every step after it", () => {
  const trace = run(
    {
      x: { shape: [2], values: [1, 0], role: "input" },
      y: { shape: [2], values: [1, "inf"] },
      z: { shape: [2], values: [2, "inf"] },
    },
    [
      { kind: "div", inputs: ["x", "x"], outputs: ["y"] },
      { kind: "add", inputs: ["y", "y"], outputs: ["z"] },
    ],
  );
  const found = tensorInsights(trace);
  expect(rules(found)).toEqual(["non-finite"]);
  expect(found[0].node).toBe("op0");
  expect(found[0].detail).toContain("division by zero");
  // Without inline values nothing is claimed.
  const paged = run({ x: { shape: [2], role: "input" }, y: { shape: [2] } }, [
    { kind: "div", inputs: ["x", "x"], outputs: ["y"] },
  ]);
  expect(tensorInsights(paged)).toEqual([]);
});

test("reductions and softmax over a single position are noticed", () => {
  const trace = run(
    {
      x: { shape: [4, 1], role: "input" },
      s: { shape: [4] },
      w: { shape: [4, 1] },
      t: { shape: [4, 1] },
    },
    [
      { kind: "sum", inputs: ["x"], outputs: ["s"], arguments: { dim: -1 } },
      { kind: "softmax", inputs: ["x"], outputs: ["w"], arguments: { dim: 1 } },
      { kind: "mean", inputs: ["x"], outputs: ["t"], arguments: {} },
    ],
    { outputs: ["s", "w", "t"] },
  );
  const found = tensorInsights(trace);
  expect(rules(found)).toEqual(["trivial-reduction", "trivial-softmax"]);
  expect(found[1].detail).toContain("every weight is 1");
});

test("squeeze, dead activations, float64, and writes into the input", () => {
  const trace = run(
    {
      x: {
        shape: [1, 3],
        role: "input",
        axes: ["batch", "features"],
        values: [-1, -2, -3],
        storage: "input",
      },
      q: { shape: [3] },
      r: { shape: [1, 3], values: [0, 0, 0] },
      d: { shape: [1, 3], dtype: "float64" },
      e: { shape: [1, 3], dtype: "float64" },
      v: { shape: [3], storage: "input" },
      v2: { shape: [3], storage: "input" },
    },
    [
      { kind: "squeeze", inputs: ["x"], outputs: ["q"] },
      { kind: "relu", inputs: ["x"], outputs: ["r"] },
      { kind: "mul", inputs: ["r", "d"], outputs: ["e"] },
      {
        kind: "add_",
        inputs: ["v"],
        outputs: ["v2"],
        mutations: [{ before: "v", after: "v2", kind: "write" }],
      },
    ],
    { outputs: ["q", "e", "v2"] },
  );
  expect(rules(tensorInsights(trace))).toEqual([
    "squeezed-batch",
    "all-zero",
    "float64",
    "input-written",
  ]);
  const written = tensorInsights(trace).at(-1)!;
  expect(written.detail).toContain("storage that x shares (through v)");
  // squeeze(dim) says exactly what it removes.
  const named = run(
    {
      x: { shape: [1, 3], role: "input", axes: ["batch", "f"] },
      q: { shape: [3] },
    },
    [{ kind: "squeeze", inputs: ["x"], outputs: ["q"], arguments: { dim: 0 } }],
  );
  expect(tensorInsights(named)).toEqual([]);
});

test("results nobody reads are listed, but not after a failure", () => {
  const steps: Step[] = [
    { kind: "chunk", inputs: ["x"], outputs: ["a", "b"] },
    { kind: "relu", inputs: ["a"], outputs: ["y"] },
    { kind: "tanh", inputs: ["x"], outputs: ["unused"] },
  ];
  const tensors = {
    x: { shape: [4], role: "input" },
    a: { shape: [2] },
    b: { shape: [2] },
    y: { shape: [2] },
    unused: { shape: [4] },
  };
  const found = tensorInsights(run(tensors, steps, { outputs: ["y"] }));
  expect(found.map((item) => item.title)).toEqual([
    "Computed but never used: unused",
  ]);
  expect(found[0].detail).toContain("not recorded");
  expect(
    tensorInsights(
      run(tensors, steps, {
        outputs: [],
        error: { type: "RuntimeError", message: "x" },
      }),
    ),
  ).toEqual([]);
});

test("Run notes lint the shape check when it replaces a stale run", () => {
  const outer = run(
    {
      x: { shape: [3, 1], role: "input" },
      b: { shape: [3] },
      y: { shape: [3, 3] },
    },
    [{ kind: "mul", inputs: ["x", "b"], outputs: ["y"] }],
  );
  const clean = run({ x: { shape: [3], role: "input" } }, []);
  const notes = collectProblems(clean, { stale: true, check: outer });
  const insight = notes.find((item) => item.id.startsWith("insight-"))!;
  expect(insight.title).toBe("Shape check: Both operands were stretched");
  expect(insight.node).toBeNull();
  expect(
    collectProblems(outer, { insights: false }).some((item) =>
      item.id.startsWith("insight-"),
    ),
  ).toBe(false);
  expect(
    collectProblems(outer).filter((item) => item.id.startsWith("insight-")),
  ).toHaveLength(1);
});

test("softmax and reductions that mix the examples of a batch", () => {
  const batch = { shape: [4, 10], role: "input", axes: ["batch", "classes"] };
  const softmax = (dim: number) =>
    tensorInsights(
      run({ x: batch, p: { shape: [4, 10] } }, [
        { kind: "softmax", inputs: ["x"], outputs: ["p"], arguments: { dim } },
      ]),
    );
  const across = softmax(0);
  expect(rules(across)).toEqual(["batch-softmax"]);
  expect(across[0].detail).toContain("is x.batch");
  expect(rules(softmax(-1))).toEqual([]);
  // Lineage finds the batch axis after it moved: softmax(dim=-1) of x.T.
  const moved = tensorInsights(
    run({ x: batch, t: { shape: [10, 4] }, p: { shape: [10, 4] } }, [
      {
        kind: "permute",
        inputs: ["x"],
        outputs: ["t"],
        lesson: { axis_order: [1, 0] },
      },
      {
        kind: "softmax",
        inputs: ["t"],
        outputs: ["p"],
        arguments: { dim: -1 },
      },
    ]),
  );
  expect(rules(moved)).toEqual(["batch-softmax"]);

  const reduce = (dim: number | null, out: number[]) =>
    rules(
      tensorInsights(
        run({ x: batch, m: { shape: out } }, [
          { kind: "mean", inputs: ["x"], outputs: ["m"], arguments: { dim } },
        ]),
      ),
    );
  expect(reduce(0, [10])).toEqual(["batch-reduction"]);
  // Per-example results and whole-batch scalars (a mean loss) are ordinary.
  expect(reduce(1, [4])).toEqual([]);
  expect(reduce(null, [])).toEqual([]);
});

test("a value that grows every pass of a loop is flagged at the loop", () => {
  const pass = (iteration: number) => [
    {
      id: "/0:7",
      line: 7,
      file: null,
      text: "for block in self.blocks",
      iteration,
    },
  ];
  const looped = (peaks: number[]) =>
    run(
      Object.fromEntries([
        ["x", { shape: [2], role: "input" }],
        ...peaks.map((peak, i) => [
          `x${i + 1}`,
          { shape: [2], name: "x", minimum: -peak / 2, maximum: peak },
        ]),
      ]),
      peaks.map((_, i) => ({
        kind: "add",
        inputs: [i ? `x${i}` : "x"],
        outputs: [`x${i + 1}`],
        loops: pass(i + 1),
      })),
    );
  const found = tensorInsights(looped([1, 3, 9]));
  expect(rules(found)).toEqual(["loop-drift"]);
  expect(found[0]).toMatchObject({
    node: "op2",
    line: 7,
    title: "x grows every pass of for block in self.blocks",
  });
  expect(found[0].detail).toContain("1.00 → 3.00 → 9.00 over 3 passes (×9.00)");
  expect(tensorInsights(looped([8, 2, 0.5]))[0].title).toContain("shrinks");
  // Unsteady or modest change is not a drift.
  expect(tensorInsights(looped([1, 9, 3]))).toEqual([]);
  expect(tensorInsights(looped([1, 2, 3]))).toEqual([]);
});

test("features an activation leaves at zero everywhere are dead units", () => {
  // [batch 2, tokens 2, features 3]: feature 1 is zero at all four positions.
  const hidden = {
    id: "t1",
    name: "hidden",
    shape: [2, 2, 3],
    axes: ["batch", "tokens", "features"],
    numel: 12,
    values: [0.5, 0, 0, 0, 0, 1, 2, 0, 0, 0.1, 0, 0],
  } as unknown as Tensor;
  expect(deadUnits(hidden)).toEqual({
    axis: 2,
    size: 3,
    per: 4,
    units: [1],
  });
  // Channels are the feature axis of an image: [batch 1, channels 2, 2 × 1].
  const image = {
    ...hidden,
    shape: [1, 2, 2, 1],
    axes: ["batch", "channels", "height", "width"],
    numel: 4,
    values: [0, 0, 3, 0],
  } as unknown as Tensor;
  expect(deadUnits(image)?.units).toEqual([0]);
  expect(
    deadUnits({ ...hidden, values: hidden.values.map(() => 1) }),
  ).toBeNull();
});
