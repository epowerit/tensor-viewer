import { expect, test } from "vitest";
import type { Run } from "../api/client";
import { tensorInsights } from "./insights";
import { collectProblems } from "./problems";

type Spec = {
  shape: number[];
  dtype?: string;
  values?: (number | string)[];
  role?: string;
  storage?: string;
  axes?: string[];
  name?: string;
};
type Step = {
  kind: string;
  inputs: string[];
  outputs: string[];
  arguments?: object;
  mutations?: object[];
  status?: string;
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
