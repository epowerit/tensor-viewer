import { describe, expect, it } from "vitest";
import type { Operation, Run, Tensor } from "../api/client";
import {
  carryFolds,
  carryOver,
  describeChange,
  matchRuns,
  matchStages,
  renamedTensors,
  type LeftOff,
} from "./reload";
import type { DetailLevel, JourneyStage } from "./stages";

type Step = [kind: string, text: string, shape?: number[], module?: string];

function trace(steps: Step[]): Run["trace"] {
  const tensors: Record<string, Tensor> = {};
  const operations = steps.map(([kind, text, shape = [2], module = "M"], i) => {
    tensors[`t${i}`] = { id: `t${i}`, dtype: "float32", shape } as Tensor;
    return {
      id: `op${i}`,
      index: i,
      kind,
      function: `torch.${kind}`,
      module,
      inputs: [],
      outputs: [`t${i}`],
      source: { line: i + 1, text, file: null },
    } as unknown as Operation;
  });
  return { operations, tensors, input_ids: [] } as unknown as Run["trace"];
}

const codeOf = (steps: Step[]) => steps.map(([, text]) => text).join("\n");
const projectOf = (code: string) => ({
  name: "p",
  code,
  class_name: "M",
  constructor: {},
  input: {},
  files: {},
});

const gpt: Step[] = [
  ["embedding", "x = self.token(tokens)"],
  ["linear", "q = self.q(x)"],
  ["layer_norm", "x = self.norm(x)"],
  ["matmul", "logits = x @ self.token.weight.T"],
  ["softmax", "return logits.softmax(dim=-1)"],
];

describe("matchRuns", () => {
  it("finds a step added at the end, keeping an edited line as the same step", () => {
    const match = matchRuns(
      trace(gpt),
      trace([
        ...gpt.slice(0, 4),
        ["softmax", "logits = logits.softmax(dim=-1)"],
        ["argmax", "return logits.argmax(dim=-1)", [1]],
      ]),
    );
    expect(match.added).toEqual(["op5"]);
    expect(match.removed).toEqual([]);
    expect(match.edited).toEqual(["op4"]);
    expect(match.operations.get("op4")).toBe("op4");
  });

  it("follows steps whose ids shift when a step is inserted above them", () => {
    const match = matchRuns(
      trace(gpt),
      trace([gpt[0], ["dropout", "x = self.drop(x)"], ...gpt.slice(1)]),
    );
    expect(match.added).toEqual(["op1"]);
    expect(match.operations.get("op1")).toBe("op2");
    expect(match.operations.get("op4")).toBe("op5");
    expect(match.reshaped).toEqual([]);
  });

  it("reports removed steps and results that changed shape", () => {
    const match = matchRuns(
      trace(gpt),
      trace([["embedding", "x = self.token(tokens)", [4]], ...gpt.slice(2)]),
    );
    expect(match.removed).toEqual(["op1"]);
    expect(match.reshaped).toEqual(["op0"]);
    expect(match.operations.get("op2")).toBe("op1");
  });
});

describe("describeChange", () => {
  it("names a few added steps and counts the rest", () => {
    const after = trace([...gpt, ["argmax", "return logits.argmax(-1)"]]);
    expect(describeChange(matchRuns(trace(gpt), after), after)).toBe(
      "argmax added",
    );
    const kept = trace([
      ...gpt.slice(0, 4),
      ["softmax", "return logits.softmax(dim=-1)", [1, 2]],
    ]);
    expect(describeChange(matchRuns(trace(gpt), kept), kept)).toBe(
      "softmax now [1, 2]",
    );
    const shorter = trace(gpt.slice(0, 4));
    expect(
      describeChange(matchRuns(trace(gpt), shorter), shorter, trace(gpt)),
    ).toBe("softmax removed");
    const tokens = (shape: number[]) => {
      const run = trace(gpt);
      run.input_ids = ["in"];
      run.tensors.in = { id: "in", name: "tokens", shape } as Tensor;
      return run;
    };
    const longer = tokens([1, 4]);
    expect(
      describeChange(
        matchRuns(tokens([1, 13]), longer),
        longer,
        tokens([1, 13]),
      ),
    ).toBe("tokens now [1, 4]");
    const same = trace(gpt);
    expect(describeChange(matchRuns(trace(gpt), same), same)).toBe(
      "same steps, values updated",
    );
  });
});

const stage = (
  id: string,
  path: string,
  operationIds: string[],
  layout = false,
) =>
  ({
    id,
    path,
    title: path,
    operationIds,
    ...(layout ? { layout: { from: [1], to: [1] } } : {}),
  }) as unknown as JourneyStage;

describe("matchStages and carryFolds", () => {
  const before = trace(gpt);
  const after = trace([
    gpt[0],
    ["dropout", "x = self.drop(x)"],
    ...gpt.slice(1),
  ]);
  const match = matchRuns(before, after);
  const old = [stage("stage-call1", "block", ["op1", "op2"])];
  const next = [
    stage("stage-call1", "drop", ["op1"]),
    stage("stage-call2", "block", ["op2", "op3"]),
  ];

  it("maps a call to the call holding its steps, whatever its id now", () => {
    expect(matchStages(old, next, match).get("stage-call1")).toBe(
      "stage-call2",
    );
  });

  it("keeps hand folds on their stages and dial settings as settings", () => {
    const stages = matchStages(old, next, match);
    expect(carryFolds(new Set(["stage-call1"]), [], [], stages)).toEqual(
      new Set(["stage-call2"]),
    );
    const level = (label: string, ids: string[]): DetailLevel => ({
      label,
      detail: "",
      collapsed: new Set(ids),
    });
    expect(
      carryFolds(
        new Set(["stage-call1"]),
        [level("blocks", ["stage-call1"])],
        [level("blocks", ["stage-call1", "stage-call2"])],
        stages,
      ),
    ).toEqual(new Set(["stage-call1", "stage-call2"]));
  });
});

describe("carryOver", () => {
  const runOf = (id: string, steps: Step[], code = codeOf(steps)) =>
    ({
      id,
      project_id: "p",
      project: projectOf(code),
      trace: { ...trace(steps), module_calls: [] },
    }) as unknown as Run;
  const left = (run: Run, selected: string): LeftOff => ({
    run,
    stages: [],
    levels: [],
    collapsed: new Set(),
    selected,
    focused: true,
    inspector: false,
    inspectorView: "code",
    expanded: true,
    selection: { nodeId: selected, tensorId: `t${selected.slice(2)}`, cell: 1 },
    playback: { playing: true, speed: 2, following: true, reveal: false },
    lens: null,
    canvas: { current: null },
  });

  it("moves a removed step's selection to the next step still there", () => {
    const before = runOf("a", gpt);
    const after = runOf("b", [...gpt.slice(0, 2), ...gpt.slice(3)]);
    const carried = carryOver(left(before, "op2"), after);
    expect(carried?.selected).toBe("op2");
    expect(carried?.expanded).toBe(false);
    expect(carried?.summary).toBe("layer_norm removed");
  });

  it("keeps the chosen cell while the tensor keeps its shape", () => {
    const before = runOf("a", gpt);
    const after = runOf("b", [["dropout", "x = self.drop(x)"], ...gpt]);
    const carried = carryOver(left(before, "op3"), after);
    expect(carried?.selected).toBe("op4");
    expect(carried?.selection).toEqual({
      nodeId: "op4",
      tensorId: "t4",
      cell: 1,
    });
    expect(carried?.expanded).toBe(true);
    expect(carried?.playback).toEqual({
      playing: true,
      speed: 2,
      following: true,
      reveal: false,
    });
  });
});

describe("matchRuns at scale", () => {
  it("lines up a 50,000-step run with three steps inserted midway", () => {
    const kinds = ["linear", "gelu", "add", "layer_norm", "matmul", "softmax"];
    const steps: Step[] = Array.from({ length: 50_000 }, (_, i) => [
      kinds[i % kinds.length],
      `x = layer${i % 300}(x)`,
    ]);
    const edited: Step[] = [
      ...steps.slice(0, 25_000),
      ["dropout", "x = self.drop(x)"],
      ["dropout", "x = self.drop(x)"],
      ["dropout", "x = self.drop(x)"],
      ...steps.slice(25_000),
    ];
    const match = matchRuns(trace(steps), trace(edited));
    expect(match.added).toEqual(["op25000", "op25001", "op25002"]);
    expect(match.removed).toEqual([]);
    expect(match.operations.get("op49999")).toBe("op50002");
  });
});

describe("renamedTensors", () => {
  it("follows a variable the code renamed, and only that", () => {
    const named = (steps: Step[], names: string[]) => {
      const run = trace(steps);
      names.forEach((name, i) => (run.tensors[`t${i}`].name = name));
      return run;
    };
    const before = named(gpt, ["x", "q", "x", "logits", "softmax"]);
    const after = named(
      [
        ...gpt.slice(0, 3),
        ["matmul", "scores = x @ self.token.weight.T"],
        gpt[4],
      ],
      ["x", "q", "x", "scores", "softmax"],
    );
    const match = matchRuns(before, after);
    expect([...renamedTensors(match, before, after)]).toEqual([
      ["logits", "scores"],
    ]);
    expect(describeChange(match, after, before)).toBe("logits → scores");
  });
});

describe("matchRuns past alignment size", () => {
  it("pairs every step at most once when matched by occurrence", () => {
    const kinds = ["linear", "gelu", "add"];
    const steps: Step[] = Array.from({ length: 2_200 }, (_, i) => [
      kinds[i % 3],
      `x = f${i % 7}(x)`,
    ]);
    // Reversed: nothing lines up at either end, and the middle is too
    // large to align, so steps pair by occurrence.
    const match = matchRuns(trace(steps), trace([...steps].reverse()));
    const targets = [...match.operations.values()];
    expect(new Set(targets).size).toBe(targets.length);
    expect(new Set(match.edited).size).toBe(match.edited.length);
    expect(match.added.length + targets.length).toBe(2_200);
  });
});

describe("carryOver of a folded loop", () => {
  it("keeps the loop selected when lines above its header move it", () => {
    const looped = (id: string, header: number, shift: number) => {
      const steps = trace(gpt);
      steps.operations.forEach((op) => {
        op.loops = [
          {
            id: `/0:${header}`,
            line: header,
            file: null,
            text: "for b in self.blocks",
            iteration: 1,
          },
        ] as Operation["loops"];
        op.source = { ...op.source!, line: op.source!.line + shift };
      });
      return {
        id,
        project_id: "p",
        project: projectOf(codeOf(gpt)),
        trace: { ...steps, module_calls: [] },
      } as unknown as Run;
    };
    const left = {
      run: looped("a", 64, 0),
      stages: [],
      levels: [],
      collapsed: new Set<string>(),
      selected: "loop:/0:64",
      focused: false,
      inspector: false,
      inspectorView: "code",
      expanded: false,
      selection: null,
      playback: { playing: false, speed: 1, following: true, reveal: false },
      lens: null,
      canvas: { current: null },
    } as LeftOff;
    expect(carryOver(left, looped("b", 65, 1))?.selected).toBe("loop:/0:65");
  });
});

describe("what a save says when no value can change", () => {
  const left = (run: Run): LeftOff => ({
    run,
    stages: [],
    levels: [],
    collapsed: new Set(),
    selected: null,
    focused: false,
    inspector: false,
    inspectorView: "code",
    expanded: false,
    selection: null,
    playback: { playing: false, speed: 1, following: true, reveal: false },
    lens: null,
    canvas: { current: null },
  });
  const runOf = (id: string, code: string) =>
    ({
      id,
      project_id: "p",
      project: projectOf(code),
      trace: { ...trace(gpt), module_calls: [] },
    }) as unknown as Run;

  it("names a comment-only edit and a re-run, with nothing to compare", () => {
    const before = runOf("a", codeOf(gpt));
    const noted = carryOver(
      left(before),
      runOf("b", "# a note\n" + codeOf(gpt)),
    );
    expect(noted?.summary).toBe("comments only");
    expect(noted?.valuesMayDiffer).toBe(false);
    const again = carryOver(left(before), runOf("c", codeOf(gpt)));
    expect(again?.summary).toBe("re-run, same code");
    const scaled = carryOver(
      left(before),
      runOf("d", codeOf(gpt).replace("dim=-1", "dim=1")),
    );
    expect(scaled?.summary).toBe("same steps, values updated");
    expect(scaled?.valuesMayDiffer).toBe(true);
  });
});
