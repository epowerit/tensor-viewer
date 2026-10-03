import { describe, expect, it, vi } from "vitest";
import type { Run } from "../api/client";
import type { AxisPart } from "../tensors/axisLineage";
import {
  layoutSemantics,
  layoutStages,
  readRelayout,
  relayoutTitle,
} from "./relayout";
import {
  detailLevelOf,
  detailLevels,
  followFolds,
  levelCards,
  readable,
  rememberFolds,
  rememberedFolds,
  startingLevel,
  type JourneyStage,
} from "./stages";

/**
 * An axis map written out: result axes split by "|", their parts by "×", each
 * part "axis:size" or, for a piece, "axis.index/of:size"; "-" is a size-1
 * axis with no source.
 */
const axes = (text: string): AxisPart[][] =>
  text.split("|").map((out) =>
    out === "-"
      ? []
      : out.split("×").map((part) => {
          const [where, size] = part.split(":");
          const [axis, piece] = where.split(".");
          const [index, of] = (piece ?? "").split("/").map(Number);
          return {
            axis: Number(axis),
            size: Number(size),
            ...(piece ? { piece: { index, of } } : {}),
          };
        }),
  );

describe("readRelayout", () => {
  it("reads attention's head split and merge", () => {
    // nanoGPT, BERT: [B, T, H·D] → [B, H, T, D], and back.
    const split = readRelayout(
      [1, 13, 16],
      [1, 2, 13, 8],
      axes("0:1|2.0/2:2|1:13|2.1/2:8"),
    );
    expect(split.title).toBe("Split into heads");
    expect(split.how).toBe(
      "the last axis, 16, splits into 2 × 8, and the 2 moves to axis 1",
    );
    expect(split.axes).toEqual([0, "heads", 1, "head features"]);
    const merge = readRelayout(
      [1, 2, 13, 8],
      [1, 13, 16],
      axes("0:1|2:13|1:2×3:8"),
    );
    expect(merge.title).toBe("Merge heads");
    expect(merge.axes).toEqual([0, 2, "features"]);
  });
  it("reads a fused q, k, v split into heads", () => {
    // timm: qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
    const timm = readRelayout(
      [1, 16, 96],
      [3, 1, 4, 16, 8],
      axes("2.0/3:3|0:1|2.1/3:4|1:16|2.2/3:8"),
    );
    expect(timm.title).toBe("Split q, k, v into heads");
    expect(timm.how).toBe(
      "the last axis, 96, splits into 3 × 4 × 8: q, k, and v, each 4 heads of 8; q, k, v move to axis 0 and the heads to axis 2",
    );
    expect(timm.axes).toEqual(["q, k, v", 0, "heads", 1, "head features"]);
  });
  it("reads vision models' windows, patches, and tokens", () => {
    // Swin: [B, H, W, C] → [B·windows, s·s, C], and back.
    const windows = readRelayout(
      [1, 8, 8, 12],
      [4, 16, 12],
      axes("1.0/2:2×2.0/2:2|1.1/2:4×2.1/2:4|3:12"),
    );
    expect(windows.title).toBe("Partition into windows");
    expect(windows.how).toBe(
      "the 8 × 8 grid is cut into 4 windows of 4 × 4, each a row of 16 positions",
    );
    expect(windows.axes).toEqual(["windows", "positions", 3]);
    expect(
      relayoutTitle(
        [4, 16, 12],
        [1, 8, 8, 12],
        axes("-|0.0/2:2×1.0/2:4|0.1/2:2×1.1/2:4|2:12"),
      ),
    ).toBe("Restore window grid");
    // A masked autoencoder: [B, C, H, W] → [B, patches, p·p·C], and back.
    expect(
      relayoutTitle(
        [1, 3, 16, 16],
        [1, 16, 48],
        axes("0:1|2.0/2:4×3.0/2:4|2.1/2:4×3.1/2:4×1:3"),
      ),
    ).toBe("Cut into patches");
    expect(
      relayoutTitle(
        [1, 16, 48],
        [1, 3, 16, 16],
        axes("0:1|2.2/3:3|1.0/2:4×2.0/3:4|1.1/2:4×2.1/3:4"),
      ),
    ).toBe("Patches to image");
    // ViT: [B, C, H, W] → [B, H·W, C], and back.
    expect(
      relayoutTitle([2, 16, 4, 4], [2, 16, 16], axes("0:2|2:4×3:4|1:16")),
    ).toBe("Grid to tokens");
    expect(
      relayoutTitle(
        [2, 196, 64],
        [2, 64, 14, 14],
        axes("0:2|2:64|1.0/2:14|1.1/2:14"),
      ),
    ).toBe("Tokens to grid");
  });
  it("reads flattens, reorders, and size-1 axes", () => {
    expect(relayoutTitle([2, 3, 4, 5], [2, 60], axes("0:2|1:3×2:4×3:5"))).toBe(
      "Flatten",
    );
    expect(relayoutTitle([4, 2, 8], [4, 16], axes("0:4|1:2×2:8"))).toBe(
      "Merge last axes",
    );
    expect(relayoutTitle([4, 16], [4, 2, 8], axes("0:4|1.0/2:2|1.1/2:8"))).toBe(
      "Split last axis",
    );
    // The console's reshape → permute → flatten: reordered on the way.
    expect(
      relayoutTitle([2, 3, 4], [2, 12], axes("0:2|2.0/2:2×1:3×2.1/2:2")),
    ).toBe("Regroup and reorder");
    expect(relayoutTitle([2, 3, 4], [4, 2, 3], axes("2:4|0:2|1:3"))).toBe(
      "Reorder axes",
    );
    expect(relayoutTitle([3, 4], [1, 3, 4, 1], axes("-|0:3|1:4|-"))).toBe(
      "Add size-1 axes",
    );
    expect(relayoutTitle([2, 3], [2, 3], axes("0:2|1:3"))).toBe("Re-layout");
  });
  it("names by shape alone when the steps cannot be traced", () => {
    expect(relayoutTitle([2, 3, 4], [2, 12])).toBe("Flatten");
    expect(relayoutTitle([2, 3], [2, 3])).toBe("Re-layout");
  });
});

const tensor = (id: string, shape: number[]) => ({
  id,
  shape,
  name: id,
  axes: shape.map((_, axis) => `axis ${axis}`),
});
const op = (
  index: number,
  kind: string,
  inputs: string[],
  outputs: string[],
  module = "Net / attention",
  order?: number[],
) => ({
  id: `op${index}`,
  index,
  kind,
  inputs,
  outputs,
  module,
  status: "ok",
  arguments: {},
  // What a trace records for a layout step: the axis order of a permute,
  // or that a reshape regroups.
  lesson: order
    ? { axis_order: order, mapping_rule: "permutation" }
    : { mapping_rule: "identity" },
});
// q from a linear, split into heads in two steps, then used by a matmul;
// k's reshape result is read twice, so its run is not folded.
const run = {
  trace: {
    output_ids: ["t9"],
    tensors: Object.fromEntries(
      [
        tensor("t0", [1, 13, 16]),
        tensor("t1", [1, 13, 2, 8]),
        tensor("t2", [1, 2, 13, 8]),
        tensor("t3", [1, 13, 16]),
        tensor("t4", [1, 13, 2, 8]),
        tensor("t5", [1, 2, 13, 8]),
        tensor("t9", [1, 2, 13, 13]),
      ].map((t) => [t.id, t]),
    ),
    operations: [
      op(0, "linear", ["x"], ["t0"]),
      op(1, "reshape", ["t0"], ["t1"]),
      op(2, "transpose", ["t1"], ["t2"], undefined, [0, 2, 1, 3]),
      op(3, "linear", ["x"], ["t3"]),
      op(4, "reshape", ["t3"], ["t4"]),
      op(5, "transpose", ["t4"], ["t5"], undefined, [0, 2, 1, 3]),
      op(6, "matmul", ["t2", "t5", "t4"], ["t9"]),
    ],
  },
} as unknown as Run;
const attention = {
  id: "stage-call1",
  parentStageId: null,
  operationIds: ["op0", "op1", "op2", "op3", "op4", "op5", "op6"],
  path: "attention",
  title: "Attention",
} as JourneyStage;

describe("layoutStages", () => {
  it("folds a chain that hands one tensor straight on", () => {
    const capsules = layoutStages(run, [attention]);
    expect(capsules).toHaveLength(1);
    expect(capsules[0]).toMatchObject({
      title: "Split into heads",
      operationIds: ["op1", "op2"],
      inputs: ["t0"],
      outputs: ["t2"],
      parentStageId: "stage-call1",
      layout: { from: [1, 13, 16], to: [1, 2, 13, 8] },
      start_index: 1,
      end_index: 3,
    });
    // Traced through reshape and transpose: the last axis cut in two, its
    // first piece moved ahead of the tokens.
    expect(capsules[0].layout?.map).toEqual([
      [{ axis: 0, size: 1 }],
      [{ axis: 2, size: 2, piece: { index: 0, of: 2 } }],
      [{ axis: 1, size: 13 }],
      [{ axis: 2, size: 8, piece: { index: 1, of: 2 } }],
    ]);
  });

  it("explains a capsule in the shapes and the steps it folds", () => {
    const [capsule] = layoutStages(run, [attention]);
    const caption = layoutSemantics(run, capsule)!;
    expect(caption.title).toBe("Split into heads");
    expect(caption.summary).toBe(
      "t0 [1 · 13 · 16] → [1 · heads 2 · 13 · head features 8]: the last axis, 16, splits into 2 × 8, and the 2 moves to axis 1. Every value is carried over unchanged; reshape then transpose only change the order they are read in.",
    );
  });

  it("names a window cut's unnamed axes from the idiom", () => {
    const grid = {
      ...tensor("g", [1, 8, 8, 12]),
      axes: ["batch", "height", "width", "features"],
    };
    const windows = run.trace.tensors;
    const swin = {
      trace: {
        output_ids: [],
        tensors: {
          g: grid,
          a: tensor("a", [1, 2, 4, 2, 4, 12]),
          b: tensor("b", [1, 2, 2, 4, 4, 12]),
          w: {
            ...tensor("w", [4, 16, 12]),
            axes: ["axis 0", "axis 1", "axis 2"],
          },
          ...windows,
        },
        operations: [
          op(0, "reshape", ["g"], ["a"]),
          op(1, "permute", ["a"], ["b"], undefined, [0, 1, 3, 2, 4, 5]),
          op(2, "reshape", ["b"], ["w"]),
          op(3, "linear", ["w"], ["t9"]),
        ],
      },
    } as unknown as Run;
    const [capsule] = layoutStages(swin, []);
    expect(capsule.title).toBe("Partition into windows");
    expect(layoutSemantics(swin, capsule)!.summary).toContain(
      "g [batch 1 · height 8 · width 8 · features 12] → [windows 4 · positions 16 · features 12]",
    );
  });

  it("folds a split's parts in with the chain that shaped them", () => {
    // timm: qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4).unbind(0)
    const fused = {
      trace: {
        output_ids: [],
        tensors: {
          x: tensor("x", [1, 16, 96]),
          r: tensor("r", [1, 16, 3, 4, 8]),
          p: tensor("p", [3, 1, 4, 16, 8]),
          q: tensor("q", [1, 4, 16, 8]),
          k: tensor("k", [1, 4, 16, 8]),
          v: tensor("v", [1, 4, 16, 8]),
          t9: tensor("t9", [1, 4, 16, 16]),
        },
        operations: [
          op(0, "reshape", ["x"], ["r"]),
          op(1, "permute", ["r"], ["p"], undefined, [2, 0, 3, 1, 4]),
          op(2, "unbind", ["p"], ["q", "k", "v"]),
          op(3, "matmul", ["q", "k"], ["t9"]),
        ],
      },
    } as unknown as Run;
    const [capsule] = layoutStages(fused, []);
    expect(capsule).toMatchObject({
      title: "Split q, k, v into heads",
      operationIds: ["op0", "op1", "op2"],
      outputs: ["q", "k", "v"],
      layout: { from: [1, 16, 96], to: [3, 1, 4, 16, 8], via: "p" },
    });
    expect(layoutSemantics(fused, capsule)!.summary).toContain(
      "x [1 · 16 · 96] → [q, k, v 3 · 1 · heads 4 · 16 · head features 8]: the last axis, 96, splits into 3 × 4 × 8: q, k, and v, each 4 heads of 8; q, k, v move to axis 0 and the heads to axis 2, which unbind hands on as 3 tensors, q, k, v.",
    );
  });

  it("adds no capsule where a recorded call is exactly the run", () => {
    const split = {
      ...attention,
      id: "stage-split",
      operationIds: ["op1", "op2"],
    };
    const inner = { ...split, parentStageId: "stage-call1" };
    expect(layoutStages(run, [attention, inner])).toEqual([]);
    // The outermost call, the whole model or a console's wrapper, does not.
    expect(layoutStages(run, [split])).toHaveLength(1);
  });
});

describe("detailLevels", () => {
  const stage = (id: string, path: string, parent: string | null, steps = 2) =>
    ({
      id,
      path,
      title: path,
      parentStageId: parent,
      operationIds: Array.from({ length: steps }, (_, i) => `${id}${i}`),
    }) as JourneyStage;
  const stages = [
    stage("gpt", "GPT", null),
    stage("b0", "blocks.0", "gpt"),
    stage("b1", "blocks.1", "gpt"),
    stage("a0", "blocks.0.attention", "b0", 19),
    stage("f0", "blocks.0.feed_forward", "b0"),
    { ...stage("c0", "", "a0"), layout: { from: [1], to: [1] } },
  ];
  const levels = detailLevels(stages);
  it("names each level after the calls it shows, coarse to fine", () => {
    expect(levels.map((level) => level.label)).toEqual([
      "GPT",
      "blocks",
      "attention · feed_forward",
      "operations",
      "every step",
    ]);
  });
  it("folds the calls at and below a level, and capsules until the last", () => {
    expect([...levels[1].collapsed].sort()).toEqual(
      ["a0", "b0", "b1", "c0", "f0"].sort(),
    );
    expect([...levels[3].collapsed]).toEqual(["c0"]);
    expect(levels[4].collapsed.size).toBe(0);
  });
  it("finds the level a set of folds is, or none", () => {
    expect(detailLevelOf(levels, new Set(["c0"]))).toBe(3);
    expect(detailLevelOf(levels, new Set(["b0"]))).toBe(-1);
  });
});

describe("followFolds", () => {
  const capsule = (id: string) => id.startsWith("cap");
  it("opens the folds around the step and closes capsules once it leaves", () => {
    const entered = followFolds(
      new Set(["cap1", "block"]),
      new Set(),
      new Set(["block", "cap1"]),
      capsule,
    );
    expect([...entered.collapsed]).toEqual([]);
    expect([...entered.revealed]).toEqual(["cap1"]);
    const left = followFolds(
      entered.collapsed,
      entered.revealed,
      new Set(["block"]),
      capsule,
    );
    // The capsule folds again; the call opened for the step stays open.
    expect([...left.collapsed]).toEqual(["cap1"]);
    expect(left.revealed.size).toBe(0);
  });
  it("leaves a capsule opened by hand open", () => {
    const moved = followFolds(new Set(), new Set(), new Set(), capsule);
    expect(moved.collapsed.size).toBe(0);
  });
});

describe("readable", () => {
  it("splits class names into words, keeping acronyms whole", () => {
    expect(readable("CausalSelfAttention")).toBe("Causal Self Attention");
    expect(readable("TinyViT")).toBe("Tiny ViT");
    expect(readable("MLPBlock")).toBe("MLP Block");
    expect(readable("GPT2Model")).toBe("GPT2 Model");
    expect(readable("BertSelfAttention")).toBe("Bert Self Attention");
    expect(readable("GPT")).toBe("GPT");
  });
});

describe("startingLevel", () => {
  // A model of two blocks, each of 15 steps with a 2-step capsule inside;
  // the second block is a folded loop's later pass.
  const block = (n: number) =>
    Array.from({ length: 15 }, (_, i) => `b${n}s${i}`);
  const ops = [...block(0), ...block(1)];
  const stages = [
    { id: "model", parentStageId: null, operationIds: ops, path: "Net" },
    {
      id: "b0",
      parentStageId: "model",
      operationIds: block(0),
      path: "blocks.0",
    },
    {
      id: "b1",
      parentStageId: "model",
      operationIds: block(1),
      path: "blocks.1",
    },
    {
      id: "c0",
      parentStageId: "b0",
      operationIds: ["b0s3", "b0s4"],
      path: "",
      layout: { from: [1], to: [1] },
    },
  ] as JourneyStage[];
  const levels = detailLevels(stages);
  const none = new Set<string>();
  it("counts the cards a level draws", () => {
    expect(levels.map((level) => levelCards(level, stages, ops, none))).toEqual(
      [1, 2, 29, 30],
    );
    // The loop's later pass is drawn once, not counted again.
    const later = new Set(block(1));
    expect(levelCards(levels[3], stages, ops, later)).toBe(15);
  });
  it("opens on the finest level that fits", () => {
    expect(startingLevel(levels, stages, ops, none)?.label).toBe("blocks");
    expect(startingLevel(levels, stages, ops, new Set(block(1)))?.label).toBe(
      "every step",
    );
  });
});

describe("remembered folds", () => {
  const store = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => store.set(key, value),
  });
  const known = [{ id: "a" }, { id: "b" }] as JourneyStage[];
  it("brings a project's folds back, dropping stages that are gone", () => {
    rememberFolds("p1", ["a", "gone"]);
    expect([...rememberedFolds("p1", known)!]).toEqual(["a"]);
    // Every step shown is kept too.
    rememberFolds("p2", []);
    expect(rememberedFolds("p2", known)?.size).toBe(0);
  });
  it("opens on the starting detail when nothing kept still applies", () => {
    expect(rememberedFolds("never", known)).toBeNull();
    rememberFolds("p3", ["gone"]);
    expect(rememberedFolds("p3", known)).toBeNull();
  });
});
