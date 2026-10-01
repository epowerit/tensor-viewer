import { expect, test } from "vitest";
import { applyCompletion, completionsAt, wordAt } from "./completions";

const shapes = { x: [2, 3, 4], v: [5], s: [] };
const at = (source: string, latest = "x") =>
  completionsAt(source, source.length, shapes, latest, 50);

test("tensor methods show the shape they would produce", () => {
  const all = at("y = x.")!;
  const detail = (label: string) =>
    all.items.find((item) => item.label === label)?.detail;
  expect(all.from).toBe(6);
  expect(detail("reshape")).toBe("[2, 3, 4] → [6, 4]");
  expect(detail("permute")).toBe("[2, 3, 4] → [4, 3, 2]");
  expect(detail("transpose")).toBe("[2, 3, 4] → [2, 4, 3]");
  expect(detail("flatten")).toBe("[2, 3, 4] → [2, 12]");
  expect(detail("sum")).toBe("[2, 3, 4] → [2, 3]");
  expect(detail("unsqueeze")).toBe("[2, 3, 4] → [1, 2, 3, 4]");
  expect(detail("repeat")).toBe("[2, 3, 4] → [4, 3, 4]");
  expect(detail("chunk")).toBe("split into parts");
  expect(all.items.find((item) => item.label === "permute")?.insert).toBe(
    "permute(2, 1, 0)",
  );
  expect(all.items.find((item) => item.label === "repeat")?.insert).toBe(
    "repeat(2, 1, 1)",
  );
});

test("the typed prefix filters, and rank rules hide what cannot apply", () => {
  const typed = at("y = x.tr")!;
  expect(typed.items.map((item) => item.label)).toEqual(["transpose"]);
  expect(typed.from).toBe("y = x.".length);
  expect(at("v.")!.items.map((item) => item.label)).not.toContain("transpose");
  expect(at("v.fl")!.items[0]).toMatchObject({
    insert: "flatten()",
    detail: "[5] → [5]",
  });
  expect(at("s.")!.items.map((item) => item.label)).toEqual([
    "unsqueeze",
    "squeeze",
    "repeat",
    "clone",
    "contiguous",
    "masked_fill",
    "shape",
  ]);
  expect(at("x.T")!.items.map((item) => item.label)).toEqual([
    "transpose",
    "T",
  ]);
  expect(at("x.shape")).toBeNull();
  expect(at("x.zzz")).toBeNull();
  expect(at("unknown.")).toBeNull();
  expect(at("x")).toBeNull();
  expect(at("x. ")).toBeNull();
  expect(completionsAt("a = x.\nb = 1", 6, shapes, "x")!.from).toBe(6);
});

test("namespaces suggest functions applied to the latest variable", () => {
  const torch = at("z = torch.s", "scores")!;
  expect(torch.items.map((item) => item.insert)).toEqual([
    "stack([scores, scores], dim=0)",
    "softmax(scores, dim=-1)",
  ]);
  expect(at("F.re", "y")!.items[0].insert).toBe("relu(y)");
  expect(at("F.nothing")).toBeNull();
});

test("words under the pointer exclude members", () => {
  expect(wordAt("scores = q @ keys.transpose(-2, -1)", 0)).toBe("scores");
  expect(wordAt("scores = q @ keys.transpose(-2, -1)", 9)).toBe("q");
  expect(wordAt("scores = q @ keys.transpose(-2, -1)", 15)).toBe("keys");
  expect(wordAt("scores = q @ keys.transpose(-2, -1)", 20)).toBeNull();
  expect(wordAt("scores = q", 7)).toBeNull();
  expect(wordAt("", 3)).toBeNull();
});

test.each([
  "# try torch.ca",
  "y = x # x.re",
  'print("torch.ca',
  "label = 'x.re",
  'label = "escaped \\" quote torch.ca',
  'doc = """first line\ntry torch.ca',
  "doc = '''first line\ntry x.re",
  'label = f"tensor {x.re',
  'label = r"x.re',
])("does not offer completions in quoted text or comments: %s", (source) => {
  expect(at(source)).toBeNull();
});

test.each([
  "# try torch.ca\ny = x.re",
  'label = "# torch.ca"; y = x.re',
  'doc = """try torch.ca\n"""\ny = x.re',
  "doc = '''try x.re\n'''\ny = x.re",
])("resumes suggestions after text and comments: %s", (source) => {
  expect(at(source)?.items.map((item) => item.label)).toContain("reshape");
});

test("accepting a suggestion replaces the whole identifier around the caret", () => {
  const source = "y = x.reshape\nz = y + 1";
  const range = completionsAt(source, "y = x.resh".length, shapes, "x")!;
  expect([range.from, range.to]).toEqual([6, 13]);
  const next = applyCompletion(source, range, range.items[0]);
  expect(next).toEqual({
    value: "y = x.reshape(-1, 4)\nz = y + 1",
    caret: "y = x.reshape(-1, 4)".length,
  });
});

test("accepting a function name preserves arguments already written", () => {
  for (const source of ["y = x.reshape(3, 8)", "y = x.reshape (3, 8)"]) {
    const range = completionsAt(source, "y = x.resh".length, shapes, "x")!;
    expect(applyCompletion(source, range, range.items[0])).toEqual({
      value: source,
      caret: "y = x.reshape".length,
    });
  }
  const source = "y = torch.cat([x, x], dim=1)";
  const range = completionsAt(source, "y = torch.ca".length, shapes, "x")!;
  expect(applyCompletion(source, range, range.items[0]).value).toBe(source);
});

test("inherited JavaScript names are not namespaces or recorded tensors", () => {
  for (const name of ["constructor", "toString", "__proto__"])
    expect(at(`${name}.`)).toBeNull();
  expect(
    completionsAt("constructor.re", 14, { constructor: [2, 3] }, "x")?.items[0]
      .insert,
  ).toBe("reshape(-1, 3)");
});
