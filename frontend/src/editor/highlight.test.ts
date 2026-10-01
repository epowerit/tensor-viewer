import { expect, test } from "vitest";
import { caretPosition, highlightLine, visualWidth } from "./highlight";

const kinds = (line: string, tensors: string[] = []) =>
  highlightLine(line, new Set(tensors))
    .filter((token) => token.text.trim())
    .map((token) => `${token.kind}:${token.text.trim()}`);

test("tensor names, calls, namespaces, and literals are told apart", () => {
  expect(
    kinds("scores = q @ keys.transpose(-2, -1)", ["scores", "q", "keys"]),
  ).toEqual([
    "tensor:scores",
    "operator:=",
    "tensor:q",
    "operator:@",
    "tensor:keys",
    "plain:.",
    "call:transpose",
    "plain:(",
    "operator:-",
    "number:2",
    "plain:,",
    "operator:-",
    "number:1",
    "plain:)",
  ]);
  expect(kinds("w = torch.softmax(x, dim=-1)", ["x"]).slice(0, 5)).toEqual([
    "plain:w",
    "operator:=",
    "namespace:torch",
    "plain:.",
    "call:softmax",
  ]);
  // A member named like a keyword or tensor is still a member.
  expect(kinds("a.x.is_floating_point", ["x"])).toEqual([
    "plain:a.x.is_floating_point",
  ]);
});

test("comments, axis annotations, and strings keep their text intact", () => {
  expect(kinds('y = f("a # b")  # axes: batch, tokens')).toEqual([
    "plain:y",
    "operator:=",
    "call:f",
    "plain:(",
    'string:"a # b"',
    "plain:)",
    "axes:# axes: batch, tokens",
  ]);
  expect(kinds("return None  # done")).toEqual([
    "keyword:return",
    "keyword:None",
    "comment:# done",
  ]);
  expect(kinds("s = 'open")).toEqual(["plain:s", "operator:=", "string:'open"]);
  for (const line of ["", "  x\t= 1.5e-3 ** 2", 'print(f"{x}")', "λ = ∑"])
    expect(
      highlightLine(line)
        .map((token) => token.text)
        .join(""),
    ).toBe(line);
});

test("widths expand tabs and caret offsets become line and column", () => {
  expect(visualWidth("abc")).toBe(3);
  expect(visualWidth("\tx")).toBe(5);
  expect(visualWidth("ab\tx")).toBe(5);
  expect(caretPosition("ab\ncde", 0)).toEqual({ line: 1, column: 1 });
  expect(caretPosition("ab\ncde", 5)).toEqual({ line: 2, column: 3 });
  expect(caretPosition("ab\n", 3)).toEqual({ line: 2, column: 1 });
});
