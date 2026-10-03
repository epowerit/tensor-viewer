import { describe, expect, it } from "vitest";
import { followLines, lineMap } from "./lineMap";

const code = [
  "def forward(self, x):",
  "    x = self.norm(x)",
  "    logits = x @ w.T",
  "    return logits.softmax(-1)",
].join("\n");

describe("lineMap", () => {
  it("shifts lines below an inserted line", () => {
    const after = code.replace(
      "    x = self.norm(x)",
      "    x = self.drop(x)\n    x = self.norm(x)",
    );
    const map = lineMap(code, after);
    expect([1, 2, 3, 4].map(map)).toEqual([1, 3, 4, 5]);
  });

  it("keeps an edited line where its edit is", () => {
    const after = code.replace(
      "return logits.softmax(-1)",
      "logits = logits.softmax(-1)\n    return logits.argmax(-1)",
    );
    expect(lineMap(code, after)(4)).toBe(4);
    expect(lineMap(code, after)(3)).toBe(3);
  });

  it("drops a line removed with nothing in its place, and moves the rest up", () => {
    const after = code.replace("    x = self.norm(x)\n", "");
    const map = lineMap(code, after);
    expect([1, 2, 3, 4].map(map)).toEqual([1, null, 2, 3]);
  });

  it("follows unchanged lines inside a larger edit", () => {
    const before = "a = 1\nb = 2\nc = 3\nd = 4";
    const after = "a = 10\nz = 0\nb = 2\nc = 30\nd = 4";
    expect([1, 2, 3, 4].map(lineMap(before, after))).toEqual([1, 3, 4, 5]);
  });
});

describe("followLines", () => {
  it("moves breakpoints with the code and drops removed ones", () => {
    const after = "# header\n" + code.replace("    logits = x @ w.T\n", "");
    expect(followLines([2, 3, 4], code, after)).toEqual([3, 4]);
  });
});
