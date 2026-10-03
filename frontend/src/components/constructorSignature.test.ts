import { describe, expect, it } from "vitest";
import {
  constructorSignature,
  meant,
  withArgument,
} from "./constructorSignature";

const code = `import torch
from torch import nn


class Helper(nn.Module):
    def __init__(self, width):
        super().__init__()


class MultiHeadAttention(nn.Module):
    """Attention."""

    def __init__(
        self,
        dim: int = 8,
        heads=2,  # how many
        *,
        dropout: float = 0.1,
        name="attn",
        sizes=(1, 2),
        vocab_size,
        **kwargs,
    ):
        super().__init__()

    def forward(self, x):
        return x
`;

describe("constructorSignature", () => {
  it("reads each parameter, its default, and whether it is open", () => {
    const signature = constructorSignature(
      { "model.py": code },
      "MultiHeadAttention",
    );
    expect(signature).toEqual({
      open: true,
      parameters: [
        { name: "dim", python: "8", json: 8 },
        { name: "heads", python: "2", json: 2 },
        { name: "dropout", python: "0.1", json: 0.1 },
        { name: "name", python: '"attn"', json: "attn" },
        { name: "sizes", python: "(1, 2)" },
        { name: "vocab_size", python: null },
      ],
    });
  });
  it("finds the class in any file, and only its own __init__", () => {
    const files = { "a.py": "x = 1\n", "lib/helper.py": code };
    expect(constructorSignature(files, "Helper")?.parameters).toEqual([
      { name: "width", python: null },
    ]);
  });
  it("gives a class without __init__ no parameters, and a missing one null", () => {
    const plain =
      "class Plain(nn.Module):\n    def forward(self, x):\n        return x\n\n\nclass Other:\n    def __init__(self, a=1):\n        pass\n";
    expect(constructorSignature({ "m.py": plain }, "Plain")).toEqual({
      parameters: [],
      open: false,
    });
    expect(constructorSignature({ "m.py": plain }, "Missing")).toBeNull();
  });
});

describe("withArgument", () => {
  it("adds the default and says where the value sits", () => {
    const added = withArgument(
      { dim: 16 },
      { name: "heads", python: "2", json: 2 },
    );
    expect(JSON.parse(added.text)).toEqual({ dim: 16, heads: 2 });
    expect(added.text.slice(added.start, added.end)).toBe("2");
  });
  it("adds a required argument as null, ready to type over", () => {
    const added = withArgument({}, { name: "vocab_size", python: null });
    expect(added.text.slice(added.start, added.end)).toBe("null");
  });
});

describe("meant", () => {
  const parameters = [
    { name: "dim", python: "8", json: 8 },
    { name: "num_heads", python: "2", json: 2 },
  ];
  it("suggests the parameter a near-miss meant", () => {
    expect(meant("dims", parameters)).toBe("dim");
    expect(meant("num_head", parameters)).toBe("num_heads");
    expect(meant("Dim", parameters)).toBe("dim");
  });
  it("suggests nothing for a different name", () => {
    expect(meant("vocab_size", parameters)).toBeNull();
  });
});
