import { expect, test } from "vitest";
import { kindName, ownName, stepLabel } from "./kindName";

test("operator methods read the way the code spells them", () => {
  expect(kindName("__getitem__")).toBe("index");
  expect(kindName("__invert__")).toBe("invert");
  expect(kindName("__rsub__")).toBe("subtract");
  expect(kindName("__lshift__")).toBe("lshift");
  expect(kindName("softmax")).toBe("softmax");
});

test("a result named after its operation has no name of its own", () => {
  expect(ownName("softmax", "softmax")).toBe(false);
  expect(ownName("chunk[1]", "chunk")).toBe(false);
  expect(ownName("scores", "softmax")).toBe(true);
});

test("a step names its result when the result has its own name", () => {
  expect(stepLabel("rsqrt", "scale")).toBe("rsqrt → scale");
  expect(stepLabel("rsqrt", "rsqrt")).toBe("rsqrt");
  expect(stepLabel("__getitem__", "x[:, 0]")).toBe("index → x[:, 0]");
});
