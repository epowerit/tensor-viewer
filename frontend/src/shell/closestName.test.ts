import { expect, test } from "vitest";
import { closestName, missingName } from "./closestName";

test("a mistyped name is matched to the nearest name in scope", () => {
  const names = ["scaled", "logged", "x", "weights", "layer_norm"];
  expect(closestName("scaledd", names)).toBe("scaled");
  expect(closestName("wieghts", names)).toBe("weights");
  expect(closestName("layernorm", names)).toBe("layer_norm");
  expect(closestName("y", names)).toBe("x");
  expect(closestName("tokens", names)).toBeNull();
});

test("the missing name is read from Python's NameError", () => {
  expect(missingName("NameError: name 'scaledd' is not defined")).toBe(
    "scaledd",
  );
  expect(missingName("TypeError: unsupported operand")).toBeNull();
});
