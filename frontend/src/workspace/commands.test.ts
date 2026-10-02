import { expect, test } from "vitest";
import { filterCommands, type Command } from "./commands";

const step = (id: string, label: string, detail: string): Command => ({
  id,
  group: "Steps",
  label,
  detail,
  run: () => {},
});

test("words typed together rank entries where they stand together", () => {
  const commands = [
    step("a", "Step 47 · softmax → weights", "[1, 2, 7, 7] · pass 1"),
    step("b", "Step 95 · softmax → weights", "[1, 2, 7, 7] · pass 2"),
  ];
  expect(filterCommands(commands, "softmax pass 2").map((c) => c.id)).toEqual([
    "b",
    "a",
  ]);
  // Without the pair, order is unchanged.
  expect(filterCommands(commands, "softmax").map((c) => c.id)).toEqual([
    "a",
    "b",
  ]);
});
