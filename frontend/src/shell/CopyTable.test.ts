import { expect, test } from "vitest";
import { toCsv } from "./CopyTable";

test("rows become CSV, quoting what needs it", () => {
  expect(
    toCsv([
      ["layer", "once", "upon"],
      ["blocks.0", 0.5, null],
      ['say "hi", then', 1, "two\nlines"],
    ]),
  ).toBe('layer,once,upon\nblocks.0,0.5,\n"say ""hi"", then",1,"two\nlines"');
});
