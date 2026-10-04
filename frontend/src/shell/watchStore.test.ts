import { expect, test } from "vitest";
import type { Run, WatchSeries } from "../api/client";
import { holdingSteps } from "./watchStore";

test("a condition pauses at the recorded steps where it is true", () => {
  const run = {
    trace: { operations: [{ id: "op4" }, { id: "op25" }, { id: "op30" }] },
  } as unknown as Run;
  const series = {
    points: [
      // The input's own state is not a step playback can pause at.
      { at: "input-t0", step: 0, value: 1 },
      { at: "op4", step: 5, value: 0 },
      { at: "op25", step: 26, value: null },
      { at: "op30", step: 31, value: 1 },
    ],
    truncated: false,
  } as WatchSeries;
  expect(holdingSteps(run, series).map((point) => point.at)).toEqual(["op30"]);
});
