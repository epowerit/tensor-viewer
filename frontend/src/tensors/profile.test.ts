import { expect, test } from "vitest";
import type { Run, Tensor } from "../api/client";
import { namedStates } from "./CellHistory";
import { axisProfile, defaultProfileAxis } from "./profile";

test("an axis profile summarises each index over every other axis", () => {
  const values: (number | string)[] = Array.from({ length: 24 }, (_, i) => i);
  const cube = { shape: [2, 3, 4], numel: 24, values };
  const profile = axisProfile(cube, 1)!;
  // Index 0 of axis 1 holds 0..3 and 12..15.
  expect(profile.mean[0]).toBe(7.5);
  expect(profile.min).toEqual([0, 4, 8]);
  expect(profile.max).toEqual([15, 19, 23]);
  // numpy.std([0, 1, 2, 3, 12, 13, 14, 15])
  expect(profile.std[0]).toBeCloseTo(6.103278, 5);
  values[5] = "nan";
  expect(axisProfile(cube, 1)!.mean[1]).toBeNaN();
  expect(axisProfile({ ...cube, values: [] }, 1)).toBeNull();
});

test("profiles start on a channel or feature axis when one is named", () => {
  expect(
    defaultProfileAxis({
      shape: [2, 8, 4, 4],
      axes: ["batch", "channels", "h", "w"],
    }),
  ).toBe(1);
  expect(
    defaultProfileAxis({
      shape: [2, 5, 16],
      axes: ["batch", "tokens", "features"],
    }),
  ).toBe(2);
  expect(defaultProfileAxis({ shape: [3, 4], axes: ["", ""] })).toBe(1);
  expect(defaultProfileAxis({ shape: [7], axes: ["x"] })).toBeNull();
});

test("a name's states are traced in the order the run wrote them", () => {
  const state = (id: string, name: string, shape = [2]) =>
    ({
      id,
      name,
      shape,
      value_source: "inline",
      values: [0, 0],
    }) as unknown as Tensor;
  const trace = {
    input_ids: ["t0"],
    operations: [
      { outputs: ["t1"], mutations: [] },
      { outputs: ["t2"], mutations: [{ before: "t1", after: "t3" }] },
      { outputs: ["t4"] },
    ],
    tensors: {
      t0: state("t0", "x"),
      t1: state("t1", "x"),
      t2: state("t2", "h"),
      t3: state("t3", "x"),
      t4: state("t4", "x", [3]),
    },
  } as unknown as Run["trace"];
  expect(
    namedStates(trace, trace.tensors.t1 as Tensor).map((s) => s.id),
  ).toEqual(["t0", "t1", "t3"]);
});
