import { expect, test } from "vitest";
import { filterCommands, type Command } from "./commands";
import { formatLocation, parseLocation } from "./links";

test("links round-trip a project, run, step, and cell", () => {
  const location = {
    project: "3f2c-ab",
    run: "9d1e",
    node: "op12",
    cell: 17,
  };
  expect(formatLocation(location)).toBe(
    "#project=3f2c-ab&run=9d1e&node=op12&cell=17",
  );
  expect(parseLocation(formatLocation(location))).toEqual(location);
  expect(parseLocation("#project=p&run=r&node=input-t0")).toEqual({
    project: "p",
    run: "r",
    node: "input-t0",
  });
  expect(formatLocation({ project: "p", run: "r", node: "op0", cell: 0 })).toBe(
    "#project=p&run=r&node=op0",
  );
  expect(formatLocation({})).toBe("");
});

test("malformed or orphaned link parts are ignored", () => {
  expect(parseLocation("")).toEqual({});
  expect(parseLocation("#run=r&node=op1")).toEqual({});
  expect(parseLocation("#project=p&node=op1&cell=3")).toEqual({ project: "p" });
  expect(parseLocation("#project=p&run=../etc&node=op1")).toEqual({
    project: "p",
  });
  expect(parseLocation("#project=p&run=r&node=op1&cell=-4")).toEqual({
    project: "p",
    run: "r",
    node: "op1",
  });
  expect(parseLocation("#project=<script>")).toEqual({});
  // A later link without a node must not inherit a stale cell.
  expect(formatLocation({ project: "p", cell: 9 })).toBe("#project=p");
});

test("commands match every typed word and rank label prefixes first", () => {
  const make = (
    id: string,
    group: Command["group"],
    label: string,
    detail?: string,
  ): Command => ({ id, group, label, detail, run: () => {} });
  const commands = [
    make("run", "Actions", "Run", "Save and record"),
    make("s1", "Steps", "Step 1 · reshape → grouped", "[2, 3, 2, 2]"),
    make("s2", "Steps", "Step 2 · permute → swapped", "[2, 2, 3, 2]"),
    make("t1", "Tensors", "swapped", "[2, 2, 3, 2] float32"),
    make("p1", "Projects", "Permutation playground"),
  ];
  const ids = (query: string) =>
    filterCommands(commands, query).map((c) => c.id);
  expect(ids("")).toEqual(["run", "s1", "s2", "t1", "p1"]);
  expect(ids("perm")).toEqual(["p1", "s2"]);
  expect(ids("swapped")).toEqual(["t1", "s2"]);
  expect(ids("step 2")).toEqual(["s2", "s1"]);
  expect(ids("2, 3, 2")).toEqual(["s1", "s2", "t1"]);
  expect(ids("tensors")).toEqual(["t1"]);
  expect(ids("nothing here")).toEqual([]);
  expect(filterCommands(commands, "", 2)).toHaveLength(2);
});
