import { expect, test } from "vitest";
import { blankProject } from "../builder/model";
import { consoleProject } from "../console/script";
import { executionContextSignature as context } from "./executionContext";

test("helper edits, binding, entry root and environment invalidate recorded inlays", () => {
  const project = {
    ...blankProject("Model"),
    blueprint: null,
    files: { "layers.py": "a = 1" },
  };
  const before = context(project, "model.py");
  for (const changes of [
    { files: { "layers.py": "a = 2" } },
    { input_binding: "keyword" as const },
    { environment: "a".repeat(64) },
    { import_root: "src" },
    { constructor: { width: 4 } },
  ])
    expect(context({ ...project, ...changes }, "model.py")).not.toBe(before);
  expect(context({ ...project, code: "changed entry" }, "layers.py")).not.toBe(
    context(project, "layers.py"),
  );
});

test("only active source text and cosmetic provenance can retain earlier-line inlays", () => {
  const project = {
    ...blankProject("Model"),
    blueprint: null,
    files: { "layers.py": "a = 1" },
  };
  expect(
    context(
      { ...project, name: "Renamed", code: "changed active file" },
      "model.py",
    ),
  ).toBe(context(project, "model.py"));
  expect(
    context(
      { ...project, files: { "layers.py": "changed active file" } },
      "layers.py",
    ),
  ).toBe(context(project, "layers.py"));
  const script = consoleProject("Experiment");
  expect(
    context(
      { ...script, script: "y = x + 1", code: "derived wrapper" },
      "model.py",
    ),
  ).toBe(context(script, "model.py"));
  expect(context({ ...script, input_binding: "keyword" }, "model.py")).not.toBe(
    context(script, "model.py"),
  );
});
