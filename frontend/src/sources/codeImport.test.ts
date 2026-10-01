import { expect, test } from "vitest";
import { codeImportIssue, projectFromCode, sourceClasses } from "./codeImport";

test("class suggestions ignore quoted text, comments and nested classes", () => {
  const source = `from torch import nn
# class Fake(nn.Module):
note = '''example
class Quoted(nn.Module):
'''
class Base(nn.Module):
    pass
class Model(
    Base,
):
    class Nested: pass
    def forward(self, x): return x
`;
  expect(sourceClasses(source)).toEqual(["Base", "Model"]);
  expect(sourceClasses("y = nn.Linear(8, 4)(x)")).toEqual([]);
  expect(sourceClasses("\ufeffclass Net(nn.Module):\n    pass")).toEqual([
    "Net",
  ]);
});

test("module source is preserved and never wrapped as tensor statements", () => {
  const code =
    "from torch import nn\nclass Net(nn.Module):\n    def forward(self, x): return x.chunk(2, dim=-1)\n";
  const project = projectFromCode("Network", code, "module", "Net");
  expect(project).toMatchObject({
    code,
    class_name: "Net",
    script: null,
    blueprint: null,
    constructor: {},
  });
  const script = "left, right = x.chunk(2, dim=-1)\nreturn left, right";
  expect(projectFromCode("Split", script, "statements", "")).toMatchObject({
    script,
    class_name: "Console",
    blueprint: null,
  });
});

test("import validates empty code, entry name, and entry-specific limits", () => {
  expect(codeImportIssue(" ", "statements", "")).toBeTruthy();
  expect(codeImportIssue("pass", "module", "bad.name")).toBeTruthy();
  expect(codeImportIssue("x".repeat(20_001), "statements", "")).toBeTruthy();
  expect(codeImportIssue("x".repeat(20_001), "module", "Net")).toBeNull();
  expect(codeImportIssue("x".repeat(500_001), "module", "Net")).toBeTruthy();
  expect(codeImportIssue("from layers import Net", "module", "Net")).toBeNull();
});
