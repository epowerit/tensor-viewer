import { describe, expect, it } from "vitest";
import { draftSignature, toDraft } from "../api/client";
import { blankProject } from "../builder/model";
import {
  chooseEntry,
  importedProject,
  pathIssue,
  projectFiles,
  sourceCode,
  updateFile,
} from "./files";

describe("project source snapshots", () => {
  it("preserves every file when changing the entry point or editing helpers", () => {
    const draft = {
      ...blankProject("Test"),
      blueprint: null,
      code: "entry",
      files: { "pkg/layer.py": "helper" },
    };
    const next = chooseEntry(draft, "pkg/layer.py");
    expect(next.code).toBe("helper");
    expect(projectFiles(next)).toEqual(projectFiles(draft));
    const edited = updateFile(next, "model.py", "changed");
    expect(sourceCode(edited, "model.py")).toBe("changed");
    expect(sourceCode(edited)).toBe("helper");
    expect(sourceCode(draft)).toBe("entry");
    expect(draftSignature(edited)).not.toBe(draftSignature(next));
    expect(chooseEntry(draft, "missing.py")).toBe(draft);
  });
  it("copies an imported snapshot without changing it or keeping a blueprint", () => {
    const source = {
      files: { "src/model.py": "entry", "src/helper.py": "helper" },
      repository: {
        url: "/repo",
        revision: "a".repeat(40),
        subdirectory: ".",
        sha256: "b".repeat(64),
      },
      skipped: 0,
    };
    const draft = importedProject(
      "Imported",
      source,
      "src/model.py",
      "Model",
      "src",
    );
    expect(draft.blueprint).toBeNull();
    expect(draft.import_root).toBe("src");
    expect(draft.files).toEqual({ "src/helper.py": "helper" });
    expect(source.files["src/model.py"]).toBe("entry");
    expect(draftSignature(draft)).toBe(draftSignature(toDraft(draft)));
  });
  it("normalizes legacy drafts and default graph connections", () => {
    const old = blankProject("Old");
    old.blueprint!.components = [{ id: "one", kind: "relu", parameters: {} }];
    const saved = toDraft(old);
    saved.blueprint = {
      ...old.blueprint!,
      components: [{ ...old.blueprint!.components[0], sources: null }],
    };
    expect(draftSignature(old)).toBe(draftSignature(saved));
    saved.blueprint.components[0].sources = ["input"];
    expect(draftSignature(old)).not.toBe(draftSignature(saved));
    const custom = { ...old, blueprint: null };
    expect(draftSignature(custom)).toBe(draftSignature(toDraft(custom)));
    expect(draftSignature({ ...custom, environment: "a".repeat(64) })).not.toBe(
      draftSignature(custom),
    );
  });
  it("rejects paths outside a relative source tree", () => {
    for (const path of [
      "../a.py",
      "/tmp/a.py",
      "a//b.py",
      "a\\b.py",
      ".git/config.py",
      "a\0.py",
      "binary.pt",
    ])
      expect(pathIssue(path)).toBeTruthy();
    expect(pathIssue("src/package/attention.py")).toBeNull();
    expect(pathIssue("requirements.txt")).toBeNull();
  });
});
