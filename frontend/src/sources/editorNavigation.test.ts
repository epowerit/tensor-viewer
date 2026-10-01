import { describe, expect, it } from "vitest";
import type { Draft, Run } from "../api/client";
import { pythonProject } from "./pythonProject";
import { lineSelection, runErrorTarget } from "./editorNavigation";

const draft = pythonProject(
  "Test",
  "from torch import nn\nclass Model(nn.Module):\n    pass",
  "Model",
);
function failedRun(project: Draft, error: NonNullable<Run["trace"]["error"]>) {
  return { project, trace: { error } } as Run;
}

describe("navigation from a failed run", () => {
  it("opens the recorded entry or helper source at the exact failing line", () => {
    expect(
      runErrorTarget(
        draft,
        failedRun(draft, {
          type: "RuntimeError",
          message: "bad shape",
          line: 2,
        }),
      ),
    ).toEqual({ file: "model.py", line: 2 });
    const source = {
      ...draft,
      entry_path: "src/model.py",
      files: { "src/layer.py": "one\ntwo\nthree" },
    };
    expect(
      runErrorTarget(
        source,
        failedRun(source, {
          type: "ValueError",
          message: "bad",
          file: "src/layer.py",
          line: 3,
        }),
      ),
    ).toEqual({ file: "src/layer.py", line: 3 });
  });
  it("does not misidentify a line after edits or a file outside the project", () => {
    const run = failedRun(draft, {
      type: "RuntimeError",
      message: "bad",
      file: "model.py",
      line: 2,
    });
    expect(
      runErrorTarget({ ...draft, code: "# inserted\n" + draft.code }, run),
    ).toEqual({ file: "model.py", line: null });
    expect(
      runErrorTarget({ ...draft, entry_path: "other.py" }, run),
    ).toBeNull();
    expect(
      runErrorTarget(
        draft,
        failedRun(draft, {
          type: "Error",
          message: "bad",
          file: "/torch/nn.py",
          line: 42,
        }),
      ),
    ).toBeNull();
    expect(
      runErrorTarget(
        draft,
        failedRun(draft, { type: "Error", message: "bad", line: 999 }),
      ),
    ).toEqual({ file: "model.py", line: null });
  });
  it("selects exact textarea offsets for Unicode, CRLF, and empty lines", () => {
    const code = "# 🟣\r\nsecond\r\n\nlast";
    const second = lineSelection(code, 2)!;
    expect(code.slice(second.start, second.end)).toBe("second");
    const empty = lineSelection(code, 3)!;
    expect(empty.start).toBe(empty.end);
    expect(code.slice(lineSelection(code, 4)!.start)).toBe("last");
    for (const line of [null, undefined, 0, -1, 1.5, Infinity, 5])
      expect(lineSelection(code, line)).toBeNull();
  });
});
