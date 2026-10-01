import { describe, expect, it } from "vitest";
import { draftSignature, toDraft } from "../api/client";
import { forwardInputs, forwardIssue } from "../inputs/forward";
import {
  pythonProject,
  readPythonFile,
  suggestModuleClass,
} from "./pythonProject";

const source = `from torch import nn
class TinyModel(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)
`;
const file = (name = "model.py", text = source, size = text.length) => ({
  name,
  size,
  text: async () => text,
});

describe("direct Python project entry", () => {
  it("keeps pasted and uploaded code identical through saving and input setup", async () => {
    const pasted = pythonProject("  Tiny model  ", source, " TinyModel ", {
      dim: 8,
    });
    const uploaded = pythonProject(
      "Tiny model",
      await readPythonFile(file()),
      "TinyModel",
      { dim: 8 },
    );
    expect(draftSignature(pasted)).toBe(draftSignature(uploaded));
    expect(draftSignature(toDraft(pasted))).toBe(draftSignature(uploaded));
    expect(pasted.blueprint).toBeNull();
    expect(pasted.code).toBe(source);
    expect(pasted.class_name).toBe("TinyModel");
    expect(pasted.constructor).toEqual({ dim: 8 });
    expect(forwardIssue(forwardInputs(pasted), pasted.capture_mode)).toBe("");
  });
  it("suggests top-level module classes while allowing explicit class selection", () => {
    expect(suggestModuleClass(source)).toBe("TinyModel");
    expect(
      suggestModuleClass(
        "# class Comment(nn.Module):\nclass Actual(torch.nn.Module):\n    pass",
      ),
    ).toBe("Actual");
    expect(suggestModuleClass("class Network(\n Module\n):\n    pass")).toBe(
      "Network",
    );
    expect(suggestModuleClass("class Plain:\n    pass")).toBe("");
    expect(
      pythonProject("Custom", "class Derived(Base):\n    pass", "Derived")
        .class_name,
    ).toBe("Derived");
  });
  it("validates file type and size before reading file contents", async () => {
    let reads = 0;
    const text = async () => {
      reads++;
      return source;
    };
    await expect(
      readPythonFile({ name: "model.txt", size: 10, text }),
    ).rejects.toThrow("(.py)");
    await expect(
      readPythonFile({ name: "model.py", size: 500001, text }),
    ).rejects.toThrow("500 KB");
    expect(reads).toBe(0);
    await expect(
      readPythonFile(file("MODEL.PY", source, 500000)),
    ).resolves.toBe(source);
  });
  it("reports empty, binary, and unreadable files without creating a project", async () => {
    await expect(readPythonFile(file("model.py", " \n"))).rejects.toThrow(
      "empty",
    );
    await expect(readPythonFile(file("model.py", "binary\0"))).rejects.toThrow(
      "text Python",
    );
    await expect(
      readPythonFile({
        name: "model.py",
        size: 10,
        text: async () => {
          throw new Error("unavailable");
        },
      }),
    ).rejects.toThrow("Try selecting it again");
  });
});
