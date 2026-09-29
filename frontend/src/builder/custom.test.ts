import { describe, expect, it } from "vitest";
import { componentTool, parseArguments } from "./custom";
import type { CustomComponent } from "../api/client";

describe("custom component settings", () => {
  it("preserves nested JSON settings and rejects non-objects or non-finite numbers", () => {
    expect(
      parseArguments('{"bias":false,"widths":[2,4],"other":null}'),
    ).toEqual({ bias: false, widths: [2, 4], other: null });
    for (const value of ["[]", "null", "2", "{broken", '{"factor":1e309}'])
      expect(() => parseArguments(value)).toThrow();
  });
  it("uses the project's pinned source even when the library changes", () => {
    const custom: CustomComponent = {
      id: "saved",
      created_at: "today",
      name: "Pinned name",
      description: "Original",
      class_name: "Example",
      code: "original",
      constructor: {},
    };
    const item = componentTool(
      { id: "node", kind: "custom", parameters: {}, custom },
      [
        {
          kind: "custom",
          title: "New name",
          description: "New",
          group: "Custom",
          parameters: [],
        },
      ],
    );
    expect(item?.title).toBe("Pinned name");
    expect(item?.custom?.code).toBe("original");
  });
});

import { draftSignature } from "../api/client";
import { blankProject } from "./model";

it("distinguishes canvas edits from regenerated code after a backend restart", () => {
  const draft = blankProject("Experiment");
  draft.blueprint = {
    has_input: true,
    components: [{ id: "a", kind: "relu", parameters: {} }],
  };
  const regenerated = {
    ...draft,
    code: "a different generated preview",
    blueprint: {
      has_input: true,
      components: [
        {
          id: "a",
          kind: "relu",
          parameters: {},
          custom: null,
          arguments: null,
        },
      ],
    },
  };
  expect(draftSignature(draft)).toBe(draftSignature(regenerated));
  expect(
    draftSignature({ ...draft, input: { ...draft.input, shape: [2, 8, 8] } }),
  ).not.toBe(draftSignature(draft));
  expect(draftSignature({ ...draft, blueprint: null })).not.toBe(
    draftSignature({ ...regenerated, blueprint: null }),
  );
});
