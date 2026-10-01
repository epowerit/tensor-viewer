import { describe, expect, it } from "vitest";
import type { ComponentSpec, CustomComponent } from "../api/client";
import {
  constructorEdit,
  currentConstructorEdit,
  reconcileConstructorEdits,
} from "./constructorEdits";

const source: CustomComponent = {
  id: "scale-v1",
  created_at: "today",
  name: "Scale",
  description: "Scale features",
  class_name: "Scale",
  code: "class Scale: pass",
  constructor: { scale: 2 },
};

function component(id: string): ComponentSpec {
  return { id, kind: "custom", parameters: {}, custom: source };
}

describe("constructor editing across model navigation", () => {
  it("auto-applies valid JSON without replacing the user's formatting", () => {
    const node = component("first");
    const text = '{ "scale": 3, "options": [true, null] }';
    const result = constructorEdit(node, text);
    expect(result.value).toEqual({ scale: 3, options: [true, null] });

    const applied = { ...node, arguments: result.value };
    const edits = { [node.id]: result.edit };
    expect(currentConstructorEdit(edits, applied)?.text).toBe(text);
    expect(currentConstructorEdit(edits, applied)?.error).toBe("");
    expect(reconcileConstructorEdits(edits, [applied])).toBe(edits);
  });

  it("retains unfinished text separately for two instances of the same source", () => {
    const first = component("first");
    const second = component("second");
    const edits = {
      first: constructorEdit(first, '{"scale":').edit,
      second: constructorEdit(second, '{"scale": 7,').edit,
    };

    // Component order and selection do not determine which draft survives.
    const afterNavigation = reconcileConstructorEdits(edits, [second, first]);
    expect(currentConstructorEdit(afterNavigation, first)?.text).toBe(
      '{"scale":',
    );
    expect(currentConstructorEdit(afterNavigation, second)?.text).toBe(
      '{"scale": 7,',
    );
    expect(currentConstructorEdit(afterNavigation, first)?.error).toBeTruthy();
    expect(currentConstructorEdit(afterNavigation, second)?.error).toBeTruthy();
  });

  it("keeps the last valid arguments when the next edit is incomplete", () => {
    const node = { ...component("first"), arguments: { scale: 5 } };
    const result = constructorEdit(node, '{"scale": 5,');

    expect(result.value).toBeUndefined();
    expect(result.edit.error).toBeTruthy();
    expect(currentConstructorEdit({ first: result.edit }, node)).toBe(
      result.edit,
    );
    expect(node.arguments).toEqual({ scale: 5 });
  });

  it.each(["[]", "null", '{"scale":1e309}', "{unfinished"])(
    "retains invalid %s for correction without applying it",
    (text) => {
      const result = constructorEdit(component("first"), text);
      expect(result.value).toBeUndefined();
      expect(result.edit.text).toBe(text);
      expect(result.edit.error).toBeTruthy();
    },
  );

  it("removes pending errors for deleted or replaced components", () => {
    const first = component("first");
    const second = component("second");
    const edits = {
      first: constructorEdit(first, "{").edit,
      second: constructorEdit(second, "{").edit,
    };
    expect(reconcileConstructorEdits(edits, [second])).toEqual({
      second: edits.second,
    });

    const newVersion = { ...first, custom: { ...source, id: "scale-v2" } };
    expect(currentConstructorEdit(edits, newVersion)).toBeUndefined();
    expect(reconcileConstructorEdits(edits, [newVersion, second])).toEqual({
      second: edits.second,
    });
    const builtIn = { id: first.id, kind: "relu", parameters: {} };
    expect(reconcileConstructorEdits(edits, [builtIn])).toEqual({});
  });

  it("does not restore obsolete pending text over externally replaced arguments", () => {
    const node = component("first");
    const edits = { first: constructorEdit(node, "{").edit };
    const replacement = { ...node, arguments: { scale: 9 } };
    expect(currentConstructorEdit(edits, replacement)).toBeUndefined();
    expect(reconcileConstructorEdits(edits, [replacement])).toEqual({});
  });
});
