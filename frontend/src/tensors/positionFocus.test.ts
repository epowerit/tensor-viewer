import { afterEach, describe, expect, it } from "vitest";
import {
  focusPosition,
  focusedPosition,
  hoverPosition,
  lockPosition,
  lockedPosition,
} from "./positionFocus";

afterEach(() => {
  focusPosition(null);
  lockPosition(null);
});

describe("position focus", () => {
  it("follows the pointer", () => {
    const handlers = hoverPosition(3, "cat");
    handlers.onMouseEnter();
    expect(focusedPosition()).toBe(3);
    handlers.onMouseLeave();
    expect(focusedPosition()).toBeNull();
  });

  it("keeps a clicked word in focus after the pointer leaves", () => {
    const handlers = hoverPosition(2, "sat");
    handlers.onMouseEnter();
    handlers.onClick();
    handlers.onMouseLeave();
    expect(focusedPosition()).toBe(2);
    expect(lockedPosition()).toEqual({ position: 2, word: "sat" });
  });

  it("lets a hovered word stand in for the locked one", () => {
    lockPosition(2, "sat");
    focusPosition(5);
    expect(focusedPosition()).toBe(5);
    focusPosition(null);
    expect(focusedPosition()).toBe(2);
  });

  it("unlocks on a second click, and moves to another word", () => {
    lockPosition(2, "sat");
    lockPosition(4, "the");
    expect(lockedPosition()?.position).toBe(4);
    hoverPosition(4, "the").onClick();
    expect(lockedPosition()).toBeNull();
  });

  it("names an unlabelled position by its index", () => {
    hoverPosition(7).onClick();
    expect(lockedPosition()).toEqual({ position: 7, word: "7" });
  });
});
