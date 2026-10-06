import { expect, test } from "vitest";
import { PANEL_GROUPS, PANEL_TABS, panelTab } from "./panelTabs";

test("every panel tab is listed once, in its group", () => {
  const ids = PANEL_TABS.map((tab) => tab.id);
  expect(new Set(ids).size).toBe(ids.length);
  expect(ids).toHaveLength(10);
  expect(PANEL_GROUPS.map((group) => group.name)).toEqual(["Run", "Model"]);
});

test("every tab opens with a title and a sentence on what it shows", () => {
  for (const { id, intro } of PANEL_TABS) {
    expect(intro.title, id).not.toBe("");
    expect(intro.text.endsWith("."), id).toBe(true);
    for (const hint of intro.read ?? [])
      expect(hint.endsWith("."), `${id}: ${hint}`).toBe(false);
  }
  expect(panelTab("lens").intro.title).toBe("What each layer would predict");
});
