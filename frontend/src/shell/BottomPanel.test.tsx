import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, test } from "vitest";
import type { Run } from "../api/client";
import recorded from "../tensors/fixtures/lineage-trace.json";
import { BottomPanel } from "./BottomPanel";
import { PANEL_TABS, type PanelTab } from "./panelTabs";
import type { Problem } from "./problems";

const next = {
  label: "Run",
  detail: "Run the code",
  run: true,
  busy: false,
  disabled: false,
  onClick: () => {},
};

const panel = (
  tab: PanelTab,
  options: {
    run?: Run | null;
    problems?: Problem[];
    nextStep?: typeof next | null;
  } = {},
) =>
  renderToStaticMarkup(
    <BottomPanel
      tab={tab}
      onTab={() => {}}
      onClose={() => {}}
      run={options.run ?? null}
      problems={options.problems ?? []}
      selected={null}
      onSelect={() => {}}
      onProblem={() => {}}
      nextStep={options.nextStep === undefined ? next : options.nextStep}
    />,
  );

describe("before a run", () => {
  test("every tab says so alike and offers the next action", () => {
    for (const { id, intro } of PANEL_TABS) {
      const html = panel(id);
      expect(html, id).toContain("Nothing has run yet.");
      expect(html, id).toContain("Run the code and this tab fills in.");
      expect(html, id).toContain('class="panel-run"');
      expect(html, id).toContain("⌘↵");
      // Each still opens with what it shows.
      expect(html, id).toContain(intro.title.replace(/'/g, "&#x27;"));
    }
  });

  test("an action other than running is named as it is", () => {
    const html = panel("weights", {
      nextStep: {
        ...next,
        label: "Set inputs",
        detail: "Set the starting tensors",
        run: false,
      },
    });
    expect(html).toContain("Set the starting tensors.");
    expect(html).toContain("Set inputs");
    expect(html).not.toContain("⌘↵");
  });

  test("run notes with notes to show list them", () => {
    const html = panel("problems", {
      problems: [
        {
          id: "input",
          severity: "warning",
          title: "The input needs a value",
          detail: "Set it first.",
          file: null,
          line: null,
          node: null,
          diagnosis: null,
        },
      ],
    });
    expect(html).toContain("The input needs a value");
    expect(html).not.toContain("Nothing has run yet.");
  });
});

test("tabs come in two groups, each tab tied to the panel it controls", () => {
  const html = panel("lens");
  expect(html.indexOf(">Run<")).toBeLessThan(html.indexOf(">Model<"));
  for (const { id } of PANEL_TABS)
    expect(html).toContain(`id="panel-tab-${id}"`);
  expect(html).toContain('aria-controls="panel-pane-main"');
  expect(html).toContain('aria-labelledby="panel-tab-lens"');
  // Only the chosen tab is a stop for the Tab key.
  expect(html.match(/role="tab"[^>]*tabindex="0"/g)).toHaveLength(1);
});

test("a failed run's output ends with its error, then the run's facts", () => {
  const run = {
    id: "run",
    created_at: "2026-10-06T10:00:00Z",
    project: { input: {} },
    trace: {
      ...recorded,
      stdout: "loss 0.5\n",
      duration_ms: 12,
      error: { type: "ValueError", message: "bad shape", line: 3, file: null },
    },
  } as unknown as Run;
  const html = panel("output", { run });
  expect(html).toContain("loss 0.5");
  expect(html).toContain("<b>ValueError</b>: bad shape");
  expect(html.indexOf("loss 0.5")).toBeLessThan(html.indexOf("ValueError"));
  expect(html).toContain("<dt>Steps</dt>");
  expect(html).toContain("<dt>Tensor states</dt>");
});
