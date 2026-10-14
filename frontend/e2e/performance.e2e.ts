import { expect, test } from "@playwright/test";
import { median, openRun, recorded, untilPainted } from "./app";

/**
 * How long the interactions people repeat take on GPT, from the action to
 * the frame that shows it (medians). Budgets are several times the times on
 * a laptop, so a shared CI machine passes: they catch a regression that
 * makes something many times slower, such as a remount or a page-wide
 * restyle per keystroke. Every run prints the table.
 */
const BUDGET_MS = {
  "startup to the canvas": 4000,
  "open the inspector": 600,
  "step with the inspector open": 500,
  "close the inspector": 600,
  "one keystroke in the editor": 60,
  "open the command palette": 500,
  "Run to the new run on screen": 3000,
};
type Measure = keyof typeof BUDGET_MS;

test("interactions stay within their budgets", async ({ page }) => {
  const times = {} as Record<Measure, number>;
  const { project, run } = recorded("gpt");

  let started = Date.now();
  await page.goto(`/#project=${project}&run=${run}&node=op6`);
  await page.waitForSelector(".canvas-world .journey-node .tensor-volume");
  times["startup to the canvas"] = Date.now() - started;
  await openRun(page, "gpt");

  const press = (label: string) =>
    untilPainted(
      page,
      `document.querySelector('button[aria-label^="${label}"]').click()`,
    );
  const open = [],
    close = [],
    step = [];
  for (let i = 0; i < 5; i++) {
    open.push(await press("Inspect current"));
    await page.waitForTimeout(500);
    step.push(await press("Next operation"));
    await page.waitForTimeout(300);
    step.push(await press("Previous operation"));
    await page.waitForTimeout(300);
    close.push(await press("Return to model canvas"));
    await page.waitForTimeout(600);
  }
  times["open the inspector"] = median(open);
  times["step with the inspector open"] = median(step);
  times["close the inspector"] = median(close);

  await page.keyboard.press("ControlOrMeta+e");
  await page.locator("textarea.code-input").waitFor();
  times["one keystroke in the editor"] = await page.evaluate(async () => {
    const code = document.querySelector<HTMLTextAreaElement>(
      "textarea.code-input",
    )!;
    code.focus();
    const middle = code.value.indexOf("\n", code.value.length >> 1);
    code.setSelectionRange(middle, middle);
    const keys: number[] = [];
    for (const key of "  # note\b\b\b\b\b\b\b\b") {
      const start = performance.now();
      if (key === "\b") document.execCommand("delete");
      else document.execCommand("insertText", false, key);
      void document.body.offsetHeight;
      keys.push(performance.now() - start);
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    return keys.sort((a, b) => a - b)[keys.length >> 1];
  });
  await page.keyboard.press("ControlOrMeta+e");

  const palette = [];
  for (let i = 0; i < 4; i++) {
    started = Date.now();
    await page.keyboard.press("ControlOrMeta+k");
    await page.locator(".command-palette").waitFor();
    await untilPainted(page, "");
    palette.push(Date.now() - started);
    await page.keyboard.press("Escape");
    await page.waitForTimeout(300);
  }
  times["open the command palette"] = median(palette);

  const runs = [];
  for (let i = 0; i < 3; i++) {
    const before = page.url();
    started = Date.now();
    await page.locator("button.run-button").click();
    await page.waitForFunction(
      (before) => location.href !== before && /run=/.test(location.hash),
      before,
    );
    await untilPainted(page, "");
    runs.push(Date.now() - started);
    await page.waitForTimeout(1200);
  }
  times["Run to the new run on screen"] = median(runs);

  const table = (Object.keys(BUDGET_MS) as Measure[]).map((measure) => ({
    measure,
    ms: Math.round(times[measure]),
    budget: BUDGET_MS[measure],
  }));
  console.table(table);
  test.info().annotations.push({
    type: "timings",
    description: table.map((row) => `${row.measure} ${row.ms} ms`).join(" · "),
  });
  for (const row of table)
    expect(row.ms, `${row.measure} took ${row.ms} ms`).toBeLessThanOrEqual(
      row.budget,
    );
});
