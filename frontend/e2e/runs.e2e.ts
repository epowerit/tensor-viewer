import { expect, test, type Page } from "@playwright/test";
import { openRun } from "./app";

// A new run of the project on screen carries on where the last one was: the
// same step selected, the same camera, the inspector still open on it, and
// across an edit the same step wherever it moved.

const view = (page: Page) =>
  page.evaluate(() => {
    const hash = new URLSearchParams(location.hash.slice(1));
    const step = document.body.innerText.match(/STEP\s+([^\n]+)/)?.[1] ?? "";
    return {
      run: hash.get("run"),
      node: hash.get("node"),
      // The step's result and shape, without its line, which edits move.
      step: step.replace(/\s*·\s*line \d+.*$/, ""),
      line: Number(step.match(/line (\d+)/)?.[1] ?? 0),
      inspecting: !!document.querySelector(".transformation-focus"),
      title: document.querySelector(".transformation-focus h2")?.textContent,
      camera: getComputedStyle(document.querySelector(".canvas-world")!)
        .transform,
    };
  });

async function runAgain(page: Page) {
  const before = (await view(page)).run;
  await page.locator("button.run-button").click();
  await page.waitForFunction(
    (before) =>
      new URLSearchParams(location.hash.slice(1)).get("run") !== before,
    before,
    { timeout: 60_000 },
  );
  // The camera settles on the carried view.
  await page.waitForTimeout(1500);
}

test("running unchanged code again keeps the step and the camera", async ({
  page,
}) => {
  await openRun(page, "gpt");
  const before = await view(page);
  await runAgain(page);
  const after = await view(page);
  expect(after.run).not.toBe(before.run);
  expect(after).toMatchObject({
    node: before.node,
    step: before.step,
    camera: before.camera,
    inspecting: false,
  });
});

test("the inspector stays open on its step through a run", async ({ page }) => {
  await openRun(page, "gpt");
  await page.getByRole("button", { name: /^Inspect (current|pass)/ }).click();
  await page.locator(".transformation-focus").waitFor();
  const before = await view(page);
  await runAgain(page);
  const after = await view(page);
  expect(after).toMatchObject({
    node: before.node,
    step: before.step,
    inspecting: true,
    title: before.title,
  });
});

test("an edit that adds a step keeps the same step selected", async ({
  page,
}) => {
  await openRun(page, "gpt");
  const before = await view(page);
  await page.keyboard.press("ControlOrMeta+e");
  const code = page.locator("textarea.code-input");
  await code.waitFor();
  // One more step right after the embeddings, before the selected one.
  await code.evaluate((element: HTMLTextAreaElement) => {
    const anchor = "self.position(positions)";
    const end = element.value.indexOf("\n", element.value.indexOf(anchor));
    element.focus();
    element.setSelectionRange(end, end);
    document.execCommand("insertText", false, "\n        x = x * 1.0");
  });
  await runAgain(page);
  const after = await view(page);
  expect(after.step).toBe(before.step);
  // Its line sits above the edit, in the attention, and stays put.
  expect(after.line).toBe(before.line);
  expect(after.node).toBe(`op${Number(before.node!.slice(2)) + 1}`);
});

test("an edit's run says what changed", async ({ page }) => {
  await openRun(page, "gpt");
  await page.keyboard.press("ControlOrMeta+e");
  const code = page.locator("textarea.code-input");
  await code.waitFor();
  await code.evaluate((element: HTMLTextAreaElement) => {
    const anchor = "self.position(positions)";
    const end = element.value.indexOf("\n", element.value.indexOf(anchor));
    element.focus();
    element.setSelectionRange(end, end);
    document.execCommand("insertText", false, "\n        x = x * 1.0");
  });
  await page.keyboard.press("ControlOrMeta+Enter");
  await expect(page.getByText(/Diagram updated · /)).toBeVisible({
    timeout: 60_000,
  });
});

test("folds and the lens carry into the next run", async ({ page }) => {
  await openRun(page, "gpt");
  // Fold the call around the selected step, and colour by spread.
  await page.locator(".journey-canvas").click({ position: { x: 5, y: 5 } });
  await page.keyboard.press("ControlOrMeta+Alt+BracketLeft");
  await page.locator(".flow-lens-select").selectOption("spread");
  await page.waitForTimeout(500);
  const folded = () =>
    page.evaluate(
      () =>
        document.querySelectorAll(".canvas-world .journey-stage-node").length,
    );
  const before = await folded();
  expect(before).toBeGreaterThan(0);
  await runAgain(page);
  expect(await folded()).toBe(before);
  await expect(page.locator(".flow-lens-select")).toHaveValue("spread");
});
