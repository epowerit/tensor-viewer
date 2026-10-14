import { expect, test } from "@playwright/test";
import { openRun, recorded, unrunProject } from "./app";

test("the command palette behaves as a modal dialog", async ({ page }) => {
  await openRun(page, "gpt");
  const palette = page.locator(".command-palette");
  const focused = () =>
    page.evaluate(() => document.activeElement?.tagName ?? "");
  await page.keyboard.press("ControlOrMeta+k");
  await expect(palette).toBeVisible();
  expect(await focused()).toBe("INPUT");
  // Tab stays in the search field.
  await page.keyboard.press("Tab");
  await page.keyboard.press("Tab");
  expect(await focused()).toBe("INPUT");
  // Workspace shortcuts wait while it is open.
  const step = () =>
    page.evaluate(
      () => document.body.innerText.match(/STEP\s+([^\n]+)/)?.[1] ?? "",
    );
  const before = await step();
  await page.keyboard.press("F11");
  await page.waitForTimeout(300);
  expect(await step()).toBe(before);
  // The page behind is not inert: focus the workspace moves there comes
  // back, so Escape still reaches the palette.
  await page.evaluate(() =>
    document.querySelector<HTMLElement>(".journey-canvas")?.focus(),
  );
  expect(await focused()).toBe("INPUT");
  await page.keyboard.press("Escape");
  await expect(palette).toHaveCount(0);
  await page.keyboard.press("ControlOrMeta+k");
  await expect(palette).toBeVisible();
  // Its backdrop closes it.
  await page.mouse.click(8, 990);
  await expect(palette).toHaveCount(0);
});

test("canvas line labels follow the code as it is edited", async ({ page }) => {
  await openRun(page, "gpt");
  const labels = () =>
    page.evaluate(() =>
      [...document.querySelectorAll(".canvas-world .node-operation")]
        .map((node) => node.textContent?.match(/L(\d+)/)?.[1])
        .filter(Boolean)
        .map(Number),
    );
  const before = await labels();
  expect(before.length).toBeGreaterThan(3);
  await page.keyboard.press("ControlOrMeta+e");
  const code = page.locator("textarea.code-input");
  await code.waitFor();
  // Two lines above everything move every label down two.
  await code.evaluate((element: HTMLTextAreaElement) => {
    element.focus();
    element.setSelectionRange(0, 0);
    document.execCommand("insertText", false, "\n\n");
  });
  await expect.poll(labels).toEqual(before.map((line) => line + 2));
  // Typing within a line moves none.
  await code.evaluate((element: HTMLTextAreaElement) => {
    element.setSelectionRange(1, 1);
    document.execCommand("insertText", false, "# a note");
  });
  await page.waitForTimeout(500);
  expect(await labels()).toEqual(before.map((line) => line + 2));
});

test("a project opens its input setup only when it has no run", async ({
  page,
}) => {
  const setupOpen = () =>
    page.evaluate(
      () =>
        (document.querySelector(".experiment-disclosure") as HTMLDetailsElement)
          ?.open,
    );
  const never = await unrunProject(page);
  await page.goto(`/#project=${never.id}`);
  await page.reload();
  await page.locator(".experiment-disclosure").waitFor();
  await page.waitForTimeout(1500);
  expect(await setupOpen()).toBe(true);
  // While a project with a run loads, its setup never opens, even briefly.
  await page.addInitScript(() => {
    const seen = window as unknown as { setupOpened?: boolean };
    seen.setupOpened = false;
    document.addEventListener(
      "toggle",
      (event) => {
        if ((event.target as HTMLDetailsElement).open) seen.setupOpened = true;
      },
      true,
    );
  });
  const { project } = recorded("gpt");
  await page.goto(`/#project=${project}`);
  await page.reload();
  await page.waitForSelector(".canvas-world .journey-node");
  await page.waitForTimeout(1000);
  expect(
    await page.evaluate(
      () => (window as unknown as { setupOpened?: boolean }).setupOpened,
    ),
  ).toBe(false);
  expect(await setupOpen()).toBe(false);
});
