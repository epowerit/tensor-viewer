import { expect, test } from "@playwright/test";
import { lessonProblems, openRun, type Model } from "./app";

// Every step of each model, with the inspector open, at a narrow, a laptop
// and a wide window: the lesson stays open from the first step to the last
// (a loop's repeats included) and every card reads whole.
for (const model of ["gpt", "captioner"] satisfies Model[])
  for (const width of [700, 1100, 1600])
    test(`${model}: every lesson reads whole at ${width}px`, async ({
      page,
    }) => {
      await page.setViewportSize({ width, height: 1000 });
      await openRun(page, model);
      await page
        .getByRole("button", { name: /^Inspect (current|pass)/ })
        .click();
      const lesson = page.locator(".transformation-focus, .stage-focus");
      await lesson.waitFor();
      const previous = page.getByRole("button", { name: "Previous operation" });
      const next = page.getByRole("button", { name: "Next operation" });
      while (await previous.isEnabled()) await previous.click();
      const problems: string[] = [];
      let steps = 0;
      for (;;) {
        // Grids line up in the frame after a lesson draws.
        await page.waitForTimeout(250);
        steps++;
        if (!(await lesson.count())) {
          problems.push(`step ${steps}: stepping left the lesson`);
          break;
        }
        problems.push(...(await lessonProblems(page)));
        if (!(await next.isEnabled())) break;
        await next.click();
      }
      expect(steps).toBeGreaterThan(10);
      expect([...new Set(problems)]).toEqual([]);
    });
