import { readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Page } from "@playwright/test";

/** The library models the suite reads, by key. */
export const LIBRARY = {
  gpt: "13 · GPT language model",
  captioner: "24 · Image captioner",
} as const;
export type Model = keyof typeof LIBRARY;
export type Recorded = Partial<Record<Model, { project: string; run: string }>>;

/** Where the setup notes each model's project and recorded run. */
export const RECORDED = join(tmpdir(), "tensorviewer-e2e-recorded.json");

export function recorded(model: Model) {
  const all = JSON.parse(readFileSync(RECORDED, "utf8")) as Recorded;
  return all[model]!;
}

/** Opens a model's recorded run on the canvas with a step selected. */
export async function openRun(page: Page, model: Model, node = "op6") {
  const { project, run } = recorded(model);
  await page.goto(`/#project=${project}&run=${run}&node=${node}`);
  await page.waitForSelector(".canvas-world .journey-node .tensor-volume", {
    timeout: 60_000,
  });
  // Lazily loaded views arrive while the browser is idle.
  await page.waitForTimeout(1500);
}

/** A project of the library that has not been run in this workspace. */
export async function unrunProject(page: Page) {
  const listed = (await (
    await page.request.get("/api/v1/projects")
  ).json()) as {
    id: string;
    name: string;
  }[];
  const ran = new Set(Object.values(LIBRARY) as string[]);
  for (const project of listed) {
    if (ran.has(project.name)) continue;
    const runs = (await (
      await page.request.get(`/api/v1/projects/${project.id}/runs`)
    ).json()) as unknown[];
    if (!runs.length) return project;
  }
  throw new Error("Every library project has a run");
}

/** Milliseconds from now until the browser has painted what `act` did. */
export function untilPainted(page: Page, act: string) {
  return page.evaluate(async (act) => {
    const start = performance.now();
    new Function(act)();
    await new Promise((resolve) =>
      requestAnimationFrame(() => setTimeout(resolve, 0)),
    );
    return performance.now() - start;
  }, act);
}

export const median = (values: number[]) =>
  [...values].sort((a, b) => a - b)[values.length >> 1];

/**
 * What is wrong with the open lesson's cards, read from the page: a card
 * that scrolls on its own or clips what it holds, side-by-side grids that do
 * not start level, cell text wider than its cell, and grid labels that
 * overlap. Empty when the lesson reads whole.
 */
export function lessonProblems(page: Page) {
  return page.evaluate(() => {
    const root = document.querySelector<HTMLElement>(
      ".transformation-focus, .stage-focus",
    );
    if (!root) return ["no lesson open"];
    const title = root.querySelector("h2")?.textContent ?? "?";
    const problems: string[] = [];
    const shown = (element: Element) => element.checkVisibility();
    for (const card of root.querySelectorAll<HTMLElement>(".tensor-card")) {
      if (!shown(card)) continue;
      if (card.scrollHeight > card.clientHeight + 2)
        problems.push(
          `${title}: a card holds ${card.scrollHeight}px in ${card.clientHeight}px`,
        );
    }
    // Grids whose cards overlap in height stand side by side.
    const grids = [...root.querySelectorAll(".tensor-card .tensor-svg")]
      .filter(shown)
      .map((svg) => ({
        card: svg.closest(".tensor-card")!.getBoundingClientRect(),
        top: svg.getBoundingClientRect().top,
      }))
      .sort((a, b) => a.card.top - b.card.top);
    let row: typeof grids = [];
    const level = () => {
      const tops = row.map((grid) => grid.top);
      if (row.length > 1 && Math.max(...tops) - Math.min(...tops) > 1)
        problems.push(
          `${title}: side-by-side grids start at ${tops.map(Math.round).join("/")}`,
        );
    };
    for (const grid of grids) {
      const bottom = Math.max(0, ...row.map((item) => item.card.bottom));
      if (row.length && grid.card.top >= bottom - 1) {
        level();
        row = [];
      }
      row.push(grid);
    }
    level();
    for (const svg of root.querySelectorAll(".tensor-card .tensor-svg")) {
      if (!shown(svg)) continue;
      for (const text of svg.querySelectorAll("svg svg > text")) {
        const box = text.closest("svg")!.getBoundingClientRect();
        if (text.getBoundingClientRect().width > box.width + 0.5)
          problems.push(
            `${title}: '${text.textContent}' is wider than its cell`,
          );
      }
      const labels = [...svg.querySelectorAll("text.cell-header")].map(
        (label) => label.getBoundingClientRect(),
      );
      for (let i = 1; i < labels.length; i++)
        if (
          Math.abs(labels[i - 1].top - labels[i].top) < 1 &&
          labels[i - 1].right > labels[i].left + 0.5
        )
          problems.push(`${title}: grid labels overlap`);
    }
    return [...new Set(problems)];
  });
}
