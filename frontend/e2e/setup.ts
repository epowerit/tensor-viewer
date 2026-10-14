import { writeFileSync } from "node:fs";
import { chromium, type FullConfig } from "@playwright/test";
import { LIBRARY, RECORDED, type Model, type Recorded } from "./app";

/**
 * A fresh workspace installs the library on first open. Record one run of
 * each model the specs read, through the app's own Run button, and note
 * where each project and run is.
 */
export default async function setup(config: FullConfig) {
  const baseURL = config.projects[0].use.baseURL!;
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 1100, height: 900 },
  });
  await page.goto(baseURL);
  const projects = async () =>
    (await (await page.request.get(`${baseURL}/api/v1/projects`)).json()) as {
      id: string;
      name: string;
    }[];
  // The first open installs the library.
  let listed = await projects();
  for (let i = 0; i < 60 && listed.length < 20; i++) {
    await page.waitForTimeout(500);
    listed = await projects();
  }
  const recorded: Recorded = {};
  for (const [key, name] of Object.entries(LIBRARY) as [Model, string][]) {
    const project = listed.find((item) => item.name === name);
    if (!project) throw new Error(`The library has no project named ${name}`);
    await page.goto(`${baseURL}/#project=${project.id}`);
    await page.reload();
    const run = page.locator("button.run-button");
    await run.waitFor();
    await run.click();
    await page.waitForFunction(() => /run=/.test(location.hash), null, {
      timeout: 120_000,
    });
    const id = new URLSearchParams(page.url().split("#")[1]).get("run")!;
    recorded[key] = { project: project.id, run: id };
  }
  writeFileSync(RECORDED, JSON.stringify(recorded));
  await browser.close();
}
