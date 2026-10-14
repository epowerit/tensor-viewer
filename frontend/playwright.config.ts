import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { defineConfig } from "@playwright/test";

// The UI health suite: the app's own interface checked in a real browser,
// against its own backend with a fresh workspace and the library installed.
// Build first (`npm run build`); the suite serves that build.
const api = Number(process.env.E2E_API_PORT) || 8020;
const web = Number(process.env.E2E_WEB_PORT) || 5320;
const data = mkdtempSync(join(tmpdir(), "tensorviewer-e2e-"));

export default defineConfig({
  testDir: "e2e",
  testMatch: "*.e2e.ts",
  globalSetup: "./e2e/setup.ts",
  // One browser at a time: timings are part of what the suite checks.
  workers: 1,
  fullyParallel: false,
  timeout: 180_000,
  expect: { timeout: 15_000 },
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${web}`,
    viewport: { width: 1100, height: 1000 },
    headless: true,
  },
  webServer: [
    {
      command: `cd ../backend && uv run uvicorn tensorviewer.app:app --host 127.0.0.1 --port ${api}`,
      url: `http://127.0.0.1:${api}/api/v1/health`,
      env: { TENSORVIEWER_DATA_DIR: data },
      timeout: 180_000,
      reuseExistingServer: false,
    },
    {
      command: `npx vite preview --host 127.0.0.1 --port ${web} --strictPort`,
      url: `http://127.0.0.1:${web}`,
      env: { TENSORVIEWER_API_URL: `http://127.0.0.1:${api}` },
      timeout: 60_000,
      reuseExistingServer: false,
    },
  ],
});
