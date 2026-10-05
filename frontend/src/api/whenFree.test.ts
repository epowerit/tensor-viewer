import { expect, test } from "vitest";
import { whenFree } from "./client";

const noWait = () => Promise.resolve();

test("an analysis waits while another run holds the backend", async () => {
  let asked = 0;
  const found = await whenFree(async () => {
    asked++;
    if (asked < 3)
      throw new Error("A run is already in progress. Wait for it to finish.");
    return "done";
  }, noWait);
  expect(found).toBe("done");
  expect(asked).toBe(3);
});

test("other failures, and a backend busy too long, still fail", async () => {
  await expect(
    whenFree(async () => {
      throw new Error("Run not found");
    }, noWait),
  ).rejects.toThrow("Run not found");
  let asked = 0;
  await expect(
    whenFree(async () => {
      asked++;
      throw new Error("A run is already in progress. Wait for it to finish.");
    }, noWait),
  ).rejects.toThrow("already in progress");
  expect(asked).toBe(13);
});
