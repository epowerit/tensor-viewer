import { expect, test } from "vitest";
import { repositoryName } from "./GitImport";

test("an imported project is named after its repository and folder", () => {
  expect(repositoryName("https://github.com/owner/vision.git")).toBe("vision");
  expect(repositoryName("https://github.com/owner/vision/", "models/vit")).toBe(
    "vision · vit",
  );
  expect(repositoryName("/Users/me/code/nanogpt", ".")).toBe("nanogpt");
  expect(repositoryName("git@github.com:owner/llm.git")).toBe("llm");
  expect(repositoryName("")).toBe("");
});
