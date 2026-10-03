import { expect, test, vi } from "vitest";
import { renamePins } from "./pins";

const store = new Map<string, string>();
vi.stubGlobal("localStorage", {
  getItem: (key: string) => store.get(key) ?? null,
  setItem: (key: string, value: string) => store.set(key, value),
});
vi.stubGlobal("window", new EventTarget());

test("pins follow a renamed tensor, and others stay", () => {
  store.set("tensorviewer.pins.p", JSON.stringify(["logits", "x"]));
  let told = "";
  const listen = (event: Event) => (told = (event as CustomEvent).detail);
  window.addEventListener("tensorviewer:pins", listen);
  renamePins("p", new Map([["logits", "scores"]]));
  window.removeEventListener("tensorviewer:pins", listen);
  expect(JSON.parse(store.get("tensorviewer.pins.p")!)).toEqual([
    "scores",
    "x",
  ]);
  expect(told).toBe("p");
});

test("nothing pinned under an old name leaves pins alone", () => {
  store.set("tensorviewer.pins.q", JSON.stringify(["x"]));
  renamePins("q", new Map([["logits", "scores"]]));
  expect(store.get("tensorviewer.pins.q")).toBe('["x"]');
});
