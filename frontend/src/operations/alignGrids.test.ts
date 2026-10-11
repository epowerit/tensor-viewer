import { expect, test } from "vitest";
import { alignGrids } from "./alignGrids";

/** A card spanning `top`–`bottom` whose grid starts at `grid`, undropped. */
function card(top: number, bottom: number, grid: number) {
  const style = new Map<string, string>();
  const drop = () => parseFloat(style.get("--grid-drop") ?? "") || 0;
  return {
    closest: () => null,
    parentElement: {
      getBoundingClientRect: () => ({ top, bottom, height: bottom - top }),
    },
    getBoundingClientRect: () => ({ top: grid + drop() }),
    style: {
      getPropertyValue: (name: string) => style.get(name) ?? "",
      setProperty: (name: string, value: string) => style.set(name, value),
      removeProperty: (name: string) => style.delete(name),
    },
  };
}
const drops = (...cards: ReturnType<typeof card>[]) => {
  alignGrids({ querySelectorAll: () => cards } as unknown as HTMLElement);
  return cards.map((each) => each.style.getPropertyValue("--grid-drop"));
};

test("side by side, a grid drops to its neighbour's", () => {
  const left = card(0, 500, 100),
    right = card(0, 500, 140);
  expect(drops(left, right)).toEqual(["40px", ""]);
  // Measured again with the drop in place, nothing moves.
  expect(drops(left, right)).toEqual(["40px", ""]);
});

test("cards starting lower still line up with the cards beside them", () => {
  expect(drops(card(0, 500, 100), card(30, 530, 150))).toEqual(["50px", ""]);
});

test("stacked cards keep their own grids", () => {
  expect(drops(card(0, 500, 100), card(520, 1000, 640))).toEqual(["", ""]);
});
