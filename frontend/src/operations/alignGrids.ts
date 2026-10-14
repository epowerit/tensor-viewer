import { useLayoutEffect, type RefObject } from "react";

/**
 * Side by side, cards line their grids up, so a row reads across them: a
 * card whose title or axes take an extra line, or with a tensor choice above
 * it, held its grid lower than its neighbours'. Each grid drops (by
 * `--grid-drop`) to the lowest one among the cards beside it; cards stacked
 * one above the other keep theirs.
 */
export function alignGrids(root: HTMLElement) {
  const placed = [
    ...root.querySelectorAll<HTMLElement>(".tensor-card > .tensor-drawing"),
  ]
    // Only grids on screen line up: one in a closed disclosure still has a
    // box beside the others, and aligning to it dropped them out of sight.
    .filter((drawing) => {
      const shown = drawing.checkVisibility();
      if (!shown) drawing.style.removeProperty("--grid-drop");
      return shown;
    })
    .flatMap((drawing) => {
      const card = drawing.parentElement!.getBoundingClientRect();
      if (!card.height) return [];
      const box = drawing.getBoundingClientRect();
      // A lesson fitted to its stage draws its cards zoomed: a drop is set in
      // the card's own pixels, and measured in the screen's.
      const scale = drawing.offsetWidth ? box.width / drawing.offsetWidth : 1;
      const drop =
        parseFloat(drawing.style.getPropertyValue("--grid-drop")) || 0;
      // Where the grid would sit without a drop.
      const top = box.top - drop * scale;
      return [{ drawing, card, top, scale }];
    })
    .sort((a, b) => a.card.top - b.card.top);
  // Cards whose heights overlap stand side by side.
  const rows: (typeof placed)[] = [];
  let bottom = -Infinity;
  for (const item of placed) {
    const row = rows.at(-1);
    if (row && item.card.top < bottom - 1) {
      row.push(item);
      bottom = Math.max(bottom, item.card.bottom);
    } else {
      rows.push([item]);
      bottom = item.card.bottom;
    }
  }
  for (const row of rows) {
    const lowest = Math.max(...row.map((item) => item.top));
    for (const { drawing, top, scale } of row) {
      const drop = Math.round((lowest - top) / scale);
      if (drop > 0) drawing.style.setProperty("--grid-drop", `${drop}px`);
      else drawing.style.removeProperty("--grid-drop");
    }
  }
}

/** Keeps the grids inside `root` lined up as its content and size change. */
export function useAlignedGrids(root: RefObject<HTMLElement | null>) {
  useLayoutEffect(() => {
    const element = root.current;
    if (!element) return;
    let frame = 0;
    const align = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => alignGrids(element));
    };
    align();
    const resize = new ResizeObserver(align);
    resize.observe(element);
    // Its own drops change only style attributes, which this ignores; a
    // disclosure opening or a title rewrapping its new text does not.
    const changes = new MutationObserver(align);
    changes.observe(element, {
      childList: true,
      subtree: true,
      characterData: true,
      attributes: true,
      attributeFilter: ["open", "hidden"],
    });
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      changes.disconnect();
    };
  }, [root]);
}
