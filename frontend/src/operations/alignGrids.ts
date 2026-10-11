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
    .filter((drawing) => !drawing.closest("[hidden]"))
    .flatMap((drawing) => {
      const card = drawing.parentElement!.getBoundingClientRect();
      if (!card.height) return [];
      const drop =
        parseFloat(drawing.style.getPropertyValue("--grid-drop")) || 0;
      // Where the grid would sit without a drop.
      const top = drawing.getBoundingClientRect().top - drop;
      return [{ drawing, card, top }];
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
    for (const { drawing, top } of row) {
      const drop = Math.round(lowest - top);
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
    // Its own drops change only style attributes, which this ignores.
    const changes = new MutationObserver(align);
    changes.observe(element, { childList: true, subtree: true });
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      changes.disconnect();
    };
  }, [root]);
}
