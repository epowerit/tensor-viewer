import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { Calculator, Grid2X2 } from "lucide-react";
import { FocusSlot } from "../journey/FocusSlot";

// The two halves every dedicated lesson shares: tensors to explore, and the
// working that explains a selected cell.
const CELLS = [
  ".normalization-tensors",
  ".normalization-parameter-tensors",
  ".linear-tensors",
  ".assembly-tensors",
  ".pooling-tensors",
  ".spatial-group-tensors",
  ".window-score-tensors",
].join(", ");
const WORKING = [
  ".linear-calculation",
  ".assembly-readout",
  ".window-score-calculation",
].join(", ");
const ZONES = `${CELLS}, ${WORKING}`;
// Never shrink text below this: beyond it, a zone scrolls instead.
const SMALLEST = 0.8;

/** Scale a box's children so they fit its height, within limits. */
function fitHeight(box: HTMLElement) {
  box.dataset.fit = "";
  box.style.setProperty("--lesson-fit", "1");
  const ratio = box.clientHeight / box.scrollHeight;
  if (ratio < 1 && box.clientHeight > 0)
    box.style.setProperty(
      "--lesson-fit",
      String(Math.max(SMALLEST, Math.floor(ratio * 100) / 100)),
    );
}
function clearFit(box: HTMLElement) {
  delete box.dataset.fit;
  box.style.removeProperty("--lesson-fit");
}

const visible = (root: HTMLElement, selector: string) =>
  [...root.querySelectorAll<HTMLElement>(selector)].some(
    (element) => !element.closest("[hidden]"),
  );

/**
 * In the focused view, a lesson's cells and its working take turns on one
 * stage instead of stacking, so neither has to be scrolled to. The selection
 * is shared, so switching lenses keeps the same cell in view.
 */
export function LessonStage({ children }: { children: ReactNode }) {
  const root = useRef<HTMLDivElement>(null);
  const [split, setSplit] = useState(false);
  const [lens, setLens] = useState<"cells" | "working">("cells");
  useLayoutEffect(() => {
    const element = root.current;
    if (!element) return;
    // Only the stage layout (zones beside an intro column) is fitted.
    const fit = () => {
      const lesson = [
        ...element.querySelectorAll<HTMLElement>(":scope > div > section"),
      ].find((section) => !section.closest("[hidden]"));
      const zones = lesson
        ? [...lesson.querySelectorAll<HTMLElement>(`:scope > :is(${ZONES})`)]
        : [];
      const staged = zones.some(
        (zone) => getComputedStyle(zone).position === "absolute",
      );
      for (const box of [lesson, ...zones]) {
        if (!box) continue;
        if (staged && getComputedStyle(box).display !== "none") fitHeight(box);
        else clearFit(box);
      }
    };
    const check = () => {
      setSplit(visible(element, CELLS) && visible(element, WORKING));
      requestAnimationFrame(fit);
    };
    check();
    const resize = new ResizeObserver(() => requestAnimationFrame(fit));
    resize.observe(element);
    // Lessons load lazily and swap with Tensor details.
    const observer = new MutationObserver(check);
    observer.observe(element, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["hidden", "data-lens"],
    });
    return () => {
      observer.disconnect();
      resize.disconnect();
    };
  }, []);

  return (
    <div
      ref={root}
      className="lesson-stage"
      data-lens={split ? lens : undefined}
    >
      {split && (
        <FocusSlot>
          <div
            className="lesson-lenses scene-lenses"
            role="radiogroup"
            aria-label="Lesson view"
          >
            <button
              role="radio"
              aria-checked={lens === "cells"}
              title="Explore the tensors cell by cell"
              onClick={() => setLens("cells")}
            >
              <Grid2X2 size={13} /> <span className="slot-label">Cells</span>
            </button>
            <button
              role="radio"
              aria-checked={lens === "working"}
              title="See how the selected cell is computed"
              onClick={() => setLens("working")}
            >
              <Calculator size={13} />{" "}
              <span className="slot-label">Working</span>
            </button>
          </div>
        </FocusSlot>
      )}
      {children}
    </div>
  );
}
