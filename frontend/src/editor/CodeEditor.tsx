import { useContext, useEffect, useMemo, useRef, useState } from "react";
import { CircleAlert, Repeat } from "lucide-react";
import type { LoopLine } from "../journey/loops";
import type { Tensor } from "../api/client";
import type { LineResult } from "../console/script";
import {
  applyCompletion,
  completionsAt,
  contractCompletionsAt,
  wordAt,
  type Completion,
} from "./completions";
import { lineSelection } from "../sources/editorNavigation";
import { caretPosition, highlightLine, visualWidth } from "./highlight";
import { ShapeGlyph } from "./ShapeGlyph";
import { TensorPeek } from "./TensorPeek";
import type { AxisStory } from "../tensors/axisLineage";
import { AxisInkContext, InkShape, SymbolicContext } from "../tensors/InkShape";

type Props = {
  value: string;
  label: string;
  readOnly?: boolean;
  placeholder?: string;
  /** Recorded results per 1-based line. */
  results: Map<number, LineResult>;
  /** Line of the step shown on the canvas. */
  activeLine: number | null;
  /** Lines of the card under the pointer on the canvas. */
  hoverLines?: ReadonlySet<number> | null;
  /** Lines whose steps the last save added, edited, or reshaped. */
  changedLines?: ReadonlySet<number> | null;
  /** Names that hold tensors in the displayed run. */
  tensors: ReadonlySet<string>;
  /** Shapes known from the displayed run, for completion previews. */
  shapes?: Record<string, number[]>;
  /** The most recent variable, used as the argument of suggested functions. */
  latest?: string;
  /** Recorded tensors by name, previewed when the pointer rests on a name. */
  peekTensors?: ReadonlyMap<string, Tensor>;
  /** Where a previewed tensor's axes come from, traced through the run. */
  lineageFor?: (tensor: Tensor) => AxisStory[] | null;
  /** Tensor-insight warnings per 1-based line, shown as lint marks. */
  lints?: ReadonlyMap<number, string[]>;
  /** Checked `# shape:` contracts per 1-based line. */
  contracts?: ReadonlyMap<number, { ok: boolean; message: string }>;
  /** Recorded loops by header line. */
  loops?: ReadonlyMap<number, LoopLine>;
  /**
   * How far playback has activated the run (an operation index). Inlays of
   * lines whose steps have not run yet stay unlit, like the canvas.
   */
  activatedThrough?: number;
  /** Lines where playback pauses (1-based). */
  breakpoints?: ReadonlySet<number>;
  /** Toggle a breakpoint; F9 toggles the caret's line. */
  onBreakpoint?: (line: number) => void;
  /** The loop whose repeats are playing on the canvas. */
  activeLoop?: string | null;
  onLoop?: (loop: LoopLine) => void;
  onChange: (value: string) => void;
  onRun: () => void;
  /** Ctrl/⌘ + Shift + Enter: check shapes without recording a run. */
  onCheck?: () => void;
  onSelectLine: (line: number) => void;
  onCursor?: (line: number, column: number) => void;
  /** The line under the pointer, so the canvas can trace its step. */
  onHoverLine?: (line: number | null) => void;
  /** Place the caret on a line, such as the one a run stopped on; a new request repeats it. */
  reveal?: { line: number | null; request: number } | null;
};

const LINE = 21;

/**
 * A code editor whose margin is the run: every line carries the tensor it
 * produced as an inlay, and hovering one previews the recorded values.
 */
export function CodeEditor({
  value,
  label,
  readOnly = false,
  placeholder,
  results,
  activeLine,
  changedLines,
  hoverLines,
  tensors,
  shapes = {},
  latest = "x",
  peekTensors,
  lineageFor,
  lints,
  contracts,
  loops,
  activeLoop,
  onLoop,
  breakpoints,
  onBreakpoint,
  activatedThrough,
  onChange,
  onRun,
  onCheck,
  onSelectLine,
  onCursor,
  onHoverLine,
  reveal,
}: Props) {
  const input = useRef<HTMLTextAreaElement>(null);
  const view = useRef<HTMLDivElement>(null);
  const inkFor = useContext(AxisInkContext);
  const symbolicOf = useContext(SymbolicContext);
  const [cursorLine, setCursorLine] = useState<number | null>(null);
  const [peek, setPeek] = useState<{
    tensor: Tensor;
    left: number;
    top: number;
  } | null>(null);
  const scrolledTo = useRef(0);
  const [completion, setCompletion] = useState<{
    from: number;
    to: number;
    caret: number;
    items: Completion[];
    index: number;
    left: number;
    top: number;
  } | null>(null);
  const measure = useRef<HTMLSpanElement>(null);
  const hoverTimer = useRef(0);
  const hoveredLine = useRef<number | null>(null);
  useEffect(() => () => clearTimeout(hoverTimer.current), []);
  const lines = useMemo(() => value.split("\n"), [value]);
  const highlighted = useMemo(
    () => lines.map((line) => highlightLine(line, tensors)),
    [lines, tensors],
  );
  const widths = useMemo(() => lines.map(visualWidth), [lines]);
  const widest = Math.max(12, ...widths);
  useEffect(() => {
    const element = input.current;
    if (!reveal || !element) return;
    element.focus({ preventScroll: true });
    const range = lineSelection(value, reveal.line);
    if (!range) return;
    element.setSelectionRange(range.start, range.end);
    reportCursor();
    const container = view.current;
    if (container)
      container.scrollTop = Math.max(
        0,
        (reveal.line! - 1) * LINE - container.clientHeight / 3,
      );
  }, [reveal?.request]);

  // Playback brings the line in focus into view, unless you are typing here.
  useEffect(() => {
    const container = view.current;
    if (
      activeLine === null ||
      !container ||
      document.activeElement === input.current
    )
      return;
    const top = (activeLine - 1) * LINE;
    if (
      top < container.scrollTop ||
      top + LINE > container.scrollTop + container.clientHeight
    )
      container.scrollTop = Math.max(0, top - container.clientHeight / 3);
  }, [activeLine]);

  function reportCursor() {
    const element = input.current;
    if (!element) return;
    const position = caretPosition(value, element.selectionStart);
    setCursorLine(position.line);
    onCursor?.(position.line, position.column);
    // The list describes the prefix at this caret, not another selection.
    setCompletion((open) =>
      open &&
      (element.selectionStart !== open.caret ||
        element.selectionEnd !== open.caret)
        ? null
        : open,
    );
  }
  function accept(item: Completion) {
    const element = input.current;
    if (!element || !completion) return;
    setCompletion(null);
    const next = applyCompletion(value, completion, item);
    edit(next.value, next.caret);
  }
  function placePeek(
    tensor: Tensor,
    box: { left: number; top: number; bottom: number },
  ) {
    setPeek({
      tensor,
      left: Math.max(8, Math.min(box.left, window.innerWidth - 300)),
      // The card is about 330px tall with its histogram; it opens above
      // the line when there is no room below.
      top:
        box.bottom + 340 > window.innerHeight
          ? Math.max(8, box.top - 326)
          : box.bottom + 6,
    });
  }
  function edit(next: string, caret: number) {
    onChange(next);
    requestAnimationFrame(() => {
      input.current?.setSelectionRange(caret, caret);
      reportCursor();
    });
  }
  /** Screen position just below the character at `offset` of `text`. */
  function below(text: string, offset: number) {
    const box = input.current!.getBoundingClientRect();
    const position = caretPosition(text, offset);
    const line = text.split("\n")[position.line - 1] ?? "";
    const character = measure.current
      ? measure.current.getBoundingClientRect().width / 10
      : 7.5;
    return {
      left: Math.max(
        8,
        Math.min(
          box.left +
            4 +
            visualWidth(line.slice(0, position.column - 1)) * character,
          window.innerWidth - 400,
        ),
      ),
      top: box.top + position.line * LINE + 2,
    };
  }

  return (
    <div
      ref={view}
      className="code-view"
      data-readonly={readOnly || undefined}
      onScroll={(event) => {
        // Typing at the end of a long line scrolls sideways to the caret; only
        // a vertical scroll moves the list away from its line.
        const top = event.currentTarget.scrollTop;
        if (top !== scrolledTo.current) setCompletion(null);
        scrolledTo.current = top;
      }}
    >
      <div
        className="code-content"
        style={{ minWidth: `calc(${widest + 2}ch + 52px)` }}
      >
        {cursorLine !== null && cursorLine <= lines.length && (
          <div
            className="code-cursor-line"
            style={{ top: 10 + (cursorLine - 1) * LINE }}
            aria-hidden="true"
          />
        )}
        {[...(hoverLines ?? [])]
          .filter((line) => line <= lines.length)
          .map((line) => (
            <div
              key={`hover-${line}`}
              className="code-hover-line"
              style={{ top: 10 + (line - 1) * LINE }}
              aria-hidden="true"
            />
          ))}
        {[...(changedLines ?? [])]
          .filter((line) => line <= lines.length)
          .map((line) => (
            <div
              key={`changed-${line}`}
              className="code-changed-line"
              style={{ top: 10 + (line - 1) * LINE }}
              aria-hidden="true"
            />
          ))}
        {activeLine !== null && activeLine <= lines.length && (
          <div
            className="code-active-line"
            style={{ top: 10 + (activeLine - 1) * LINE }}
            aria-hidden="true"
          />
        )}
        <div className="code-gutter">
          {onBreakpoint &&
            lines.map((line, i) =>
              line.trim() || breakpoints?.has(i + 1) ? (
                <button
                  key={`breakpoint-${i}`}
                  className={`code-breakpoint ${breakpoints?.has(i + 1) ? "set" : ""}`}
                  style={{ top: i * LINE }}
                  tabIndex={-1}
                  aria-pressed={breakpoints?.has(i + 1) ?? false}
                  aria-label={`${breakpoints?.has(i + 1) ? "Remove" : "Set"} breakpoint on line ${i + 1}`}
                  title={
                    breakpoints?.has(i + 1)
                      ? "Playback pauses at this line's steps. Select to remove (F9)."
                      : "Pause playback at this line's steps (F9)"
                  }
                  onClick={() => onBreakpoint(i + 1)}
                />
              ) : null,
            )}
          {lines.map((_, i) => {
            const result = results.get(i + 1);
            const steps = result?.operations.length ?? 0;
            const notes = lints?.get(i + 1);
            const linted = notes?.length ? "linted" : "";
            const noteText = notes?.length ? `\n${notes.join("\n")}` : "";
            return steps ? (
              <button
                key={i}
                className={`${result!.error ? "failed" : ""} ${result!.fresh ? "" : "stale"} ${linted}`}
                aria-label={`Line ${i + 1}: show its ${steps === 1 ? "step" : `${steps} steps`}${notes?.length ? `. ${notes.join(". ")}` : ""}`}
                title={`${steps === 1 ? "1 step" : `${steps} steps`} recorded on this line${noteText}`}
                onClick={() => onSelectLine(i + 1)}
              >
                {i + 1}
              </button>
            ) : (
              <span
                key={i}
                className={`${result?.error ? "failed" : ""} ${linted}`}
                title={notes?.length ? notes.join("\n") : undefined}
              >
                {i + 1}
              </span>
            );
          })}
        </div>
        <div className="code-text">
          <pre className="code-highlight" aria-hidden="true">
            {highlighted.map((tokens, i) => (
              <div
                key={i}
                className={
                  results.get(i + 1)?.error && results.get(i + 1)?.fresh
                    ? "code-error-line"
                    : lints?.get(i + 1)?.length
                      ? "code-lint-line"
                      : ""
                }
              >
                {tokens.length
                  ? tokens.map((token, j) => (
                      <span key={j} className={`tok-${token.kind}`}>
                        {token.text}
                      </span>
                    ))
                  : " "}
              </div>
            ))}
          </pre>
          <textarea
            ref={input}
            className="code-input"
            aria-label={label}
            spellCheck={false}
            autoCapitalize="off"
            autoCorrect="off"
            wrap="off"
            readOnly={readOnly}
            placeholder={placeholder}
            value={value}
            style={{ height: lines.length * LINE }}
            onChange={(event) => {
              const next = event.target.value;
              onChange(next);
              setPeek(null);
              const caret = event.target.selectionStart;
              // After code, a comment being started offers contract clauses
              // from the line's recorded tensor.
              const found = readOnly
                ? null
                : (completionsAt(next, caret, shapes, latest) ??
                  contractCompletionsAt(next, caret, (line) => {
                    const result = results.get(line);
                    return result?.fresh && !result.error && !result.predicted
                      ? result.output
                      : null;
                  }));
              setCompletion(
                found && {
                  ...found,
                  caret: event.target.selectionStart,
                  index: 0,
                  ...below(next, found.from),
                },
              );
            }}
            onSelect={reportCursor}
            onBlur={() => {
              setCursorLine(null);
              setCompletion(null);
              setPeek(null);
            }}
            onMouseMove={(event) => {
              clearTimeout(hoverTimer.current);
              const box = event.currentTarget.getBoundingClientRect();
              const row = Math.floor((event.clientY - box.top) / LINE);
              if (row + 1 !== hoveredLine.current) {
                hoveredLine.current = row + 1;
                onHoverLine?.(row + 1);
              }
              if (!peekTensors?.size || !measure.current) return;
              const column = Math.floor(
                (event.clientX - box.left - 4) /
                  (measure.current.getBoundingClientRect().width / 10),
              );
              const word = wordAt(lines[row] ?? "", column);
              const tensor = word ? peekTensors.get(word) : undefined;
              if (!tensor) {
                setPeek(null);
                return;
              }
              const point = {
                left: event.clientX,
                top: box.top + row * LINE,
                bottom: box.top + (row + 1) * LINE,
              };
              hoverTimer.current = window.setTimeout(
                () => placePeek(tensor, point),
                350,
              );
            }}
            onMouseLeave={() => {
              clearTimeout(hoverTimer.current);
              setPeek(null);
              hoveredLine.current = null;
              onHoverLine?.(null);
            }}
            onKeyDown={(event) => {
              const element = event.currentTarget;
              const { selectionStart: start, selectionEnd: end } = element;
              // Ctrl/⌘ + I peeks at the tensor named at the caret, as a
              // hover would; Escape closes it.
              if (
                (event.metaKey || event.ctrlKey) &&
                event.key.toLowerCase() === "i"
              ) {
                const position = caretPosition(value, start);
                const line = lines[position.line - 1] ?? "";
                const word =
                  wordAt(line, position.column - 1) ??
                  wordAt(line, position.column - 2);
                const tensor = word ? peekTensors?.get(word) : undefined;
                if (tensor) {
                  event.preventDefault();
                  const spot = below(value, start);
                  placePeek(tensor, {
                    left: spot.left,
                    top: spot.top - LINE,
                    bottom: spot.top,
                  });
                  return;
                }
              }
              if (peek && event.key === "Escape") {
                event.preventDefault();
                event.stopPropagation();
                setPeek(null);
                return;
              }
              // Run to cursor: show the caret line's next recorded step.
              if (event.key === "F10" && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                onSelectLine(caretPosition(value, start).line);
                return;
              }
              if (event.key === "F9" && onBreakpoint) {
                event.preventDefault();
                onBreakpoint(caretPosition(value, start).line);
                return;
              }
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                if (event.shiftKey && onCheck) onCheck();
                else onRun();
              } else if (readOnly) return;
              else if (event.key === "Tab" && event.shiftKey) {
                // Tab indents code; Shift+Tab remains a way out of the editor.
                setCompletion(null);
                return;
              } else if (completion && event.key === "Escape") {
                event.preventDefault();
                event.stopPropagation();
                setCompletion(null);
              } else if (
                completion &&
                (event.key === "ArrowDown" || event.key === "ArrowUp")
              ) {
                event.preventDefault();
                const count = completion.items.length;
                const step = event.key === "ArrowDown" ? 1 : -1;
                setCompletion({
                  ...completion,
                  index: (completion.index + step + count) % count,
                });
              } else if (
                completion &&
                (event.key === "Enter" || event.key === "Tab")
              ) {
                event.preventDefault();
                accept(completion.items[completion.index]);
              } else if (event.key === "Tab") {
                event.preventDefault();
                edit(
                  value.slice(0, start) + "    " + value.slice(end),
                  start + 4,
                );
              } else if (
                event.key === "Enter" &&
                !event.shiftKey &&
                !event.altKey
              ) {
                // Keep the current indentation, and indent after a colon.
                const lineStart = value.lastIndexOf("\n", start - 1) + 1;
                const current = value.slice(lineStart, start);
                const indent = /^[ \t]*/.exec(current)![0];
                const extra = /:\s*(#.*)?$/.test(current) ? "    " : "";
                if (!indent && !extra) return;
                event.preventDefault();
                const insert = `\n${indent}${extra}`;
                edit(
                  value.slice(0, start) + insert + value.slice(end),
                  start + insert.length,
                );
              }
            }}
          />
          <span ref={measure} className="code-measure" aria-hidden="true">
            0000000000
          </span>
          <div className="code-inlays">
            {lines.map((_, i) => {
              const result = results.get(i + 1);
              const loop = loops?.get(i + 1);
              // A loop header names how often it ran and how the canvas shows it.
              const loopChip = loop && (
                <button
                  className={`code-loop ${loop.folded ? "folded" : ""} ${activeLoop === loop.id ? "selected" : ""}`}
                  aria-current={activeLoop === loop.id ? "true" : undefined}
                  aria-label={
                    loop.folded
                      ? `Line ${i + 1} loop ran ${loop.passes} identical passes, drawn once. Play its repeats.`
                      : loop.passes === 1
                        ? `Line ${i + 1} loop ran once. Go to its first step.`
                        : `Line ${i + 1} loop ran ${loop.passes} passes that differ, shown in full. Go to its first step.`
                  }
                  title={
                    loop.folded
                      ? "Every pass did the same work, so the canvas draws the body once. Select to play the repeats."
                      : loop.passes === 1
                        ? "The loop ran once, so the canvas shows its one pass. Select to go to its first step."
                        : "The passes did different work, so the canvas shows each one. Select to go to the first step."
                  }
                  onClick={() => onLoop?.(loop)}
                >
                  <Repeat size={11} aria-hidden="true" />×{loop.passes}
                  <span>
                    {loop.folded
                      ? "drawn once"
                      : loop.passes === 1
                        ? "ran once"
                        : "passes differ"}
                  </span>
                </button>
              );
              if (
                !result ||
                (!result.operations.length &&
                  !result.error &&
                  !result.predicted)
              )
                return loopChip ? (
                  <span
                    key={i}
                    className="code-inlay-slot"
                    style={{
                      top: i * LINE,
                      left: `calc(${widths[i]}ch + 3ch)`,
                    }}
                  >
                    {loopChip}
                  </span>
                ) : null;
              const steps = result.operations.length;
              const unlit =
                activatedThrough !== undefined &&
                result.fresh &&
                !result.predicted &&
                steps > 0 &&
                result.operations.every((op) => op.index > activatedThrough);
              return (
                // The slot is measured in the code font, so `ch` matches the text.
                <span
                  key={i}
                  className="code-inlay-slot"
                  style={{
                    top: i * LINE,
                    left: `calc(${widths[i]}ch + 3ch)`,
                  }}
                >
                  <button
                    className={`code-inlay ${unlit ? "unlit" : ""} ${result.predicted ? "predicted" : ""} ${result.fresh ? "" : "stale"} ${result.error ? "failed" : ""} ${activeLine === i + 1 ? "selected" : ""} ${contracts?.get(i + 1) ? (contracts.get(i + 1)!.ok ? "contract-ok" : "contract-failed") : ""}`}
                    title={contracts?.get(i + 1)?.message}
                    disabled={!steps}
                    aria-current={activeLine === i + 1 ? "true" : undefined}
                    aria-label={
                      result.error
                        ? `Line ${i + 1} failed: ${result.error}`
                        : result.output
                          ? `Line ${i + 1} ${result.predicted ? "would produce" : "produced"} ${result.output.name}, shape ${result.output.shape.join(" by ") || "scalar"}${result.predicted ? ", from a shape check" : result.fresh ? "" : ", from the previous run"}`
                          : `Line ${i + 1}: ${result.operations.at(-1)?.kind}`
                    }
                    onClick={() => onSelectLine(i + 1)}
                    onMouseEnter={(event) => {
                      onHoverLine?.(i + 1);
                      if (result.output && !result.error)
                        placePeek(
                          result.output,
                          event.currentTarget.getBoundingClientRect(),
                        );
                    }}
                    onMouseLeave={() => {
                      setPeek(null);
                      onHoverLine?.(null);
                    }}
                  >
                    {result.error ? (
                      <>
                        <CircleAlert size={11} />
                        {result.error.split(":")[0]}
                      </>
                    ) : result.output ? (
                      <>
                        <ShapeGlyph shape={result.output.shape} />
                        <b>{result.output.name}</b>
                        <span>
                          <InkShape
                            shape={result.output.shape}
                            labels={
                              result.fresh && !result.predicted
                                ? symbolicOf?.(result.output.id)
                                : null
                            }
                            ink={
                              (result.fresh || result.predicted) && !unlit
                                ? inkFor?.(result.output)
                                : null
                            }
                          />
                        </span>
                      </>
                    ) : result.needsValues ? (
                      "needs values: run to continue"
                    ) : (
                      result.operations.at(-1)?.kind
                    )}
                    {steps > 1 && <i>{steps} steps</i>}
                  </button>
                  {loopChip}
                </span>
              );
            })}
          </div>
        </div>
      </div>
      {completion && (
        <ul
          className="completion-list"
          role="listbox"
          aria-label="Suggestions"
          style={{ left: completion.left, top: completion.top }}
        >
          {completion.items.map((item, i) => (
            <li
              key={item.label}
              role="option"
              aria-selected={i === completion.index}
              // Keep focus in the editor while choosing with the pointer.
              onMouseDown={(event) => {
                event.preventDefault();
                accept(item);
              }}
            >
              <b>{item.label}</b>
              <span>{item.detail}</span>
            </li>
          ))}
        </ul>
      )}
      {peek && (
        <div
          className="tensor-peek-layer"
          style={{ left: peek.left, top: peek.top }}
        >
          <TensorPeek
            tensor={peek.tensor}
            lineage={lineageFor?.(peek.tensor)}
          />
        </div>
      )}
    </div>
  );
}
