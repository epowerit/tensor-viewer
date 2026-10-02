import {
  useEffect,
  useId,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import {
  ChevronLeft,
  ChevronRight,
  CornerLeftUp,
  Maximize2,
  Redo2,
  MoreHorizontal,
  Pause,
  Play,
  RotateCcw,
  ScanLine,
  X,
  CircleAlert,
} from "lucide-react";
import type { Operation } from "../api/client";
import { useSceneProgress, type SceneClock } from "./useSceneClock";
import "./sceneTransport.css";
import { kindName } from "../operations/kindName";

type Props = {
  clock?: SceneClock;
  /** Playback steps: recorded operations, or a loop's repeats. */
  operations: Pick<Operation, "id" | "kind" | "outputs">[];
  index: number;
  frameLabel?: string;
  framePosition?: string;
  nextIndex?: number;
  playing: boolean;
  /** Playback stopped here, and why. */
  breakpoint?: "breakpoint" | "warning";
  /** Step over the calls the current step makes (absent: nothing to skip). */
  onStepOver?: () => void;
  /** Step out of the innermost call around the current step. */
  onStepOut?: () => void;
  /** Whether playback also pauses at steps with warnings; absent hides it. */
  pauseOnWarnings?: boolean;
  onPauseOnWarnings?: (pause: boolean) => void;
  busy: boolean;
  expanded: boolean;
  canInspect?: boolean;
  inspectLabel?: string;
  speed: number;
  reveal: boolean;
  following: boolean;
  notices?: number;
  stopped?: boolean;
  onPlay: () => void;
  onSeek: (index: number) => void;
  onInspect: () => void;
  onOverview: () => void;
  onSpeed: (speed: number) => void;
  onReveal: (reveal: boolean) => void;
  onFollow: (follow: boolean) => void;
  children: ReactNode;
};

/** One transport for the recorded canvas and its detailed tensor inspections. */
export function SceneTransport({
  clock,
  operations,
  index,
  frameLabel,
  framePosition,
  nextIndex = index + 1,
  playing,
  breakpoint,
  onStepOver,
  onStepOut,
  pauseOnWarnings,
  onPauseOnWarnings,
  busy,
  expanded,
  canInspect = index >= 0 && index < operations.length,
  inspectLabel = "Inspect current operation",
  speed,
  reveal,
  following,
  notices = 0,
  stopped = false,
  onPlay,
  onSeek,
  onInspect,
  onOverview,
  onSpeed,
  onReveal,
  onFollow,
  children,
}: Props) {
  const [options, setOptions] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const optionsId = useId();
  const current = operations[index];
  const timelineFill =
    operations.length > 1
      ? Math.max(0, Math.min(100, (index / (operations.length - 1)) * 100))
      : 0;
  function closeOptions() {
    setOptions(false);
    trigger.current?.focus();
  }
  useEffect(() => {
    if (!options) return;
    function outside(event: PointerEvent) {
      if (!root.current?.contains(event.target as Node)) setOptions(false);
    }
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [options]);
  return (
    <div
      className="journey-playback scene-transport"
      ref={root}
      role="group"
      aria-label="Recorded tensor playback"
      onBlur={(event) => {
        if (
          event.relatedTarget &&
          !event.currentTarget.contains(event.relatedTarget)
        ) {
          setOptions(false);
        }
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape" && options) {
          event.preventDefault();
          event.stopPropagation();
          closeOptions();
        }
      }}
    >
      <div className="playback-buttons">
        <button
          aria-label="Previous operation"
          title="Previous operation"
          disabled={busy || index <= 0}
          onClick={() => onSeek(index - 1)}
        >
          <ChevronLeft size={17} />
        </button>
        <button
          className="play-toggle"
          aria-label={playing ? "Pause playback" : "Play tensor journey"}
          title={
            playing ? "Pause playback" : "Play tensor journey on the canvas"
          }
          disabled={busy || !operations.length}
          onClick={onPlay}
        >
          {playing ? <Pause size={15} /> : <Play size={15} />}
        </button>
        <button
          aria-label="Next operation"
          title="Next operation"
          disabled={
            busy || !operations.length || nextIndex >= operations.length
          }
          onClick={() => onSeek(nextIndex)}
        >
          <ChevronRight size={17} />
        </button>
        <button
          className="step-call"
          aria-label="Step over"
          title="Step over: the next step at this depth, past the calls this one makes (F10)"
          aria-keyshortcuts="F10"
          disabled={busy || !onStepOver}
          onClick={onStepOver}
        >
          <Redo2 size={15} />
        </button>
        <button
          className="step-call"
          aria-label="Step out"
          title="Step out: the first step after the call around this one (Shift+F11)"
          aria-keyshortcuts="Shift+F11"
          disabled={busy || !onStepOut}
          onClick={onStepOut}
        >
          <CornerLeftUp size={15} />
        </button>
      </div>
      <div className="scene-timeline">
        {clock && <FrameProgress clock={clock} />}
        <div className="scene-frame-label">
          <span
            title={frameLabel ?? (current ? kindName(current.kind) : undefined)}
          >
            {breakpoint && (
              <i
                className={`scene-breakpoint ${breakpoint}`}
                role="img"
                aria-label={
                  breakpoint === "warning"
                    ? "Paused at a warning"
                    : "Paused at a breakpoint"
                }
                title={
                  breakpoint === "warning"
                    ? "Paused at a step with a warning. See Run notes; Play continues."
                    : "Paused at a breakpoint. Play continues to the next one."
                }
              />
            )}
            {frameLabel ??
              (current ? kindName(current.kind) : "Tensor journey")}
          </span>
          <span>
            {framePosition ??
              `${index >= 0 ? index + 1 : 0} / ${operations.length}`}
          </span>
        </div>
        <input
          type="range"
          style={{ "--timeline-fill": `${timelineFill}%` } as CSSProperties}
          min={0}
          max={Math.max(0, operations.length - 1)}
          value={Math.max(0, index)}
          disabled={busy || !operations.length}
          aria-label="Scrub recorded operations"
          aria-valuetext={
            current
              ? `Step ${index + 1} of ${operations.length}: ${kindName(current.kind)}`
              : "Before the first operation"
          }
          onChange={(event) => onSeek(Number(event.target.value))}
        />
      </div>
      <button
        className="scene-action scene-inspect"
        aria-label={expanded ? "Return to model canvas" : inspectLabel}
        title={
          expanded
            ? "Return to model canvas"
            : `${inspectLabel} (Enter on canvas)`
        }
        aria-expanded={expanded}
        disabled={!expanded && !canInspect}
        onClick={expanded ? onOverview : onInspect}
      >
        {expanded ? <ScanLine size={16} /> : <Maximize2 size={16} />}
        <span>{expanded ? "Canvas" : "Inspect"}</span>
      </button>
      {!following && !expanded && (
        <button
          className="scene-action scene-resume"
          aria-label="Resume following tensors"
          title="Resume following tensors"
          onClick={() => onFollow(true)}
        >
          <ScanLine size={16} />
        </button>
      )}
      <button
        className="scene-action"
        ref={trigger}
        aria-label="Playback options"
        title={
          stopped
            ? "Playback options · run stopped"
            : notices
              ? `Playback options · ${notices} tracking notices`
              : "Playback options"
        }
        aria-description={
          stopped
            ? "This run stopped. See Run details."
            : notices
              ? `${notices} tracking notices in Run details.`
              : undefined
        }
        aria-expanded={options}
        aria-controls={options ? optionsId : undefined}
        onClick={() => setOptions(!options)}
      >
        {stopped || notices ? (
          <CircleAlert size={16} className="scene-notice" />
        ) : (
          <MoreHorizontal size={18} />
        )}
      </button>
      {options && (
        <section
          id={optionsId}
          className="scene-options"
          aria-label="Playback options"
        >
          <header>
            <b>Playback</b>
            <button
              className="icon-button"
              aria-label="Close playback options"
              onClick={closeOptions}
            >
              <X size={14} />
            </button>
          </header>
          <label>
            Speed
            <select
              aria-label="Playback speed"
              value={speed}
              onChange={(event) => onSpeed(Number(event.target.value))}
            >
              <option value={0.5}>0.5× · Slow</option>
              <option value={1}>1× · Normal</option>
              <option value={2}>2× · Fast</option>
            </select>
          </label>
          <label>
            <input
              type="checkbox"
              checked={following}
              onChange={(event) => onFollow(event.target.checked)}
            />
            Follow tensors with the camera
          </label>
          <label>
            <input
              type="checkbox"
              checked={reveal}
              disabled={!operations.length}
              onChange={(event) => onReveal(event.target.checked)}
            />
            Reveal operations as they play
          </label>
          {onPauseOnWarnings && (
            <label>
              <input
                type="checkbox"
                checked={!!pauseOnWarnings}
                onChange={(event) => onPauseOnWarnings(event.target.checked)}
              />
              Pause at steps with warnings
            </label>
          )}
          <label>
            Jump to
            <select
              aria-label="Jump to operation"
              value={current?.id ?? ""}
              disabled={!operations.length}
              onChange={(event) => {
                onSeek(
                  operations.findIndex((op) => op.id === event.target.value),
                );
                closeOptions();
              }}
            >
              <option value="" disabled>
                Choose a step
              </option>
              {operations.map((op, i) => (
                <option key={op.id} value={op.id}>
                  {i + 1} · {kindName(op.kind)}
                  {op.outputs.length > 1
                    ? ` · ${op.outputs.length} outputs`
                    : ""}
                </option>
              ))}
            </select>
          </label>
          <button
            className="scene-restart"
            disabled={!operations.length || busy}
            onClick={() => {
              onSeek(0);
              closeOptions();
            }}
          >
            <RotateCcw size={14} /> Restart from first operation
          </button>
          {children}
        </section>
      )}
    </div>
  );
}

function FrameProgress({ clock }: { clock: SceneClock }) {
  const progress = useSceneProgress(clock);
  return (
    <span
      className="scene-frame-progress"
      aria-hidden="true"
      data-progress={progress.toFixed(3)}
    >
      <span style={{ transform: `scaleX(${progress})` }} />
    </span>
  );
}
