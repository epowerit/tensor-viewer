import { useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Code2,
  Layers3,
  Pause,
  Play,
  RotateCcw,
} from "lucide-react";
import type { Run } from "../api/client";
import { ancestors, buildJourney } from "../journey/graph";
import { JourneyCanvas } from "../journey/JourneyCanvas";
import { JourneyInspector } from "../journey/JourneyInspector";
import { TransformationFocus } from "../journey/TransformationFocus";
import { TensorVolumeDialog } from "../tensors/TensorVolumeDialog";

type Props = {
  run: Run | null;
  busy: boolean;
  active: boolean;
  onInspect: () => void;
};
export function Walkthrough({ run, busy, active, onInspect }: Props) {
  const [selected, setSelected] = useState<string | null>(null);
  const [inspector, setInspector] = useState(false);
  const [inspectorView, setInspectorView] = useState<"code" | "values">("code");
  const [showValues, setShowValues] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const [volume, setVolume] = useState<{ id: string; index: number } | null>(
    null,
  );
  const stage = useRef<HTMLDivElement>(null);
  const [playing, setPlaying] = useState(false);
  const [reveal, setReveal] = useState(false);
  const [focusKey, setFocusKey] = useState(0);
  const graph = useMemo(() => (run ? buildJourney(run.trace) : null), [run]);
  const operations = run?.trace.operations ?? [];
  const current = graph?.nodes.find((node) => node.id === selected);
  const index = current?.operation ? operations.indexOf(current.operation) : -1;
  const highlighted = useMemo(
    () => (graph && selected ? ancestors(graph, selected) : new Set<string>()),
    [graph, selected],
  );

  useEffect(() => {
    setSelected(null);
    setVolume(null);
    setInspector(false);
    setInspectorView("code");
    setExpanded(false);
    setPlaying(false);
    setReveal(false);
    setFocusKey(0);
  }, [run?.id]);
  useEffect(() => {
    if (!active || busy) setPlaying(false);
  }, [active, busy]);
  useEffect(() => {
    if (!playing || !active || busy) return;
    if (index >= operations.length - 1) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(() => {
      setSelected(operations[index + 1].id);
      setFocusKey((key) => key + 1);
    }, 2800);
    return () => clearTimeout(timer);
  }, [playing, active, busy, index, operations]);

  function select(id: string) {
    setSelected(id);
    setExpanded(true);
    setPlaying(false);
    setFocusKey((key) => key + 1);
    onInspect();
  }
  function overview() {
    setExpanded(false);
    setPlaying(false);
    setFocusKey(0);
    stage.current
      ?.querySelector<HTMLElement>(".journey-canvas")
      ?.focus({ preventScroll: true });
  }
  function jump(next: number) {
    if (operations[next]) select(operations[next].id);
  }

  if (!run || !graph)
    return (
      <section className="journey-view">
        <div className="empty-workspace">
          <Layers3 size={38} />
          <h2>
            {busy
              ? "Recording your forward pass…"
              : "Your tensor journey starts here"}
          </h2>
          <p>
            Choose your code and input settings, then run to see the full
            transformation path.
          </p>
        </div>
      </section>
    );
  return (
    <section className="journey-view" aria-label="Tensor journey">
      <div
        className={`journey-stage ${expanded && active ? "has-focus" : ""}`}
        ref={stage}
      >
        <header className="canvas-heading">
          <div>
            <Layers3 size={17} />
            <h1>Tensor journey</h1>
            <span className="canvas-class">{run.project.class_name}</span>
          </div>
          <div className="canvas-status">
            <span
              className={`status-dot ${run.trace.error ? "error-dot" : ""}`}
            />
            <span className="recording-label">
              {run.trace.error
                ? "Stopped"
                : run.project.capture_mode === "shapes"
                  ? "Shape preview"
                  : "Recorded"}
            </span>
            <span className="canvas-separator">·</span>
            {operations.length} operations
            <label
              className="reveal-toggle"
              title="Reveal transformations in execution order"
            >
              <input
                type="checkbox"
                checked={reveal}
                disabled={!operations.length}
                onChange={(event) => {
                  setReveal(event.target.checked);
                  if (event.target.checked && index < 0 && operations[0]) {
                    setSelected(operations[0].id);
                    setFocusKey((key) => key + 1);
                  }
                }}
              />
              Reveal steps
            </label>
          </div>
        </header>
        {run.trace.error && (
          <div className="trace-error-strip" role="alert">
            <CircleAlert size={16} />
            <div>
              <b>
                {run.trace.error.type}
                {run.trace.error.line ? ` · line ${run.trace.error.line}` : ""}
              </b>
              <p>{run.trace.error.message}</p>
            </div>
          </div>
        )}
        <JourneyCanvas
          key={run.id}
          graph={graph}
          selectedId={selected}
          highlighted={highlighted}
          focusKey={focusKey}
          visibleThrough={
            reveal ? (current?.operation?.index ?? -1) : undefined
          }
          onSelect={select}
          onOverview={overview}
          onTensorInspect={(id, index) => {
            setPlaying(false);
            setVolume({ id, index });
          }}
        />
        {volume && run.trace.tensors[volume.id] && (
          <TensorVolumeDialog
            tensor={run.trace.tensors[volume.id]}
            runId={run.id}
            initialIndex={volume.index}
            onClose={() => setVolume(null)}
          />
        )}
        {expanded && active && current && (
          <TransformationFocus
            run={run}
            node={current}
            inspectorOpen={inspector}
            codeOpen={inspector && inspectorView === "code"}
            onSelect={select}
            onClose={overview}
            onCode={(open) => {
              setInspector(open);
              setInspectorView("code");
            }}
            showValues={showValues}
            onShowValues={setShowValues}
          />
        )}
        <div className="journey-playback">
          <div className="playback-buttons">
            <button
              aria-label="Restart walkthrough"
              title="Restart walkthrough"
              onClick={() => jump(0)}
              disabled={!operations.length}
            >
              <RotateCcw size={15} />
            </button>
            <button
              aria-label="Previous operation"
              title="Previous operation"
              disabled={index <= 0}
              onClick={() => jump(index - 1)}
            >
              <ChevronLeft size={18} />
            </button>
            <button
              className="play-toggle"
              disabled={busy || !operations.length}
              aria-label={playing ? "Pause playback" : "Play walkthrough"}
              onClick={() => {
                if (playing) {
                  setPlaying(false);
                  return;
                }
                if (index < 0 || index === operations.length - 1) {
                  setSelected(operations[0].id);
                  setFocusKey((key) => key + 1);
                }
                setPlaying(true);
                onInspect();
              }}
            >
              {playing ? <Pause size={14} /> : <Play size={14} />}
              <span>{playing ? "Pause" : "Play"}</span>
            </button>
            <button
              aria-label="Next operation"
              title="Next operation"
              disabled={!operations.length || index === operations.length - 1}
              onClick={() => jump(index + 1)}
            >
              <ChevronRight size={18} />
            </button>
          </div>
          <select
            aria-label="Jump to operation"
            value={current?.operation?.id ?? ""}
            onChange={(event) => select(event.target.value)}
          >
            <option value="" disabled>
              Explore {operations.length} operations
            </option>
            {operations.map((op) => (
              <option key={op.id} value={op.id}>
                {op.index + 1} / {operations.length} · {op.kind}
                {op.outputs[0]
                  ? ` → ${run.trace.tensors[op.outputs[0]].name}`
                  : ""}
              </option>
            ))}
          </select>
          <button
            className="inspector-toggle"
            aria-label={
              inspector && active && inspectorView === "code"
                ? "Close code panel"
                : "Open code panel"
            }
            aria-pressed={inspector && active && inspectorView === "code"}
            onClick={() => {
              if (inspector && active && inspectorView === "code")
                setInspector(false);
              else {
                if (!selected)
                  setSelected(operations[0]?.id ?? graph.nodes[0]?.id ?? null);
                setInspector(true);
                setInspectorView("code");
                onInspect();
              }
            }}
          >
            <Code2 size={16} />
            <span>Code</span>
          </button>
        </div>
        <details className="journey-run-details">
          <summary>Run details</summary>
          <div>
            <p>
              {Object.keys(run.trace.tensors).length} tensor states ·{" "}
              {run.trace.duration_ms.toFixed(0)} ms including tracing
            </p>
            <p>
              {new Date(run.created_at).toLocaleString()} ·{" "}
              {run.project.capture_mode === "shapes"
                ? "Shapes only"
                : "CPU values"}{" "}
              · evaluation mode
            </p>
            <p>
              Tensor drawings are schematic. Stacks represent leading
              dimensions; weights are available in the inspector.
            </p>
            {run.trace.stdout && <pre>{run.trace.stdout}</pre>}
          </div>
        </details>
      </div>
      {inspector && active && current && (
        <JourneyInspector
          run={run}
          node={current}
          onSelect={select}
          onClose={() => setInspector(false)}
          tab={inspectorView}
          onTab={setInspectorView}
          showValues={showValues}
          onShowValues={setShowValues}
        />
      )}
    </section>
  );
}
