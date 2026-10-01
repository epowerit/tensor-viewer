import { useEffect, useMemo, useRef, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  Code2,
  Layers3,
  Pause,
  Play,
  SlidersHorizontal,
} from "lucide-react";
import type { Run } from "../api/client";
import { ancestors, buildJourney } from "../journey/graph";
import { JourneyCanvas } from "../journey/JourneyCanvas";
import { JourneyInspector } from "../journey/JourneyInspector";
import { TransformationFocus } from "../journey/TransformationFocus";
import { TensorVolumeDialog } from "../tensors/TensorVolumeDialog";
import {
  collapseJourney,
  journeyStages,
  stageAncestors,
} from "../journey/stages";
import { StageControls } from "../journey/StageControls";
import { StageFocus } from "../journey/StageFocus";
import { InspectionActivityContext } from "../journey/InspectionActivity";
import { FocusConnections } from "../journey/FocusConnections";
import { PlaybackOptions } from "../journey/PlaybackOptions";

type Props = {
  run: Run | null;
  busy: boolean;
  active: boolean;
  stale: boolean;
  onInspect: () => void;
  onEditModel: () => void;
  onEditInputs: () => void;
};
export function Walkthrough({
  run,
  busy,
  active,
  stale,
  onInspect,
  onEditModel,
  onEditInputs,
}: Props) {
  const [selected, setSelected] = useState<string | null>(null);
  const [inspector, setInspector] = useState(false);
  const [showValues, setShowValues] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const [connectionsOpen, setConnectionsOpen] = useState(false);
  const [volume, setVolume] = useState<{ id: string; index: number } | null>(
    null,
  );
  const stage = useRef<HTMLDivElement>(null);
  const [playing, setPlaying] = useState(false);
  const [reveal, setReveal] = useState(false);
  const [focusKey, setFocusKey] = useState(0);
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const fullGraph = useMemo(
    () => (run ? buildJourney(run.trace) : null),
    [run],
  );
  const stages = useMemo(() => (run ? journeyStages(run) : []), [run]);
  const graph = useMemo(
    () =>
      run && fullGraph
        ? collapseJourney(fullGraph, stages, collapsed, run)
        : null,
    [run, fullGraph, stages, collapsed],
  );
  const operations = run?.trace.operations ?? [];
  const current =
    graph?.nodes.find((node) => node.id === selected) ??
    fullGraph?.nodes.find((node) => node.id === selected);
  const index = current?.operation
    ? operations.indexOf(current.operation)
    : (current?.stage?.start_index ?? 0) - 1;
  const highlighted = useMemo(
    () => (graph && selected ? ancestors(graph, selected) : new Set<string>()),
    [graph, selected],
  );

  useEffect(() => {
    setSelected(null);
    setVolume(null);
    setInspector(false);
    setExpanded(false);
    setConnectionsOpen(false);
    setPlaying(false);
    setReveal(false);
    setFocusKey(0);
    setCollapsed(
      new Set(
        operations.length > 24
          ? stages.filter((item) => !item.parentStageId).map((item) => item.id)
          : [],
      ),
    );
  }, [run?.id]);
  useEffect(() => {
    if (!selected) return;
    const parents = new Set(
      stageAncestors(stages, selected).map((stage) => stage.id),
    );
    if (!parents.size) return;
    setCollapsed((previous) =>
      [...previous].some((id) => parents.has(id))
        ? new Set([...previous].filter((id) => !parents.has(id)))
        : previous,
    );
  }, [selected, stages]);
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
  function toggleStage(id: string) {
    const closing = !collapsed.has(id);
    setCollapsed((previous) => {
      const next = new Set(previous);
      if (next.has(id)) {
        next.delete(id);
        // Open one level at a time, keeping large nested blocks readable.
        stages
          .filter((item) => item.parentStageId === id)
          .forEach((item) => next.add(item.id));
      } else next.add(id);
      return next;
    });
    setSelected(closing ? id : null);
    setInspector(false);
    overview();
  }
  function stageOverview() {
    setCollapsed(
      new Set(
        stages.filter((item) => !item.parentStageId).map((item) => item.id),
      ),
    );
    setSelected(null);
    setInspector(false);
    overview();
  }
  function expandAll() {
    setCollapsed(new Set());
    setSelected(null);
    setInspector(false);
    overview();
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
            {busy
              ? "Running your model and recording its tensor transformations."
              : "Set up your model and inputs, then generate the diagram."}
          </p>
        </div>
      </section>
    );
  const connections = current && fullGraph && (
    <FocusConnections
      key={`connections-${run.id}-${current.id}`}
      run={run}
      graph={
        graph.nodes.some((node) => node.id === current.id) ? graph : fullGraph
      }
      node={current}
      open={connectionsOpen}
      onOpen={setConnectionsOpen}
      onSelect={select}
    />
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
            {operations.length}{" "}
            {operations.length === 1 ? "operation" : "operations"}
          </div>
        </header>
        <StageControls
          stages={stages}
          collapsed={collapsed}
          disabled={reveal}
          onToggle={toggleStage}
          onOverview={stageOverview}
          onExpandAll={expandAll}
        />
        {run.trace.error && (
          <div className="trace-error-strip" role="alert">
            <CircleAlert size={16} />
            <div className="run-error-content">
              <b>The run stopped here</b>
              {stale && (
                <p className="run-error-hint">
                  This is a saved execution. Run again to use your current code
                  and inputs.
                </p>
              )}
              <p className="run-error-message">{run.trace.error.message}</p>
              <small>
                {run.trace.error.type}
                {run.trace.error.line
                  ? ` · ${run.trace.error.file ?? "line"} ${run.trace.error.line}`
                  : ""}
              </small>
              <div className="run-error-actions">
                <button
                  className="secondary-button small"
                  disabled={busy}
                  onClick={onEditModel}
                >
                  <Code2 size={14} />
                  {run.project.blueprint ? "Edit model" : "Fix code"}
                </button>
                <button
                  className="secondary-button small"
                  disabled={busy}
                  onClick={onEditInputs}
                >
                  <SlidersHorizontal size={14} />
                  Edit inputs
                </button>
              </div>
              {!!operations.length && (
                <p className="run-error-hint">
                  Earlier steps are still available in the diagram.
                </p>
              )}
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
          onStageToggle={toggleStage}
        />
        {active && volume && run.trace.tensors[volume.id] && (
          <TensorVolumeDialog
            tensor={run.trace.tensors[volume.id]}
            runId={run.id}
            initialIndex={volume.index}
            onClose={() => setVolume(null)}
          />
        )}
        <InspectionActivityContext value={expanded && active && !busy}>
          {current?.stage && (
            <StageFocus
              key={`${run.id}-${current.stage.id}`}
              active={expanded && active}
              run={run}
              stage={current.stage}
              connections={connections}
              onClose={overview}
              onExpand={() => toggleStage(current.stage!.id)}
              onSelect={select}
              showValues={showValues}
              onShowValues={setShowValues}
            />
          )}
          {current && !current.stage && (
            <TransformationFocus
              key={`${run.id}-${current.id}`}
              active={expanded && active}
              run={run}
              node={current}
              connections={connections}
              inspectorOpen={inspector}
              codeOpen={inspector}
              onSelect={select}
              onClose={overview}
              onCode={(open) => {
                setInspector(open);
              }}
              showValues={showValues}
              onShowValues={setShowValues}
            />
          )}
        </InspectionActivityContext>
        <div className="journey-playback">
          <div className="playback-buttons">
            <button
              aria-label="Previous operation"
              title="Previous operation in execution order"
              disabled={
                current?.stage ? current.stage.start_index === 0 : index <= 0
              }
              onClick={() =>
                jump(current?.stage ? current.stage.start_index - 1 : index - 1)
              }
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
              title="Next operation in execution order"
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
              Explore {operations.length}{" "}
              {operations.length === 1 ? "operation" : "operations"}
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
          {(!expanded || !!current?.stage) && (
            <button
              className="inspector-toggle"
              aria-label={
                inspector && active
                  ? "Close executed code"
                  : "Open executed code"
              }
              aria-pressed={inspector && active}
              onClick={() => {
                if (inspector && active) setInspector(false);
                else {
                  if (current?.stage)
                    select(operations[current.stage.start_index].id);
                  else if (!selected)
                    setSelected(
                      operations[0]?.id ?? graph.nodes[0]?.id ?? null,
                    );
                  setInspector(true);
                  onInspect();
                }
              }}
            >
              <Code2 size={16} />
              <span>Executed code</span>
            </button>
          )}
          <PlaybackOptions
            key={run.id}
            active={active}
            disabled={busy || !operations.length}
            reveal={reveal}
            onReveal={(value) => {
              setReveal(value);
              if (value) {
                setCollapsed(new Set());
                if (current?.stage) {
                  setSelected(operations[current.stage.start_index].id);
                  setExpanded(false);
                }
                if (index < 0 && operations[0]) {
                  setSelected(operations[0].id);
                  setFocusKey((key) => key + 1);
                }
              }
            }}
            onRestart={() => jump(0)}
          />
        </div>
        <details className="journey-run-details">
          <summary>
            Run details{run.trace.warnings?.length ? " · tracking notice" : ""}
          </summary>
          <div>
            {run.trace.warnings?.map((warning, i) => (
              <p className="run-tracking-warning" key={i}>
                <CircleAlert size={14} /> {warning}
              </p>
            ))}
            {run.trace.operations.some((op) => op.mutations?.length) && (
              <p>
                Dashed connections carry shared-storage dependencies. Select an
                in-place step to compare each recorded view.
              </p>
            )}
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
              Entry: <code>{run.project.entry_path ?? "model.py"}</code> ·{" "}
              {run.project.class_name}
            </p>
            {run.project.repository && (
              <p title={run.project.repository.url}>
                Source imported from commit{" "}
                <code>{run.project.repository.revision.slice(0, 12)}</code>.
                This run preserves its own source snapshot.
              </p>
            )}
            {run.trace.runtime?.Python && (
              <p>
                Python {run.trace.runtime.Python} · PyTorch{" "}
                {run.trace.runtime.torch ?? "unknown"} ·{" "}
                {run.project.environment
                  ? "selected environment"
                  : "TensorViewer environment"}
              </p>
            )}
            <p>
              Tensor drawings are schematic. Stacks represent leading
              dimensions; weights are available in Tensor details.
            </p>
            {run.project.weights ? (
              <p className="run-weight-provenance">
                Weights: <b>{run.project.weights.name}</b> ·{" "}
                {run.trace.weight_check?.compatible
                  ? "matched to model"
                  : "not validated"}
                <br />
                <code>SHA-256 {run.project.weights.sha256}</code>
              </p>
            ) : (
              <p>
                Weights: initialized by the module · model seed{" "}
                {run.project.input.seed}
              </p>
            )}
            {run.trace.stdout && <pre>{run.trace.stdout}</pre>}
          </div>
        </details>
      </div>
      {inspector && current && !current.stage && (
        <JourneyInspector
          active={active}
          run={run}
          node={current}
          onSelect={select}
          onClose={() => setInspector(false)}
        />
      )}
    </section>
  );
}
