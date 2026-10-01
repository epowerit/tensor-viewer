import { useEffect, useMemo, useRef, useState } from "react";
import { CircleAlert, Code2, Layers3, SlidersHorizontal } from "lucide-react";
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
import { SceneTransport } from "../journey/SceneTransport";
import { cellMotionPlan } from "../journey/cellMotion";
import { useSceneClock } from "../journey/useSceneClock";
import { operationSemantics } from "../journey/sceneSemantics";
import { traceCellContributors } from "../journey/cellContributors";
import type { CanvasProbe } from "../journey/CanvasCellProbe";
import { producedTensorIds } from "../tensors/provenance";
import { InspectionActivityContext } from "../journey/InspectionActivity";
import { FocusConnections } from "../journey/FocusConnections";

type Props = {
  run: Run | null;
  busy: boolean;
  active: boolean;
  /** The shown run was recorded from different code or inputs than the editor holds. */
  stale: boolean;
  onInspect: () => void;
  onEditModel: () => void;
  onEditInputs: () => void;
  /** An editor asks for one operation; a new key repeats the same request. */
  focusOperation?: { id: string; key: number; cell?: number } | null;
  onCurrentOperation?: (id: string | null) => void;
  onCell?: (node: string, index: number) => void;
};
export function Walkthrough({
  run,
  busy,
  active,
  stale,
  onInspect,
  onEditModel,
  onEditInputs,
  focusOperation,
  onCurrentOperation,
  onCell,
}: Props) {
  const [selected, setSelected] = useState<string | null>(null);
  const [tensorChoices, setTensorChoices] = useState<Record<string, string>>(
    {},
  );
  const [selection, setSelection] = useState<{
    nodeId: string;
    tensorId?: string;
    cell?: number;
  } | null>(null);
  const [inspector, setInspector] = useState(false);
  const [inspectorView, setInspectorView] = useState<"code" | "values">("code");
  const [showValues, setShowValues] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const [connectionsOpen, setConnectionsOpen] = useState(false);
  const [volume, setVolume] = useState<{ id: string; index: number } | null>(
    null,
  );
  const stage = useRef<HTMLDivElement>(null);
  const [playing, setPlaying] = useState(false);
  const [tracedCell, setTracedCell] = useState<{
    operationId: string;
    tensorId: string;
    index: number;
  } | null>(null);
  const [cycle, setCycle] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [following, setFollowing] = useState(true);
  const [reveal, setReveal] = useState(false);
  const [focusKey, setFocusKey] = useState(0);
  const [selectionKey, setSelectionKey] = useState(0);
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
    : (current?.stage?.start_index ?? -1);
  const highlighted = useMemo(
    () => (graph && selected ? ancestors(graph, selected) : new Set<string>()),
    [graph, selected],
  );
  const motionPlan = useMemo(
    () =>
      run && current?.operation
        ? cellMotionPlan(run, current.operation)
        : undefined,
    [run, current?.operation],
  );
  const semantics = useMemo(
    () =>
      run && current?.operation
        ? operationSemantics(run, current.operation)
        : undefined,
    [run, current?.operation],
  );
  const probe = useMemo<CanvasProbe | undefined>(() => {
    if (!run || !current?.operation || tracedCell?.operationId !== current.id)
      return undefined;
    const tensor = run.trace.tensors[tracedCell.tensorId];
    if (!tensor) return undefined;
    return {
      tensor,
      index: tracedCell.index,
      runId: run.id,
      result: traceCellContributors(
        run,
        current.operation,
        tensor.id,
        tracedCell.index,
      ),
    };
  }, [run, current?.operation, tracedCell]);
  const clock = useSceneClock({
    key: `${run?.id ?? ""}/${current?.operation?.id ?? ""}/${cycle}`,
    playing: playing && active && !busy,
    speed,
    onComplete: () => {
      if (index >= operations.length - 1) setPlaying(false);
      else cue(index + 1);
    },
  });

  useEffect(() => {
    setSelected(null);
    setTensorChoices({});
    setSelection(null);
    setVolume(null);
    setInspector(false);
    setInspectorView("code");
    setExpanded(false);
    setConnectionsOpen(false);
    setPlaying(false);
    setTracedCell(null);
    setCycle(0);
    setFollowing(true);
    setReveal(false);
    setFocusKey(0);
    setSelectionKey(0);
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
    const pauseWhenHidden = () => {
      if (document.hidden) setPlaying(false);
    };
    document.addEventListener("visibilitychange", pauseWhenHidden);
    return () =>
      document.removeEventListener("visibilitychange", pauseWhenHidden);
  }, []);

  useEffect(() => {
    if (
      focusOperation &&
      fullGraph?.nodes.some((node) => node.id === focusOperation.id)
    )
      select(
        focusOperation.id,
        undefined,
        focusOperation.cell,
        focusOperation.cell !== undefined,
      );
  }, [focusOperation?.key]);
  const currentOperationId = current?.stage ? null : (current?.id ?? null);
  useEffect(() => {
    onCurrentOperation?.(currentOperationId);
  }, [currentOperationId, expanded, run?.id]);

  function select(
    id: string,
    tensorId?: string,
    cell?: number,
    inspect = false,
  ) {
    setTracedCell(null);
    setSelected(id);
    setSelection({ nodeId: id, tensorId: tensorId ?? tensorChoices[id], cell });
    setExpanded(inspect);
    setPlaying(false);
    setFocusKey((key) => key + 1);
    setSelectionKey((key) => key + 1);
    onInspect();
  }
  function inspect(id: string, tensorId?: string, cell?: number) {
    select(id, tensorId, cell, true);
  }
  function inspectCurrent() {
    if (!current) return;
    if (tracedCell?.operationId === current.id) {
      inspect(current.id, tracedCell.tensorId, tracedCell.index);
    } else {
      const chosen = selection?.nodeId === current.id ? selection : null;
      inspect(current.id, chosen?.tensorId, chosen?.cell);
    }
  }
  function returnToCanvas() {
    setExpanded(false);
    setInspector(false);
    setPlaying(false);
    setFocusKey((key) => key + 1);
    setSelectionKey((key) => key + 1);
    stage.current
      ?.querySelector<HTMLElement>(".journey-canvas")
      ?.focus({ preventScroll: true });
  }
  function overview() {
    setExpanded(false);
    setInspector(false);
    setTracedCell(null);
    setPlaying(false);
    setFocusKey(0);
    stage.current
      ?.querySelector<HTMLElement>(".journey-canvas")
      ?.focus({ preventScroll: true });
  }
  function cue(next: number) {
    if (!operations[next]) return;
    setTracedCell(null);
    setCycle((value) => value + 1);
    setSelected(operations[next].id);
    setSelection(null);
    setFocusKey((key) => key + 1);
  }
  function jump(next: number) {
    setPlaying(false);
    if (expanded && operations[next]) inspect(operations[next].id);
    else cue(next);
  }
  function togglePlayback() {
    if (playing) {
      setPlaying(false);
      return;
    }
    setTracedCell(null);
    setExpanded(false);
    setInspector(false);
    setSelection(null);
    if (current?.stage) cue(current.stage.start_index);
    else if (
      index < 0 ||
      (index >= operations.length - 1 && clock.getSnapshot() >= 1)
    )
      cue(0);
    else if (clock.getSnapshot() >= 1) cue(index);
    else setFocusKey((key) => key + 1);
    setPlaying(true);
    onInspect();
  }
  function changeReveal(next: boolean) {
    setReveal(next);
    if (next) {
      setCollapsed(new Set());
      if (current?.stage) {
        cue(current.stage.start_index);
        setExpanded(false);
      } else if (index < 0) cue(0);
    }
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
      onSelect={(id) => inspect(id)}
    />
  );
  return (
    <section className="journey-view" aria-label="Tensor journey">
      <div
        className={`journey-stage ${expanded && active ? "has-focus" : ""}`}
        ref={stage}
      >
        <div className="scene-context">
          <span className="scene-model-name" title={run.project.class_name}>
            {run.project.class_name}
          </span>
          <StageControls
            stages={stages}
            collapsed={collapsed}
            disabled={reveal || playing}
            onToggle={toggleStage}
            onOverview={stageOverview}
            onExpandAll={expandAll}
          />
        </div>
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
          selectionKey={selectionKey}
          playing={playing}
          onPlaybackToggle={
            operations.length && !busy ? togglePlayback : undefined
          }
          sceneMode={!expanded}
          semantics={semantics}
          probe={!expanded ? probe : undefined}
          onTraceCell={(tensorId, index) => {
            if (!current?.operation) return;
            setPlaying(false);
            setTracedCell({ operationId: current.id, tensorId, index });
          }}
          onClearTrace={() => setTracedCell(null)}
          outputIds={run.trace.output_ids}
          followPlayback={following}
          motion={
            !expanded && motionPlan
              ? { plan: motionPlan, clock, tensors: run.trace.tensors }
              : undefined
          }
          onFollowPlaybackChange={setFollowing}
          onTensorChoice={(nodeId, tensorId) =>
            setTensorChoices((previous) => ({
              ...previous,
              [nodeId]: tensorId,
            }))
          }
          visibleThrough={
            reveal ? (current?.operation?.index ?? -1) : undefined
          }
          onSelect={select}
          onInspect={current ? inspectCurrent : undefined}
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
            onSelect={(index) => {
              if (
                current?.operation &&
                producedTensorIds(current.operation).includes(volume.id)
              )
                setTracedCell({
                  operationId: current.id,
                  tensorId: volume.id,
                  index,
                });
            }}
            onClose={() => setVolume(null)}
          />
        )}
        <InspectionActivityContext value={expanded && active && !busy}>
          {expanded && current?.stage && (
            <StageFocus
              key={`${run.id}-${current.stage.id}-${selection?.nodeId === current.id ? (selection.tensorId ?? "") : ""}`}
              active={active}
              run={run}
              stage={current.stage}
              connections={connections}
              initialTensorId={
                selection?.nodeId === current.id
                  ? selection.tensorId
                  : undefined
              }
              onClose={returnToCanvas}
              onExpand={() => toggleStage(current.stage!.id)}
              onSelect={inspect}
              showValues={showValues}
              onShowValues={setShowValues}
            />
          )}
          {expanded && current && !current.stage && (
            <TransformationFocus
              key={`${run.id}-${current.id}`}
              active={active}
              run={run}
              node={current}
              connections={connections}
              inspectorOpen={inspector}
              codeOpen={inspector && inspectorView === "code"}
              onSelect={inspect}
              onClose={returnToCanvas}
              onCode={(open) => {
                setInspector(open);
                setInspectorView("code");
              }}
              showValues={showValues}
              onShowValues={setShowValues}
              initialTensorId={
                selection?.nodeId === current.id
                  ? selection.tensorId
                  : undefined
              }
              initialCell={
                selection?.nodeId === current.id ? selection.cell : undefined
              }
              onCell={(index) => onCell?.(current.id, index)}
            />
          )}
        </InspectionActivityContext>
        <SceneTransport
          clock={clock}
          operations={operations}
          index={index}
          frameLabel={
            current?.stage
              ? current.stage.title
              : !current?.operation
                ? current?.tensors[0]?.name
                : undefined
          }
          framePosition={
            current?.stage
              ? `${current.stage.start_index + 1}–${current.stage.end_index} / ${operations.length}`
              : undefined
          }
          nextIndex={current?.stage ? current.stage.start_index : index + 1}
          playing={playing}
          busy={busy}
          expanded={expanded}
          canInspect={!!current}
          inspectLabel={
            current?.stage
              ? "Inspect current stage"
              : current && !current.operation
                ? "Inspect current tensor"
                : "Inspect current operation"
          }
          speed={speed}
          reveal={reveal}
          following={following}
          notices={run.trace.warnings?.length ?? 0}
          stopped={!!run.trace.error}
          onPlay={togglePlayback}
          onSeek={jump}
          onInspect={inspectCurrent}
          onOverview={returnToCanvas}
          onSpeed={setSpeed}
          onReveal={changeReveal}
          onFollow={setFollowing}
        >
          <details
            className="journey-run-details"
            open={!!run.trace.warnings?.length || !!run.trace.error}
          >
            <summary>
              Run details
              {run.trace.warnings?.length ? " · tracking notice" : ""}
            </summary>
            <div>
              {run.trace.error && (
                <p className="run-tracking-warning">
                  <CircleAlert size={14} /> {run.trace.error.type}:{" "}
                  {run.trace.error.message}
                </p>
              )}
              {run.trace.warnings?.map((warning, i) => (
                <p className="run-tracking-warning" key={i}>
                  <CircleAlert size={14} /> {warning}
                </p>
              ))}
              {run.trace.operations.some((op) => op.mutations?.length) && (
                <p>
                  Dashed connections carry shared-storage dependencies. Select
                  an in-place step to compare each recorded view.
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
        </SceneTransport>
      </div>
      {inspector && current && !current.stage && (
        <JourneyInspector
          active={active}
          run={run}
          node={current}
          initialTensorId={
            selection?.nodeId === current.id ? selection.tensorId : undefined
          }
          initialCell={
            selection?.nodeId === current.id ? selection.cell : undefined
          }
          onSelect={inspect}
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
