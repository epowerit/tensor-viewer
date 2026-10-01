import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Maximize, Minus, Plus, UnfoldHorizontal } from "lucide-react";
import type { JourneyGraph, JourneyNode } from "./graph";
import { NODE_HEIGHT, NODE_WIDTH } from "./graph";
import { TensorGlyph } from "./TensorGlyph";
import { edgeDescription, journeyPorts } from "./ports";
import { operationScene, sceneView } from "./scene";
import { projectOperationScene } from "./sceneGraph";
import type { OperationSemantics } from "./sceneSemantics";
import { CanvasCellMotion } from "./CanvasCellMotion";
import { CanvasCellProbe, type CanvasProbe } from "./CanvasCellProbe";
import type { CellMotionPlan } from "./cellMotion";
import type { SceneClock } from "./useSceneClock";
import type { Tensor } from "../api/client";
import {
  fitCanvas,
  reframeCanvas,
  showTensorCells,
  type CanvasViewport,
  type Viewport,
} from "./viewport";
import "./journeyCanvas.css";

type Props = {
  graph: JourneyGraph;
  selectedId: string | null;
  highlighted: Set<string>;
  focusKey: number;
  selectionKey?: number;
  visibleThrough?: number;
  playing?: boolean;
  sceneMode?: boolean;
  followPlayback?: boolean;
  semantics?: OperationSemantics;
  outputIds?: string[];
  probe?: CanvasProbe;
  onTraceCell?: (tensorId: string, index: number) => void;
  onClearTrace?: () => void;
  motion?: {
    plan: CellMotionPlan;
    clock: SceneClock;
    tensors: Record<string, Tensor>;
  };
  onPlaybackToggle?: () => void;
  onFollowPlaybackChange?: (follow: boolean) => void;
  onTensorChoice?: (nodeId: string, tensorId: string) => void;
  onSelect: (id: string, tensorId?: string) => void;
  onInspect?: () => void;
  onOverview: () => void;
  onTensorInspect: (tensorId: string, index: number) => void;
  onStageToggle: (id: string) => void;
};
const clamp = (value: number) => Math.max(0.001, Math.min(2, value));

export function JourneyCanvas({
  graph: modelGraph,
  selectedId,
  highlighted,
  focusKey,
  selectionKey = 0,
  visibleThrough,
  playing = false,
  sceneMode = false,
  followPlayback = true,
  motion,
  semantics,
  outputIds,
  probe,
  onTraceCell,
  onClearTrace,
  onPlaybackToggle,
  onFollowPlaybackChange,
  onTensorChoice,
  onSelect,
  onInspect,
  onOverview,
  onTensorInspect,
  onStageToggle,
}: Props) {
  const contributors = useMemo(() => {
    const result = new Map<string, number[]>();
    for (const source of probe?.result.sources ?? []) {
      const indices = result.get(source.tensorId) ?? [];
      if (!indices.includes(source.index)) indices.push(source.index);
      result.set(source.tensorId, indices);
    }
    return result;
  }, [probe]);
  const returned = useMemo(() => new Set(outputIds ?? []), [outputIds]);
  const roleLabels = useMemo(() => {
    const labels = {
      input: new Map<string, string[]>(),
      output: new Map<string, string[]>(),
    };
    for (const side of ["input", "output"] as const)
      for (const item of semantics?.[side === "input" ? "inputs" : "outputs"] ??
        []) {
        const names = labels[side].get(item.tensorId) ?? [];
        names.push(item.label);
        labels[side].set(item.tensorId, names);
      }
    return labels;
  }, [semantics]);
  const graph = useMemo(
    () =>
      sceneMode && focusKey > 0 && selectedId && motion
        ? projectOperationScene(modelGraph, selectedId, motion.tensors)
        : modelGraph,
    [modelGraph, sceneMode, focusKey > 0, selectedId, motion?.tensors],
  );
  const frame = useRef<HTMLDivElement>(null);
  const world = useRef<HTMLDivElement>(null);
  const probeFocus = useRef<{ tensorId: string; index: number } | null>(null);
  useEffect(() => {
    const pending = probeFocus.current;
    if (
      !pending ||
      pending.tensorId !== probe?.tensor.id ||
      pending.index !== probe.index
    )
      return;
    const node = [
      ...(world.current?.querySelectorAll<HTMLElement>(".journey-node") ?? []),
    ].find(
      (item) =>
        item.dataset.tensorId === pending.tensorId &&
        item.dataset.ownerId === selectedId,
    );
    node
      ?.querySelector<SVGGElement>(`[data-volume-index="${pending.index}"]`)
      ?.focus({ preventScroll: true });
    probeFocus.current = null;
  }, [probe, selectedId]);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [viewport, setViewport] = useState<CanvasViewport>({
    view: { x: 0, y: 0, scale: 1 },
    frame: null,
    overview: null,
  });
  const view = viewport.view;
  function setView(next: Viewport | ((previous: Viewport) => Viewport)) {
    setViewport((previous) => ({
      ...previous,
      view: typeof next === "function" ? next(previous.view) : next,
    }));
  }
  const [tensorChoices, setTensorChoices] = useState<Record<string, number>>(
    {},
  );
  const [previewNode, setPreviewNode] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const navigation = useRef(onFollowPlaybackChange);
  navigation.current = onFollowPlaybackChange;
  function manualNavigation() {
    navigation.current?.(false);
  }
  function clearTrace() {
    onClearTrace?.();
    frame.current?.focus({ preventScroll: true });
  }
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const marker = useId().replace(/:/g, "");
  const byId = useMemo(
    () => new Map(graph.nodes.map((node) => [node.id, node])),
    [graph],
  );
  const ports = useMemo(() => journeyPorts(graph), [graph]);
  const repeatedEdges = useMemo(() => {
    const groups = new Map<string, string[]>();
    for (const edge of graph.edges) {
      const key = JSON.stringify([edge.source, edge.target, edge.tensorId]);
      const ids = groups.get(key) ?? [];
      ids.push(edge.id);
      groups.set(key, ids);
    }
    return new Set([...groups.values()].filter((ids) => ids.length > 1).flat());
  }, [graph]);
  const scene = useMemo(
    () => operationScene(graph, sceneMode && focusKey > 0 ? selectedId : null),
    [graph, selectedId, sceneMode, focusKey > 0],
  );
  const edgeDescriptions = useMemo(
    () =>
      new Map(
        graph.edges.map((edge) => [edge.id, edgeDescription(graph, edge)]),
      ),
    [graph],
  );
  const focusId =
    previewNode && byId.has(previewNode) ? previewNode : selectedId;
  const chosen = previewNode && byId.get(previewNode);
  const previewTensor = chosen
    ? chosen.tensors[tensorChoices[chosen.id] ?? 0]?.id
    : null;
  const directEdges = new Set(
    graph.edges
      .filter(
        (edge) =>
          edge.target === focusId ||
          (edge.source === focusId &&
            (!previewTensor || edge.tensorId === previewTensor)),
      )
      .map((edge) => edge.id),
  );
  const labelledEdges = new Set(
    [...(scene.nodes.size ? scene.edges : directEdges)].slice(0, 8),
  );
  useEffect(() => setPreviewNode(null), [selectedId]);
  const visibleNodes = graph.nodes.filter(
    (node) =>
      visibleThrough === undefined ||
      !node.operation ||
      node.operation.index <= visibleThrough,
  );
  const visibleIds = new Set(visibleNodes.map((node) => node.id));
  const visibleEdges = graph.edges.filter(
    (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target),
  );

  useEffect(() => {
    const target = frame.current;
    if (!target) return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry.contentRect.width && entry.contentRect.height)
        setSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
    });
    observer.observe(target);
    return () => observer.disconnect();
  }, []);

  function fittedView(): Viewport {
    const scale = Math.min(
      1,
      Math.max(
        0.001,
        Math.min(
          (size.width - 72) / modelGraph.width,
          (size.height - 170) / modelGraph.height,
        ),
      ),
    );
    return {
      scale,
      x: (size.width - modelGraph.width * scale) / 2,
      y: (size.height - modelGraph.height * scale) / 2,
    };
  }
  function fit() {
    if (!size.width || !size.height) return;
    manualNavigation();
    setViewport((previous) =>
      fitCanvas(previous, modelGraph, size, fittedView()),
    );
  }
  function focusedView(node: JourneyNode, previous: Viewport): Viewport {
    const scale = Math.max(0.8, Math.min(1.15, previous.scale));
    return {
      scale,
      x: size.width / 2 - (node.x + NODE_WIDTH / 2) * scale,
      y: size.height / 2 - (node.y + NODE_HEIGHT / 2) * scale,
    };
  }
  function focus(node: JourneyNode) {
    manualNavigation();
    setView((previous) => focusedView(node, previous));
  }
  useEffect(() => {
    if (!size.width || !size.height) return;
    const selected = graph.nodes.find((node) => node.id === selectedId);
    setViewport((previous) =>
      reframeCanvas(
        previous,
        { graph: modelGraph, layout: graph, focusKey, selectionKey, size },
        fittedView(),
        selected
          ? (current) =>
              sceneMode
                ? followPlayback ||
                  previous.frame?.selectionKey !== selectionKey
                  ? sceneView(graph, selected.id, size, current)
                  : current
                : focusedView(selected, current)
          : undefined,
      ),
    );
    // Drawers resize around the same world point; inspection has its own framing.
  }, [
    modelGraph,
    graph,
    focusKey,
    selectionKey,
    size.width,
    size.height,
    sceneMode,
  ]);

  useEffect(() => {
    if (followPlayback && sceneMode && focusKey > 0 && selectedId)
      setView((previous) => sceneView(graph, selectedId, size, previous));
    // Follow also reframes on resize; manual navigation keeps its own framing.
  }, [followPlayback, size.width, size.height]);

  function zoom(factor: number, x = size.width / 2, y = size.height / 2) {
    manualNavigation();
    setView((previous) => {
      const scale = clamp(previous.scale * factor);
      const ratio = scale / previous.scale;
      return {
        scale,
        x: x - (x - previous.x) * ratio,
        y: y - (y - previous.y) * ratio,
      };
    });
  }
  useEffect(() => {
    const target = frame.current;
    if (!target) return;
    function wheel(event: WheelEvent) {
      event.preventDefault();
      manualNavigation();
      const bounds = target!.getBoundingClientRect();
      if (event.ctrlKey || event.metaKey)
        zoom(
          Math.exp(-event.deltaY * 0.008),
          event.clientX - bounds.left,
          event.clientY - bounds.top,
        );
      else
        setView((previous) => ({
          ...previous,
          x: previous.x - event.deltaX,
          y: previous.y - event.deltaY,
        }));
    }
    target.addEventListener("wheel", wheel, { passive: false });
    return () => target.removeEventListener("wheel", wheel);
  }, []);

  return (
    <div
      ref={frame}
      className={`journey-canvas ${dragging ? "is-panning" : ""} ${scene.nodes.size ? "has-scene" : ""} ${playing ? "is-playing" : ""} ${sceneMode && followPlayback ? "is-following" : ""} ${probe?.result.status === "mapped" && !probe.result.truncated ? "has-cell-probe" : ""}`}
      role="region"
      aria-label="Tensor transformation canvas"
      aria-keyshortcuts="Space Enter Escape Home + -"
      tabIndex={0}
      style={{
        backgroundPosition: `${view.x}px ${view.y}px`,
        backgroundSize: `${28 * Math.max(0.8, view.scale)}px ${28 * Math.max(0.8, view.scale)}px`,
      }}
      onKeyDown={(event) => {
        if (event.defaultPrevented) return;
        if (event.key === "Escape" && probe) {
          event.preventDefault();
          event.stopPropagation();
          clearTrace();
          return;
        }
        if (event.target !== event.currentTarget) return;
        if (event.key === "Escape" && focusKey > 0) {
          event.preventDefault();
          event.stopPropagation();
          onOverview();
          return;
        }
        if (event.key === "Enter" && selectedId && onInspect) {
          event.preventDefault();
          onInspect();
          return;
        }
        if (event.key === " " && onPlaybackToggle) {
          event.preventDefault();
          onPlaybackToggle();
          return;
        }
        const delta = {
          ArrowLeft: [70, 0],
          ArrowRight: [-70, 0],
          ArrowUp: [0, 70],
          ArrowDown: [0, -70],
        }[event.key];
        if (delta) {
          event.preventDefault();
          manualNavigation();
          setView((p) => ({ ...p, x: p.x + delta[0], y: p.y + delta[1] }));
        }
        if (event.key === "+" || event.key === "=") {
          event.preventDefault();
          zoom(1.2);
        }
        if (event.key === "-") {
          event.preventDefault();
          zoom(1 / 1.2);
        }
        if (event.key === "Home" || event.key === "0") {
          event.preventDefault();
          onOverview();
          fit();
        }
      }}
      onPointerDown={(event) => {
        if (
          (event.target as Element).closest(
            "button,select,input,[role=button],.journey-node,.canvas-cell-probe,.canvas-zoom,.canvas-minimap",
          ) ||
          (event.button !== 0 && event.button !== 1)
        )
          return;
        event.preventDefault();
        manualNavigation();
        event.currentTarget.focus({ preventScroll: true });
        pointers.current.set(event.pointerId, {
          x: event.clientX,
          y: event.clientY,
        });
        event.currentTarget.setPointerCapture(event.pointerId);
        setDragging(true);
      }}
      onPointerMove={(event) => {
        const previous = pointers.current.get(event.pointerId);
        if (!previous) return;
        const other = [...pointers.current.entries()].find(
          ([id]) => id !== event.pointerId,
        )?.[1];
        pointers.current.set(event.pointerId, {
          x: event.clientX,
          y: event.clientY,
        });
        if (!other)
          setView((p) => ({
            ...p,
            x: p.x + event.clientX - previous.x,
            y: p.y + event.clientY - previous.y,
          }));
        else {
          const bounds = event.currentTarget.getBoundingClientRect();
          const oldDistance = Math.hypot(
            previous.x - other.x,
            previous.y - other.y,
          );
          const newDistance = Math.hypot(
            event.clientX - other.x,
            event.clientY - other.y,
          );
          if (!oldDistance) return;
          const x = (previous.x + other.x) / 2 - bounds.left;
          const y = (previous.y + other.y) / 2 - bounds.top;
          setView((p) => {
            const scale = clamp((p.scale * newDistance) / oldDistance);
            return {
              scale,
              x:
                x +
                (event.clientX - previous.x) / 2 -
                ((x - p.x) * scale) / p.scale,
              y:
                y +
                (event.clientY - previous.y) / 2 -
                ((y - p.y) * scale) / p.scale,
            };
          });
        }
      }}
      onPointerUp={(event) => {
        pointers.current.delete(event.pointerId);
        if (!pointers.current.size) setDragging(false);
      }}
      onPointerCancel={(event) => {
        pointers.current.delete(event.pointerId);
        if (!pointers.current.size) setDragging(false);
      }}
    >
      <div
        className="canvas-world"
        ref={world}
        style={{
          width: graph.width,
          height: graph.height,
          transform: `translate(${view.x}px, ${view.y}px) scale(${view.scale})`,
        }}
      >
        <svg
          className="journey-edges"
          width={graph.width}
          height={graph.height}
          aria-hidden="true"
        >
          <defs>
            <marker
              id={marker}
              markerWidth="7"
              markerHeight="7"
              refX="6"
              refY="3.5"
              orient="auto"
            >
              <path d="M0,0 L7,3.5 L0,7" fill="context-stroke" />
            </marker>
          </defs>
          {visibleEdges.map((edge) => {
            const from = byId.get(edge.source)!;
            const to = byId.get(edge.target)!;
            const anchors = ports.get(edge.id)!;
            const x = from.x + NODE_WIDTH;
            const y = from.y + anchors.sourceY;
            const endY = to.y + anchors.targetY;
            const bend = Math.max(38, (to.x - x) * 0.5);
            const focused =
              sceneMode && scene.nodes.size
                ? scene.edges.has(edge.id)
                : directEdges.has(edge.id) ||
                  (highlighted.has(edge.source) &&
                    highlighted.has(edge.target));
            const path = `M${x},${y} C${x + bend},${y} ${to.x - bend},${endY} ${to.x - 3},${endY}`;
            const carried = from.tensors.find(
              (tensor) => tensor.id === edge.tensorId,
            );
            const name = carried?.name ?? edge.tensorId;
            const probeEdge =
              probe &&
              ((contributors.has(edge.tensorId) &&
                (to.tensors.some((tensor) => tensor.id === probe.tensor.id) ||
                  (to.junction &&
                    graph.actorOrigins?.[to.id] === selectedId))) ||
                (from.junction &&
                  graph.actorOrigins?.[from.id] === selectedId &&
                  edge.tensorId === probe.tensor.id));
            const repeated =
              scene.edges.has(edge.id) && repeatedEdges.has(edge.id);
            const caption = repeated
              ? (semantics?.inputs[edge.inputIndex]?.label ??
                `Input ${edge.inputIndex + 1}`)
              : `${name.length > 18 ? name.slice(0, 16) + "…" : name} · ${edge.kind === "storage" ? "storage" : to.stage ? "stage input" : `in ${edge.inputIndex + 1}`}`;
            return (
              <g
                key={edge.id}
                className={`${focused ? "edge-highlighted" : ""} ${scene.edges.has(edge.id) ? "scene-edge" : ""} ${edge.kind === "storage" ? "edge-storage" : ""} ${probeEdge ? "probe-connection" : ""}`}
              >
                <title>{edgeDescriptions.get(edge.id)}</title>
                <path className="journey-edge-hit" d={path} />
                <path
                  className="journey-edge"
                  markerEnd={`url(#${marker})`}
                  d={path}
                />
                {scene.edges.has(edge.id) && edge.kind !== "storage" && (
                  <path className="scene-flow" d={path} />
                )}
                {view.scale >= 0.45 &&
                  labelledEdges.has(edge.id) &&
                  (!scene.edges.has(edge.id) ||
                    repeated ||
                    edge.kind === "storage") && (
                    <text
                      className="edge-label"
                      x={edge.source === focusId ? x + 46 : to.x - 46}
                      y={(edge.source === focusId ? y : endY) - 9}
                      textAnchor="middle"
                    >
                      {caption}
                    </text>
                  )}
              </g>
            );
          })}
        </svg>
        {visibleNodes.map((node) => {
          const ownerId = graph.actorOrigins?.[node.id] ?? node.id;
          if (node.junction)
            return (
              <button
                key={node.id}
                className="scene-operation-junction"
                style={{
                  left: node.x,
                  top: node.y + NODE_HEIGHT / 2 - 18,
                  width: NODE_WIDTH,
                }}
                onClick={() => onSelect(ownerId)}
                aria-label={`Focus ${node.operation?.kind ?? node.stage?.title ?? "operation"} junction`}
                title="Focus this operation and its tensors"
              >
                <span>
                  {node.operation?.kind ?? node.stage?.title ?? "operation"}
                </span>
              </button>
            );
          const actor = graph.actorRoles?.[node.id];
          const operation = node.operation;
          const group = node.stage;
          const tensorChoice = Math.min(
            tensorChoices[node.id] ?? 0,
            Math.max(0, node.tensors.length - 1),
          );
          const tensor = node.tensors[tensorChoice];
          const selectedCell =
            tensor?.id === probe?.tensor.id ? probe?.index : undefined;
          const sourceCells = tensor ? contributors.get(tensor.id) : undefined;
          const notContributing =
            probe?.result.status === "mapped" &&
            actor &&
            selectedCell === undefined &&
            !sourceCells?.length;
          const actorLabels = actor
            ? (roleLabels[actor.side].get(tensor?.id ?? "") ?? [])
            : [];
          const actorLabel =
            actorLabels.length > 1
              ? `${actorLabels.length} uses`
              : actorLabels[0];
          const inspectNode = () => {
            if (modelGraph.nodes.some((item) => item.id === ownerId))
              onSelect(ownerId, tensor?.id);
            else if (tensor) onTensorInspect(tensor.id, 0);
          };
          const multiple = node.tensors.length > 1;
          const detailed = showTensorCells(
            node,
            view,
            size,
            NODE_WIDTH,
            NODE_HEIGHT,
          );
          const rootLabel =
            tensor?.role === "input"
              ? "Input tensor"
              : tensor?.role === "parameter"
                ? "Parameter tensor"
                : "Captured tensor";
          return (
            <article
              key={node.id}
              className={`journey-node ${multiple ? "has-tensor-choices" : ""} ${group ? "journey-stage-node" : ""} ${selectedId === ownerId ? "node-selected" : ""} ${scene.nodes.has(node.id) ? "scene-actor" : ""} ${highlighted.has(ownerId) ? "node-connected" : ""} ${notContributing ? "node-not-contributing" : ""} ${operation?.status === "error" || group?.failed ? "node-error" : ""} category-node-${operation?.lesson.category ?? (group ? "layout" : "input")}`}
              style={{
                left: node.x,
                top: node.y,
                width: NODE_WIDTH,
                height: NODE_HEIGHT,
              }}
              onClick={inspectNode}
              onFocus={(event) => {
                // Keyboard navigation must bring off-screen tensors into view.
                // Pointer focus waits for the click so the target doesn't move
                // between pressing and releasing the mouse.
                const target = event.target as Element;
                if (actor && target.matches("[data-volume-index]")) return;
                if (target.matches(":focus-visible")) {
                  // Returning from the 3D dialog or tabbing across visible
                  // actors must not break the camera's operation framing.
                  const bounds = event.currentTarget.getBoundingClientRect();
                  const canvas = frame.current?.getBoundingClientRect();
                  if (
                    actor &&
                    canvas &&
                    bounds.left >= canvas.left &&
                    bounds.right <= canvas.right &&
                    bounds.top >= canvas.top &&
                    bounds.bottom <= canvas.bottom
                  )
                    return;
                  frame.current?.scrollTo(0, 0);
                  focus(node);
                }
              }}
              aria-label={
                group
                  ? `Stage: ${group.title}, ${group.operationIds.length} operations, ${group.path}`
                  : operation
                    ? `Step ${operation.index + 1}: ${operation.kind}, ${tensor?.name ?? (operation.status === "error" ? "execution error" : "no tensor output")}, shape ${tensor ? tensor.shape.join(", ") || "scalar" : "none"}`
                    : `${rootLabel} ${tensor?.name}, shape ${tensor?.shape.join(", ") || "scalar"}`
              }
              data-node-id={node.id}
              data-owner-id={ownerId}
              data-tensor-id={tensor?.id}
              title={
                group
                  ? `${group.path} · steps ${group.start_index + 1}–${group.end_index}. Select to focus this recorded call.`
                  : (operation?.lesson.summary ??
                    (tensor?.role === "input"
                      ? "The original input tensor"
                      : "A tensor captured before its first recorded use"))
              }
            >
              <div className="journey-node-top">
                <button
                  className="journey-node-select"
                  aria-label={
                    group
                      ? `Focus stage ${group.title}, ${group.path}, ${group.id}`
                      : operation
                        ? `Step ${operation.index + 1}: ${operation.kind}, ${tensor?.name ?? (operation.status === "error" ? "error" : "no tensor output")}, shape ${tensor?.shape.join(", ") ?? "none"}`
                        : `${rootLabel} ${tensor?.name}, shape ${tensor?.shape.join(", ")}`
                  }
                  aria-pressed={selectedId === ownerId}
                  title={tensor?.name}
                  onClick={(e) => {
                    e.stopPropagation();
                    inspectNode();
                  }}
                >
                  <b>{group?.title ?? tensor?.name ?? operation?.kind}</b>
                </button>
                <span title={actorLabels.join(" · ") || undefined}>
                  {actorLabel ??
                    (group
                      ? `${group.operationIds.length} OPS`
                      : (
                            outputIds
                              ? tensor && returned.has(tensor.id)
                              : node.terminal
                          )
                        ? "OUTPUT"
                        : !operation
                          ? tensor?.role === "input"
                            ? "INPUT"
                            : "CAPTURED"
                          : `${operation.index + 1}`.padStart(2, "0"))}
                </span>
              </div>
              {multiple && (
                <div
                  className="node-tensor-choice"
                  onClick={(event) => event.stopPropagation()}
                >
                  <select
                    aria-label={
                      group
                        ? `Tensor shown for stage ${group.title}`
                        : `Tensor shown at step ${(operation?.index ?? -1) + 1}`
                    }
                    title={`${tensorChoice + 1} of ${node.tensors.length}: ${tensor.name} [${tensor.shape.join(", ")}]`}
                    value={tensorChoice}
                    onChange={(event) => {
                      const choice = Number(event.target.value);
                      setTensorChoices((previous) => ({
                        ...previous,
                        [node.id]: choice,
                      }));
                      setPreviewNode(node.id);
                      onTensorChoice?.(node.id, node.tensors[choice].id);
                    }}
                  >
                    {node.tensors.map((item, index) => (
                      <option key={`${item.id}-${index}`} value={index}>
                        {index + 1} · {item.name} [{item.shape.join(", ")}]
                      </option>
                    ))}
                  </select>
                </div>
              )}
              {tensor && detailed ? (
                <TensorGlyph
                  tensor={tensor}
                  selected={selectedCell ?? sourceCells?.[0]}
                  highlights={
                    selectedCell !== undefined ? [selectedCell] : sourceCells
                  }
                  keyboardNavigation={actor?.side === "output"}
                  onSelect={(index) =>
                    actor?.side === "output" && onTraceCell
                      ? onTraceCell(tensor.id, index)
                      : onTensorInspect(tensor.id, index)
                  }
                />
              ) : tensor ? (
                <div
                  className="node-tensor-overview"
                  title="Shape overview. Zoom in or enlarge to inspect actual cells."
                >
                  <span>
                    {tensor.shape.length
                      ? `${tensor.shape.length} axes`
                      : "scalar"}
                  </span>
                  <b>{tensor.numel.toLocaleString()}</b>
                  <small>elements · zoom for cells</small>
                </div>
              ) : (
                <div className="node-failure">
                  {operation?.status === "error" || group?.failed
                    ? "Execution stopped"
                    : "No tensor returned"}
                </div>
              )}
              <div
                className="node-shape"
                title={tensor ? `[${tensor.shape.join(", ")}]` : undefined}
              >
                {tensor
                  ? tensor.shape.length
                    ? `[${tensor.shape.join(", ")}]`
                    : "scalar · shape []"
                  : "No output"}
              </div>
              <div className="node-operation">
                <span className="operation-dot" />
                <span>
                  {group
                    ? `Steps ${group.start_index + 1}–${group.end_index}`
                    : operation?.mutations?.length
                      ? `${operation.kind} · in place`
                      : (operation?.kind ?? rootLabel)}
                </span>
                <small>
                  {node.tensors.length > 1
                    ? `${node.tensors.length} tensors`
                    : group?.failed
                      ? "Stopped"
                      : node.parameterCount && actor?.side !== "output"
                        ? "+ weights"
                        : operation?.source
                          ? `L${operation.source.line}`
                          : ""}
                </small>
                {tensor && (
                  <button
                    className="node-enlarge"
                    aria-label={
                      group
                        ? `Enlarge ${group.title} tensor ${tensor.name}`
                        : operation
                          ? `Enlarge step ${operation.index + 1} tensor ${tensor.name}`
                          : `Enlarge ${rootLabel.toLowerCase()} ${tensor.name}`
                    }
                    title="Enlarge tensor in 3D"
                    onClick={(event) => {
                      event.stopPropagation();
                      onTensorInspect(
                        tensor.id,
                        selectedCell ?? sourceCells?.[0] ?? 0,
                      );
                    }}
                  >
                    <Maximize size={12} />
                  </button>
                )}
                {group && (
                  <button
                    className="node-enlarge stage-expand-button"
                    aria-label={`Expand stage ${group.title}, ${group.path}, ${group.id}`}
                    title="See inside this stage"
                    onClick={(event) => {
                      event.stopPropagation();
                      onStageToggle(group.id);
                    }}
                  >
                    <UnfoldHorizontal size={13} />
                  </button>
                )}
              </div>
            </article>
          );
        })}
        {motion && selectedId && focusKey > 0 && !probe && (
          <CanvasCellMotion
            world={world}
            graph={graph}
            operationId={selectedId}
            tensors={motion.tensors}
            plan={motion.plan}
            clock={motion.clock}
            geometryKey={`${Object.entries(tensorChoices)
              .map(([id, choice]) => `${id}:${choice}`)
              .join(
                "|",
              )}/${size.width}/${size.height}/${view.x}/${view.y}/${view.scale}`}
            onInspect={(tensorId, index, keyboard) => {
              if (keyboard && onTraceCell)
                probeFocus.current = { tensorId, index };
              (onTraceCell ?? onTensorInspect)(tensorId, index);
            }}
          />
        )}
      </div>
      {probe && onClearTrace ? (
        <CanvasCellProbe
          probe={probe}
          onInspect={() => onTensorInspect(probe.tensor.id, probe.index)}
          onClear={clearTrace}
        />
      ) : (
        sceneMode &&
        semantics &&
        graph.sceneOperationId && (
          <div className="canvas-scene-caption" aria-live="polite">
            <strong>{semantics.title}</strong>
            <span title={semantics.summary}>{semantics.summary}</span>
          </div>
        )
      )}
      <div className="canvas-zoom" onPointerDown={(e) => e.stopPropagation()}>
        <button onClick={() => zoom(1.25)} aria-label="Zoom in" title="Zoom in">
          <Plus size={17} />
        </button>
        <span aria-live="polite">{Math.round(view.scale * 100)}%</span>
        <button
          onClick={() => zoom(0.8)}
          aria-label="Zoom out"
          title="Zoom out"
        >
          <Minus size={17} />
        </button>
        <div className="control-divider" />
        <button
          onClick={() => {
            onOverview();
            fit();
          }}
          aria-label="Fit entire journey"
          title="Fit entire journey (0)"
        >
          <Maximize size={16} />
        </button>
      </div>
      {graph.nodes.length > 10 && (
        <button
          className="canvas-minimap"
          title="Fit entire journey"
          aria-label="Journey overview — fit all"
          onClick={() => {
            onOverview();
            fit();
          }}
        >
          <svg
            viewBox={`0 0 ${graph.width} ${graph.height}`}
            aria-hidden="true"
          >
            {visibleEdges.map((edge) => {
              const a = byId.get(edge.source)!;
              const b = byId.get(edge.target)!;
              return (
                <path
                  key={edge.id}
                  d={`M${a.x + NODE_WIDTH / 2},${a.y + NODE_HEIGHT / 2}L${b.x + NODE_WIDTH / 2},${b.y + NODE_HEIGHT / 2}`}
                />
              );
            })}
            {visibleNodes.map((node) => (
              <rect
                className={selectedId === node.id ? "minimap-selected" : ""}
                key={node.id}
                x={node.x}
                y={node.y}
                width={NODE_WIDTH}
                height={NODE_HEIGHT}
                rx="12"
              />
            ))}
            <rect
              className="minimap-viewport"
              x={-view.x / view.scale}
              y={-view.y / view.scale}
              width={size.width / view.scale}
              height={size.height / view.scale}
            />
          </svg>
        </button>
      )}
    </div>
  );
}
