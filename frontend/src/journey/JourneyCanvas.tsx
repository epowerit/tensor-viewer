import {
  memo,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { Maximize, Minus, Plus, Repeat, UnfoldHorizontal } from "lucide-react";
import type { JourneyNode } from "./graph";
import type { LoopGraph, LoopView } from "./loops";
import type { TensorLight } from "../tensors/TensorVolume";
import { ancestors, descendants, NODE_HEIGHT, NODE_WIDTH } from "./graph";
import { edgeWidths, flowNeighbor, mainPath, spreadJumps } from "./flow";
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
  overviewView,
  reframeCanvas,
  showTensorCells,
  type CanvasViewport,
  type Viewport,
} from "./viewport";
import { AxisInkContext, InkShape } from "../tensors/InkShape";
import "./journeyCanvas.css";
import { kindName } from "../operations/kindName";
import { stageLabel } from "./stages";

type Props = {
  graph: LoopGraph;
  selectedId: string | null;
  /**
   * The furthest operation index playback has activated. Later tensors are
   * unlit glass (no values yet); earlier ones keep glowing.
   */
  reachedThrough?: number;
  /** Tensors past this index were unlit when the current step began. */
  kindleAbove?: number;
  /** Pixels at the top the fitted overview keeps clear. */
  topInset?: number;
  /** Nodes that wrote a name followed from the tensor shelf. */
  threaded?: ReadonlySet<string>;
  /** The folded loop whose repeats are playing. */
  activeLoopId?: string;
  onLoopIteration?: (loopId: string, iteration: number) => void;
  onLoopSelect?: (loopId: string) => void;
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
  reachedThrough,
  kindleAbove = -1,
  topInset = 0,
  threaded,
  activeLoopId,
  onLoopIteration,
  onLoopSelect,
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
  const inkFor = useContext(AxisInkContext);
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
        ? {
            ...projectOperationScene(modelGraph, selectedId, motion.tensors),
            loops: modelGraph.loops,
            passLoops: modelGraph.passLoops,
          }
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
  // Hovering a node traces its tensor's lineage: where it came from and
  // where it goes. A short delay keeps passing pointers from flickering.
  const [hoverNode, setHoverNode] = useState<string | null>(null);
  const hoverTimer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(hoverTimer.current), []);
  const lineage = useMemo(
    () =>
      hoverNode && byId.has(hoverNode)
        ? {
            up: ancestors(graph, hoverNode),
            down: descendants(graph, hoverNode),
          }
        : null,
    [graph, hoverNode, byId],
  );
  const lineageOf = (edge: { source: string; target: string }) =>
    !lineage
      ? ""
      : lineage.up.has(edge.source) && lineage.up.has(edge.target)
        ? "edge-upstream"
        : lineage.down.has(edge.source) && lineage.down.has(edge.target)
          ? "edge-downstream"
          : "edge-off-lineage";
  // Thicker connections carry more data.
  const widths = useMemo(() => edgeWidths(graph), [graph]);
  // The hovered node's main line of flow, as names and shapes.
  const ribbon = useMemo(
    () => (hoverNode ? mainPath(graph, hoverNode) : []),
    [graph, hoverNode],
  );
  const jumps = useMemo(() => spreadJumps(ribbon), [ribbon]);
  // Tensors read by three or more steps say so at their output.
  const fanOut = new Map<string, number>();
  for (const edge of visibleEdges)
    if (edge.kind !== "storage")
      fanOut.set(edge.source, (fanOut.get(edge.source) ?? 0) + 1);
  const loopFrames = frameLoops(graph, visibleNodes);
  // Light follows execution: unlit before the step runs, on fire while it is
  // the active result (or a repeating loop's body), lit after. The nodes and
  // the minimap share it.
  function lightOf(node: JourneyNode): TensorLight {
    const ownerId = graph.actorOrigins?.[node.id] ?? node.id;
    const ran = node.operation?.index ?? node.stage?.start_index;
    if (
      activeLoopId &&
      loopFrames.some(
        (frame) =>
          frame.loop.id === activeLoopId && frame.memberIds.has(node.id),
      )
    )
      return "active";
    if (
      reachedThrough !== undefined &&
      ran !== undefined &&
      ran > reachedThrough
    )
      return "pending";
    return focusKey > 0 &&
      selectedId === ownerId &&
      graph.actorRoles?.[node.id]?.side !== "input"
      ? "active"
      : "lit";
  }
  // How far data has flowed: edges into steps not yet run wait, faint and
  // dashed; edges whose step has run carry their tensor, bright.
  function flowOf(edge: { target: string }): string {
    if (reachedThrough === undefined) return "";
    const target = byId.get(edge.target);
    if (!target) return "";
    return lightOf(target) === "pending" ? "edge-waiting" : "edge-carried";
  }
  const passFrames = framePasses(graph, visibleNodes);
  // A step inside a loop frame leaves room for the frame's header above it,
  // so the scene caption never covers it.
  const loopHeadroom = loopFrames.some(
    (frame) => !!selectedId && frame.memberIds.has(selectedId),
  )
    ? 64
    : 0;
  const focusedLoop = loopFrames.find(
    (frame) => `loop:${frame.loop.id}` === selectedId,
  );
  // Reframe on layout changes only: a loop showing another iteration swaps
  // tensors in place and must not move the camera.
  const structure = useMemo(() => ({}), [layoutSignature(modelGraph)]);
  const layoutStructure = useMemo(() => ({}), [layoutSignature(graph)]);

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
    // Room kept clear at the top, such as for a failed run's error card.
    return overviewView(modelGraph, size, topInset);
  }
  function fit() {
    if (!size.width || !size.height) return;
    manualNavigation();
    setViewport((previous) =>
      fitCanvas(previous, structure, size, fittedView()),
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
  /** Frame a loop's body with its header and return arc. */
  function loopView(frame: LoopFrame): Viewport {
    const top = frame.top - arcRise(frame) - 12;
    const width = frame.right - frame.left,
      height = frame.bottom - top;
    // The whole loop, header and arc included, even when that means small.
    const scale = Math.max(
      0.08,
      Math.min(1.1, (size.width - 96) / width, (size.height - 250) / height),
    );
    return {
      scale,
      x: size.width / 2 - ((frame.left + frame.right) / 2) * scale,
      y: (size.height - 110) / 2 - ((top + frame.bottom) / 2) * scale,
    };
  }
  function focus(node: JourneyNode) {
    manualNavigation();
    setView((previous) => focusedView(node, previous));
  }
  useEffect(() => {
    const selected = graph.nodes.find((node) => node.id === selectedId);
    setViewport((previous) =>
      reframeCanvas(
        previous,
        {
          graph: structure,
          layout: layoutStructure,
          focusKey,
          selectionKey,
          size,
        },
        fittedView(),
        selected
          ? (current) =>
              sceneMode
                ? followPlayback ||
                  previous.frame?.selectionKey !== selectionKey
                  ? sceneView(graph, selected.id, size, current, loopHeadroom)
                  : current
                : focusedView(selected, current)
          : focusedLoop
            ? (current) =>
                followPlayback || previous.frame?.selectionKey !== selectionKey
                  ? loopView(focusedLoop)
                  : current
            : undefined,
      ),
    );
    // Drawers resize around the same world point; inspection has its own framing.
  }, [
    structure,
    layoutStructure,
    focusKey,
    selectionKey,
    size.width,
    size.height,
    sceneMode,
  ]);

  useEffect(() => {
    if (followPlayback && focusedLoop) setView(loopView(focusedLoop));
    else if (followPlayback && sceneMode && focusKey > 0 && selectedId)
      setView((previous) =>
        sceneView(graph, selectedId, size, previous, loopHeadroom),
      );
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
      aria-keyshortcuts="Space Enter Escape Home + - [ ]"
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
        // Alt+→ and Alt+← walk the flow: to the step that reads the selected
        // tensor, or back to the one that made its first operand.
        if (
          event.altKey &&
          (event.key === "ArrowRight" || event.key === "ArrowLeft") &&
          selectedId &&
          !(event.target as Element).closest("input, select, textarea")
        ) {
          const next = flowNeighbor(
            modelGraph,
            selectedId,
            event.key === "ArrowRight" ? "forward" : "back",
          );
          event.preventDefault();
          if (next) onSelect(next);
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
        // [ and ] step the iteration shown by the loop around the selection,
        // or by the only loop on screen.
        if ((event.key === "[" || event.key === "]") && onLoopIteration) {
          const around =
            loopFrames.find(
              (frame) =>
                `loop:${frame.loop.id}` === selectedId ||
                (!!selectedId && frame.memberIds.has(selectedId)),
            ) ?? (loopFrames.length === 1 ? loopFrames[0] : undefined);
          if (around) {
            event.preventDefault();
            const count = around.loop.iterations.length;
            const next =
              (around.loop.shown - 1 + (event.key === "]" ? 1 : count - 1)) %
              count;
            onLoopIteration(around.loop.id, next + 1);
          }
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
          // Lines and overview labels stay legible at any zoom.
          ["--canvas-scale" as string]: view.scale,
        }}
      >
        <svg
          className="journey-edges"
          width={graph.width}
          height={graph.height}
          overflow="visible"
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
            <marker
              id={`${marker}-loop`}
              markerWidth="9"
              markerHeight="9"
              refX="7"
              refY="4.5"
              orient="auto"
            >
              <path d="M0,0 L9,4.5 L0,9" fill="context-stroke" />
            </marker>
          </defs>
          {passFrames.map((frame) => (
            <g key={frame.key} className="loop-pass">
              <rect
                x={frame.left}
                y={frame.top}
                width={frame.right - frame.left}
                height={frame.bottom - frame.top}
                rx={20}
              />
              <text x={frame.left + 14} y={frame.top + 17}>
                {`↻ pass ${frame.pass} of ${frame.count}${frame.pass === 1 ? ` · ${frame.text} · passes differ` : ""}`}
              </text>
            </g>
          ))}
          {loopFrames.map((frame) => (
            <g
              key={frame.loop.id}
              className={`loop-frame ${frame.loop.id === activeLoopId ? "is-repeating" : ""}`}
            >
              <rect
                x={frame.left}
                y={frame.top}
                width={frame.right - frame.left}
                height={frame.bottom - frame.top}
                rx={26}
              />
              {/* The return arc: the end of the body feeds its start again. */}
              <path
                className="loop-return"
                markerEnd={`url(#${marker}-loop)`}
                d={`M${frame.right - 44},${frame.top} C${frame.right - 44},${frame.top - arcRise(frame) * 1.33} ${frame.left + 44},${frame.top - arcRise(frame) * 1.33} ${frame.left + 44},${frame.top - 2}`}
              />
            </g>
          ))}
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
                className={`${focused ? "edge-highlighted" : ""} ${scene.edges.has(edge.id) ? "scene-edge" : ""} ${edge.kind === "storage" ? "edge-storage" : ""} ${probeEdge ? "probe-connection" : ""} ${flowOf(edge)} ${lineageOf(edge)}`}
              >
                <title>{edgeDescriptions.get(edge.id)}</title>
                <path className="journey-edge-hit" d={path} />
                <path
                  className="journey-edge"
                  markerEnd={`url(#${marker})`}
                  d={path}
                  style={
                    {
                      "--edge-width": widths.get(edge.id) ?? 1.4,
                    } as React.CSSProperties
                  }
                />
                {lineage &&
                  view.scale >= 0.45 &&
                  carried &&
                  lineageOf(edge) !== "edge-off-lineage" && (
                    // The traced lineage reads its shapes along the way.
                    <text
                      className="edge-shape"
                      x={(x + to.x) / 2}
                      y={(y + endY) / 2 - 7}
                      textAnchor="middle"
                    >
                      {`[${carried.shape.join(", ")}]`}
                    </text>
                  )}
                {scene.edges.has(edge.id) && edge.kind !== "storage" && (
                  <path className="scene-flow" d={path} />
                )}
                {view.scale < 0.6 &&
                  edge.kind !== "storage" &&
                  !!selectedId &&
                  (graph.actorOrigins?.[edge.target] ?? edge.target) ===
                    selectedId &&
                  flowOf(edge) === "edge-carried" && (
                    // Zoomed out, tensors visibly travel into the current step.
                    <circle
                      className="flow-pulse"
                      r={Math.min(14, 3.5 / view.scale)}
                    >
                      <animateMotion
                        dur="1.2s"
                        repeatCount="indefinite"
                        path={path}
                      />
                    </circle>
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
          {view.scale >= 0.45 &&
            visibleNodes.map((node) => {
              const uses = fanOut.get(node.id) ?? 0;
              return uses >= 3 ? (
                <text
                  key={`fan-${node.id}`}
                  className="fan-out"
                  x={node.x + NODE_WIDTH + 8}
                  y={node.y + 18}
                >
                  {`→ ${uses}`}
                  <title>{`Read by ${uses} steps`}</title>
                </text>
              ) : null;
            })}
          {view.scale < 0.6 &&
            visibleNodes
              .filter(
                (node) =>
                  !!selectedId &&
                  (graph.actorOrigins?.[node.id] ?? node.id) === selectedId &&
                  graph.actorRoles?.[node.id]?.side !== "input",
              )
              .map((node) => (
                // Where playback is, readable from the overview.
                <rect
                  key={`current-${node.id}`}
                  className="current-step-ring"
                  x={node.x - 10}
                  y={node.y - 10}
                  width={NODE_WIDTH + 20}
                  height={NODE_HEIGHT + 20}
                  rx={22}
                />
              ))}
        </svg>
        {loopFrames.map(({ loop, left, top, right }) => {
          const count = loop.iterations.length;
          return (
            <div
              key={loop.id}
              className={`loop-frame-header ${loop.id === activeLoopId ? "is-repeating" : ""}`}
              style={{
                left: left + 16,
                top: top + 8,
                width: right - left - 32,
              }}
            >
              <button
                className="loop-frame-title"
                aria-label={`Loop ${loop.text}, ${count} identical iterations drawn once. Play its repeats.`}
                title={`Line ${loop.line}: ${loop.text}. Every iteration does the same work, so the body is drawn once. Select to play the repeats.`}
                onClick={() => onLoopSelect?.(loop.id)}
              >
                <Repeat size={13} aria-hidden="true" />
                <code>{loop.text}</code>
                <small>
                  ×{count} · L{loop.line}
                </small>
              </button>
              <PassStrip loop={loop} onShow={onLoopIteration} />
              <div
                className="loop-iterations"
                role="group"
                aria-label={`Iteration shown for ${loop.text}`}
              >
                {count <= 10 ? (
                  loop.iterations.map((_, i) => (
                    <button
                      key={i}
                      aria-pressed={loop.shown === i + 1}
                      aria-label={`Show iteration ${i + 1} of ${count}`}
                      title={`Iteration ${i + 1} of ${count}${rangeText(loop.ranges[i])} · [ and ] on the canvas step through passes`}
                      onClick={() => onLoopIteration?.(loop.id, i + 1)}
                    />
                  ))
                ) : (
                  <input
                    type="range"
                    min={1}
                    max={count}
                    value={loop.shown}
                    aria-label={`Iteration shown, of ${count}`}
                    onChange={(event) =>
                      onLoopIteration?.(loop.id, Number(event.target.value))
                    }
                  />
                )}
                <span aria-live="polite">
                  {loop.shown}/{count}
                </span>
              </div>
            </div>
          );
        })}
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
                aria-label={`Focus ${node.operation ? kindName(node.operation.kind) : (node.stage?.title ?? "operation")} junction`}
                title="Focus this operation and its tensors"
              >
                <span>
                  {node.operation
                    ? kindName(node.operation.kind)
                    : (node.stage?.title ?? "operation")}
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
          const ran = operation?.index ?? group?.start_index;
          const light = lightOf(node);
          const rootLabel =
            tensor?.role === "input"
              ? "Input tensor"
              : tensor?.role === "parameter"
                ? "Parameter tensor"
                : "Captured tensor";
          return (
            <article
              key={node.id}
              className={`journey-node node-light-${light} ${threaded?.has(ownerId) ? "node-threaded" : ""} ${multiple ? "has-tensor-choices" : ""} ${group ? "journey-stage-node" : ""} ${selectedId === ownerId ? "node-selected" : ""} ${scene.nodes.has(node.id) ? "scene-actor" : ""} ${highlighted.has(ownerId) ? "node-connected" : ""} ${notContributing ? "node-not-contributing" : ""} ${operation?.status === "error" || group?.failed ? "node-error" : ""} ${lineage ? (lineage.up.has(node.id) || lineage.down.has(node.id) ? "node-in-lineage" : "node-off-lineage") : ""} category-node-${operation?.lesson.category ?? (group ? "layout" : "input")}`}
              style={{
                left: node.x,
                top: node.y,
                width: NODE_WIDTH,
                height: NODE_HEIGHT,
              }}
              onClick={inspectNode}
              onMouseEnter={() => {
                window.clearTimeout(hoverTimer.current);
                hoverTimer.current = window.setTimeout(
                  () => setHoverNode(node.id),
                  160,
                );
              }}
              onMouseLeave={() => {
                window.clearTimeout(hoverTimer.current);
                setHoverNode(null);
              }}
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
                    ? `Step ${operation.index + 1}: ${kindName(operation.kind)}, ${tensor?.name ?? (operation.status === "error" ? "execution error" : "no tensor output")}, shape ${tensor ? tensor.shape.join(", ") || "scalar" : "none"}`
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
                      ? `Focus stage ${stageLabel(group)}`
                      : operation
                        ? `Step ${operation.index + 1}: ${kindName(operation.kind)}, ${tensor?.name ?? (operation.status === "error" ? "error" : "no tensor output")}, shape ${tensor?.shape.join(", ") ?? "none"}`
                        : `${rootLabel} ${tensor?.name}, shape ${tensor?.shape.join(", ")}`
                  }
                  aria-pressed={selectedId === ownerId}
                  title={tensor?.name}
                  onClick={(e) => {
                    e.stopPropagation();
                    inspectNode();
                  }}
                >
                  <b>
                    {group?.title ??
                      tensor?.name ??
                      (operation && kindName(operation.kind))}
                  </b>
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
                <LitGlyph
                  // A new tensor (another loop iteration) ignites again.
                  key={tensor.id}
                  clock={motion?.clock}
                  playing={playing}
                  kindling={
                    light === "active" && ran !== undefined && ran > kindleAbove
                  }
                  light={light}
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
                {tensor ? (
                  tensor.shape.length ? (
                    <InkShape shape={tensor.shape} ink={inkFor?.(tensor)} />
                  ) : (
                    "scalar · shape []"
                  )
                ) : (
                  "No output"
                )}
              </div>
              <div className="node-operation">
                <span className="operation-dot" />
                <span>
                  {group
                    ? `Steps ${group.start_index + 1}–${group.end_index}`
                    : operation?.mutations?.length
                      ? `${kindName(operation.kind)} · in place`
                      : operation
                        ? kindName(operation.kind)
                        : rootLabel}
                </span>
                {group && <StageLoops loops={graph.loops} node={node} />}
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
                    aria-label={`Expand stage ${stageLabel(group)}`}
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
        {view.scale < 0.45 && (
          <div className="overview-labels" aria-hidden="true">
            {visibleNodes.map((node) => {
              const current =
                !!selectedId &&
                (graph.actorOrigins?.[node.id] ?? node.id) === selectedId &&
                graph.actorRoles?.[node.id]?.side !== "input";
              // Too far out for every name: keep only where playback is.
              if (view.scale < 0.2 && !current) return null;
              const tensor = node.tensors[0];
              const name =
                node.stage?.title ??
                tensor?.name ??
                (node.operation ? kindName(node.operation.kind) : "");
              return (
                <div
                  key={node.id}
                  className={`overview-label overview-${lightOf(node)} ${current ? "overview-current" : ""}`}
                  style={{
                    left: node.x + NODE_WIDTH / 2,
                    top: node.y + NODE_HEIGHT / 2,
                    maxWidth: current ? 220 : (NODE_WIDTH + 70) * view.scale,
                  }}
                >
                  <b>{name}</b>
                  {tensor && <small>[{tensor.shape.join(", ")}]</small>}
                </div>
              );
            })}
          </div>
        )}
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
      {ribbon.length > 1 && (
        <ol className="flow-ribbon" aria-label="Main path to the hovered node">
          {(ribbon.length > 9
            ? [...ribbon.slice(0, 3), null, ...ribbon.slice(-5)]
            : ribbon
          ).map((node, i) =>
            node ? (
              <li key={node.id}>
                {i > 0 && <span className="flow-ribbon-arrow">→</span>}
                <span>
                  <b>
                    {node.stage?.title ??
                      node.tensors[0]?.name ??
                      (node.operation ? kindName(node.operation.kind) : "")}
                  </b>
                  {node.tensors[0] && (
                    <small>[{node.tensors[0].shape.join(", ")}]</small>
                  )}
                  {typeof node.tensors[0]?.histogram?.std === "number" && (
                    <small
                      className={`flow-spread ${jumps.has(node.id) ? "flow-jump" : ""}`}
                    >
                      σ {formatSpread(node.tensors[0].histogram.std)}
                      {jumps.has(node.id) &&
                        ` ${jumps.get(node.id)! > 1 ? "↑" : "↓"}×${formatSpread(
                          jumps.get(node.id)! > 1
                            ? jumps.get(node.id)!
                            : 1 / jumps.get(node.id)!,
                        )}`}
                    </small>
                  )}
                </span>
              </li>
            ) : (
              <li key={`gap-${i}`} className="flow-ribbon-gap">
                <span className="flow-ribbon-arrow">→</span>…
              </li>
            ),
          )}
        </ol>
      )}
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
                className={`minimap-${lightOf(node)} ${threaded?.has(graph.actorOrigins?.[node.id] ?? node.id) ? "minimap-threaded" : ""} ${selectedId === node.id ? "minimap-selected" : ""}`}
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

type LoopFrame = {
  loop: LoopView;
  memberIds: Set<string>;
  left: number;
  top: number;
  right: number;
  bottom: number;
};

/** A folded loop's frame surrounds the visible nodes of its body. */
function frameLoops(graph: LoopGraph, nodes: JourneyNode[]): LoopFrame[] {
  const loops = graph.loops ?? [];
  if (!loops.length) return [];
  const operationsOf = (node: JourneyNode) =>
    node.repOperationIds ??
    node.stage?.operationIds ??
    (node.operation ? [graph.actorOrigins?.[node.id] ?? node.id] : []);
  const found = loops.flatMap((loop) => {
    const members = nodes.filter((node) => {
      const ids = operationsOf(node);
      return (
        !node.junction &&
        ids.length > 0 &&
        ids.every((id) => loop.members.has(id))
      );
    });
    return members.length ? [{ loop, members }] : [];
  });
  return found.map(({ loop, members }) => {
    // Leave room inside for the frames of loops nested in this one.
    const inner = found.filter(
      (other) =>
        other.loop.depth > loop.depth &&
        other.members.every((node) => members.includes(node)),
    ).length;
    const pad = 22 + inner * 20;
    const left = Math.min(...members.map((node) => node.x)) - pad,
      right = Math.max(...members.map((node) => node.x + NODE_WIDTH)) + pad;
    // Wide enough for the header: the loop's code and its iteration pips.
    const grow = Math.max(0, 360 - (right - left)) / 2;
    return {
      loop,
      memberIds: new Set(members.map((node) => node.id)),
      left: left - grow,
      top: Math.min(...members.map((node) => node.y)) - pad - 34 - inner * 26,
      right: right + grow,
      bottom: Math.max(...members.map((node) => node.y + NODE_HEIGHT)) + pad,
    };
  });
}

/** The return arc rises with the loop's width, so it reads as an arc. */
const arcRise = (frame: { left: number; right: number }) =>
  Math.min(220, 44 + (frame.right - frame.left) * 0.05);

/**
 * A loop whose passes differ is drawn in full; each pass gets a light outline
 * and its number, so the repetition still reads.
 */
function framePasses(graph: LoopGraph, nodes: JourneyNode[]) {
  return (graph.passLoops ?? []).flatMap((loop) =>
    loop.iterations.flatMap((ops, i) => {
      const pass = new Set(ops);
      const members = nodes.filter((node) => {
        const ids =
          node.repOperationIds ??
          node.stage?.operationIds ??
          (node.operation ? [graph.actorOrigins?.[node.id] ?? node.id] : []);
        return (
          !node.junction && ids.length > 0 && ids.every((id) => pass.has(id))
        );
      });
      if (!members.length) return [];
      return [
        {
          key: `${loop.id}#${i + 1}`,
          pass: i + 1,
          count: loop.iterations.length,
          text: loop.text,
          left: Math.min(...members.map((node) => node.x)) - 14,
          top: Math.min(...members.map((node) => node.y)) - 30,
          right: Math.max(...members.map((node) => node.x + NODE_WIDTH)) + 14,
          bottom: Math.max(...members.map((node) => node.y + NODE_HEIGHT)) + 14,
        },
      ];
    }),
  );
}

const layoutSignature = (graph: LoopGraph) =>
  `${graph.nodes.map((node) => `${node.id}@${node.x},${node.y}`).join("|")}#${graph.edges.length}`;

const noClock = () => () => {};

/**
 * A node's tensor. A step running for the first time stays unlit while its
 * values arrive, then ignites; a tensor that already glowed never goes dark.
 */
function LitGlyph({
  clock,
  playing,
  kindling,
  light,
  onSelect,
  ...glyph
}: React.ComponentProps<typeof TensorGlyph> & {
  clock?: SceneClock;
  playing: boolean;
  kindling: boolean;
  light: TensorLight;
}) {
  const receiving = useSyncExternalStore(
    clock?.subscribe ?? noClock,
    () => !!clock && playing && kindling && clock.getSnapshot() < 0.72,
    () => false,
  );
  // The handler changes every render; the glyph always calls the latest one,
  // so it can skip re-rendering when only the handler changed.
  const latest = useRef(onSelect);
  latest.current = onSelect;
  const select = useCallback((index: number) => latest.current?.(index), []);
  return (
    <StableGlyph
      {...glyph}
      onSelect={onSelect ? select : undefined}
      light={receiving ? "receiving" : light}
    />
  );
}

const sameCells = (a?: readonly number[], b?: readonly number[]) =>
  a === b ||
  (!!a && !!b && a.length === b.length && a.every((cell, i) => cell === b[i]));

/** Cube glyphs are the canvas's costliest part: draw one only when it changed. */
const StableGlyph = memo(
  TensorGlyph,
  (previous, next) =>
    previous.tensor === next.tensor &&
    previous.selected === next.selected &&
    previous.light === next.light &&
    previous.keyboardNavigation === next.keyboardNavigation &&
    !!previous.onSelect === !!next.onSelect &&
    previous.onSelect === next.onSelect &&
    sameCells(previous.highlights, next.highlights),
);

const formatRange = (value: number) =>
  Math.abs(value) >= 1000 || (value !== 0 && Math.abs(value) < 0.01)
    ? value.toExponential(1)
    : value.toFixed(2);
const rangeText = (range: LoopView["ranges"][number]) =>
  range
    ? ` · ${range.name} from ${formatRange(range.min)} to ${formatRange(range.max)}`
    : "";

/**
 * How the value a loop carries evolves: one bar per pass, spanning the range
 * of the body's last result, on a shared scale. Growing or drifting values
 * show up without stepping through the passes.
 */
function PassStrip({
  loop,
  onShow,
}: {
  loop: LoopView;
  onShow?: (loopId: string, iteration: number) => void;
}) {
  const ranges = loop.ranges;
  if (ranges.length > 32 || ranges.some((range) => !range)) return null;
  const low = Math.min(...ranges.map((range) => range!.min));
  const high = Math.max(...ranges.map((range) => range!.max));
  const span = high - low || 1;
  const step = 8,
    height = 18;
  const y = (value: number) => 2 + (1 - (value - low) / span) * (height - 4);
  return (
    <svg
      className="loop-pass-strip"
      width={ranges.length * step + 4}
      height={height}
      role="img"
      aria-label={`${ranges[0]!.name} per pass: ${ranges
        .map(
          (range, i) =>
            `pass ${i + 1} from ${formatRange(range!.min)} to ${formatRange(range!.max)}`,
        )
        .join("; ")}`}
    >
      <title>{`${ranges[0]!.name} range per pass, ${formatRange(low)} to ${formatRange(high)}`}</title>
      {ranges.map((range, i) => (
        <line
          key={i}
          className={loop.shown === i + 1 ? "shown" : undefined}
          x1={4 + i * step}
          x2={4 + i * step}
          y1={y(range!.max)}
          y2={Math.max(y(range!.min), y(range!.max) + 1.5)}
          onClick={() => onShow?.(loop.id, i + 1)}
        />
      ))}
    </svg>
  );
}

/** A collapsed stage that hides a folded loop says so. */
function StageLoops({
  loops,
  node,
}: {
  loops?: LoopView[];
  node: JourneyNode;
}) {
  const ids = node.repOperationIds ?? node.stage?.operationIds ?? [];
  const inside = (loops ?? []).filter(
    (loop) =>
      ids.some((id) => loop.members.has(id)) &&
      !ids.every((id) => loop.members.has(id)),
  );
  if (!inside.length) return null;
  return (
    <i
      className="node-loop-badge"
      title={inside
        .map(
          (loop) =>
            `Line ${loop.line}: ${loop.text}, ${loop.iterations.length} identical passes. Expand to see it drawn once.`,
        )
        .join("\n")}
    >
      <Repeat size={10} aria-hidden="true" />×
      {inside.map((loop) => loop.iterations.length).join(",")}
    </i>
  );
}

/** A spread or factor in two significant digits. */
function formatSpread(value: number): string {
  return value >= 100 || value < 0.01
    ? value.toExponential(1)
    : String(Number(value.toPrecision(2)));
}
