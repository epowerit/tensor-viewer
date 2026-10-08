import {
  memo,
  useCallback,
  useContext,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
  type ReactNode,
} from "react";
import {
  FoldHorizontal,
  Maximize,
  Minus,
  Plus,
  Repeat,
  UnfoldHorizontal,
} from "lucide-react";
import type { JourneyNode } from "./graph";
import type { LoopGraph, LoopView } from "./loops";
import type { TensorLight } from "../tensors/TensorVolume";
import { ancestors, descendants, NODE_HEIGHT, NODE_WIDTH } from "./graph";
import {
  edgeWidths,
  flowNeighbor,
  flowSibling,
  lensColor,
  lensScale,
  lensText,
  lensValue,
  mainPath,
  routeEdge,
  spreadJumps,
  type FlowLens,
} from "./flow";
import { TensorGlyph } from "./TensorGlyph";
import { edgeDescription, journeyPorts } from "./ports";
import { CAPTION_ROOM, operationScene, sceneView } from "./scene";
import { TensorPeek } from "../editor/TensorPeek";
import { AxisWiring } from "./AxisWiring";
import { StageContents } from "./StageContents";
import { projectOperationScene } from "./sceneGraph";
import type { OperationSemantics } from "./sceneSemantics";
import { CanvasCellMotion } from "./CanvasCellMotion";
import { CanvasCellProbe, type CanvasProbe } from "./CanvasCellProbe";
import type { CellMotionPlan } from "./cellMotion";
import type { SceneClock } from "./useSceneClock";
import type { Operation, Tensor } from "../api/client";
import {
  fitCanvas,
  overviewView,
  reframeCanvas,
  showTensorCells,
  type CanvasViewport,
  type Viewport,
} from "./viewport";
import type { CanvasMemory, CarriedView } from "./reload";
import { AxisInkContext, InkShape, SymbolicContext } from "../tensors/InkShape";
import "./journeyCanvas.css";
import { kindName } from "../operations/kindName";
import { stageLabel, type JourneyStage } from "./stages";
import { frameStages } from "./stageFrames";
import { stepBytes, stepFlops } from "./cost";
import { broadcastReuse, reuseText, type Reuse } from "./broadcast";

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
  /** Colour every tensor by a value statistic. */
  lens?: FlowLens | null;
  /** For the gradient lens: the gradient's size at each tensor, by id. */
  gradients?: Map<string, number>;
  /** More about the lens, under its scale. */
  lensExtra?: ReactNode;
  /** For the live-memory lens: activation bytes alive at each step. */
  liveBytes?: Map<string, number>;
  /** Each step's time in µs, by operation id, for the time lens. */
  stepTimes?: Map<string, number>;
  /** For the change lens: how much each operation changed since a run. */
  changes?: Map<string, number>;
  /** A step previewed elsewhere, traced as if hovered. */
  previewStep?: string | null;
  /** Reports the steps under the pointer (a stage: all of its steps). */
  onHoverStep?: (ids: string[] | null) => void;
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
  /** Stages drawn open: each is framed, with a tab that folds it again. */
  openStages?: JourneyStage[];
  /**
   * The stage just folded or unfolded, and its steps: the canvas keeps that
   * spot where it was on screen instead of fitting the new layout.
   */
  foldAnchor?: { ids: string[]; key: number };
  /** Where this canvas is, kept for the project's next run. */
  memory?: { current: CanvasMemory | null };
  /** The last run's camera, kept on the same card in this run. */
  carry?: CarriedView | null;
  /** Steps an edit just added, edited, or reshaped. */
  changed?: ReadonlySet<string> | null;
  /** Where a recorded line is in the code now, edited since the run. */
  placeLine?: (file: string | null, line: number) => number | null;
  /**
   * A folded card's own steps, each with its result: the card shows the one
   * playback is on ("▸ transpose · 2 of 2") in place of its step range, and
   * its result in place of the card's own, flipping through them by `clock`
   * while the card plays as one step.
   */
  /** The run's tensors, for what a folded card takes in. */
  tensors?: Record<string, Tensor>;
  /** The run's operations, for what a folded card holds. */
  operations?: Operation[];
  playingInside?: {
    id: string;
    steps: { text: string; tensor?: Tensor }[];
    /** The step shown, or `clock` to flip through them over the beat. */
    at?: number;
    clock?: SceneClock;
  };
};
const clamp = (value: number) => Math.max(0.001, Math.min(2, value));

/**
 * A canvas shorter than this (the tensor shelf open on a laptop screen, say)
 * keeps its step caption to the title, so the model keeps the room; one
 * shorter still drops the caption's flow and wiring lines as well.
 */
const SHORT = 480;
const CRAMPED = 340;

export function JourneyCanvas({
  graph: modelGraph,
  selectedId,
  reachedThrough,
  kindleAbove = -1,
  topInset = 0,
  threaded,
  lens,
  gradients,
  lensExtra,
  liveBytes,
  stepTimes,
  changes,
  previewStep,
  onHoverStep,
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
  openStages,
  foldAnchor,
  memory,
  carry,
  changed,
  placeLine = (_, line) => line,
  playingInside,
  tensors,
  operations,
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
  // How far down the canvas the step's caption reaches: a long summary or
  // the lineage ribbon makes it taller than the room scenes keep for it.
  const [captionDepth, setCaptionDepth] = useState(0);
  const caption = useCallback((element: HTMLDivElement | null) => {
    if (!element) return;
    const measure = () => {
      const canvas = frame.current;
      if (canvas)
        setCaptionDepth(
          Math.round(
            element.getBoundingClientRect().bottom -
              canvas.getBoundingClientRect().top,
          ),
        );
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => {
      observer.disconnect();
      setCaptionDepth(0);
    };
  }, []);
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
  const symbolicOf = useContext(SymbolicContext);
  const [viewport, setViewport] = useState<CanvasViewport>({
    view: { x: 0, y: 0, scale: 1 },
    frame: null,
    overview: null,
  });
  // Panning and wheel or pinch zooming move the world on screen at once and
  // let React catch up when they pause: they send many small moves, and
  // drawing every card again for each one made the canvas stutter on large
  // models. Cards switch between cells and outlines once a zoom settles.
  const panned = useRef<Viewport | null>(null);
  const panTimer = useRef(0);
  const minimapView = useRef<SVGRectElement>(null);
  const view = viewport.view;
  /** The camera on screen: a pan React has not caught up with yet, or the view. */
  const shown = panned.current ?? view;
  const committed = useRef(view);
  committed.current = view;
  // Handlers set up once read the canvas size as it is now.
  const sizeNow = useRef(size);
  sizeNow.current = size;
  function showView(next: Viewport) {
    if (world.current) {
      world.current.style.transform = `translate(${next.x}px, ${next.y}px) scale(${next.scale})`;
      // Only the edges and fold tabs read the scale: set on the whole world,
      // a new scale restyled every cell of every tensor.
      for (const scaled of world.current.querySelectorAll<
        SVGElement | HTMLElement
      >(".journey-edges, .stage-frame-tab"))
        scaled.style.setProperty("--canvas-scale", `${next.scale}`);
    }
    if (frame.current) {
      const grid = 28 * Math.max(0.8, next.scale);
      frame.current.style.backgroundPosition = `${next.x}px ${next.y}px`;
      frame.current.style.backgroundSize = `${grid}px ${grid}px`;
    }
    const rect = minimapView.current;
    if (rect) {
      rect.setAttribute("x", `${-next.x / next.scale}`);
      rect.setAttribute("y", `${-next.y / next.scale}`);
      rect.setAttribute("width", `${sizeNow.current.width / next.scale}`);
      rect.setAttribute("height", `${sizeNow.current.height / next.scale}`);
    }
  }
  /** Hand a pending pan to React, ahead of any other change to the camera. */
  function settlePan() {
    window.clearTimeout(panTimer.current);
    const next = panned.current;
    if (!next) return;
    panned.current = null;
    setViewport((previous) => ({ ...previous, view: next }));
  }
  /** Move the camera on screen now; React catches up when moves pause. */
  function moveCamera(next: (base: Viewport) => Viewport) {
    panned.current = next(panned.current ?? committed.current);
    showView(panned.current);
    window.clearTimeout(panTimer.current);
    panTimer.current = window.setTimeout(settlePan, 120);
  }
  function panBy(dx: number, dy: number) {
    moveCamera((base) => ({ ...base, x: base.x + dx, y: base.y + dy }));
  }
  /** Zoom by a factor about a point of the canvas, as a camera move. */
  function zoomAt(factor: number, x: number, y: number) {
    moveCamera((base) => {
      const scale = clamp(base.scale * factor);
      const ratio = scale / base.scale;
      return {
        scale,
        x: x - (x - base.x) * ratio,
        y: y - (y - base.y) * ratio,
      };
    });
  }
  useEffect(() => () => window.clearTimeout(panTimer.current), []);
  // React writes a style only when it differs from what it wrote last, which
  // a camera move may have changed meanwhile: a view returning to an earlier
  // value, as Fit does after a pan, is written here.
  useLayoutEffect(() => {
    if (!panned.current) showView(view);
  });
  function setView(next: Viewport | ((previous: Viewport) => Viewport)) {
    settlePan();
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
  const [pointerNode, setHoverNode] = useState<string | null>(null);
  // A step previewed elsewhere (a Flow table row) traces the same way; a
  // step inside a collapsed stage lights its stage.
  const tracedPreview = previewStep
    ? byId.has(previewStep)
      ? previewStep
      : graph.nodes.find((node) =>
          node.stage?.operationIds.includes(previewStep),
        )?.id
    : undefined;
  const hoverNode = pointerNode ?? tracedPreview ?? null;
  useEffect(() => {
    if (!pointerNode) {
      onHoverStep?.(null);
      return;
    }
    const stage = byId.get(pointerNode)?.stage;
    onHoverStep?.(
      stage
        ? stage.operationIds
        : [graph.actorOrigins?.[pointerNode] ?? pointerNode],
    );
  }, [pointerNode, graph, byId, onHoverStep]);
  const hoverTimer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(hoverTimer.current), []);
  // A pointer that rests on a card a moment longer, for its contents.
  const [lingered, setLingered] = useState<string | null>(null);
  useEffect(() => {
    if (!pointerNode) {
      setLingered(null);
      return;
    }
    const timer = window.setTimeout(() => setLingered(pointerNode), 450);
    return () => window.clearTimeout(timer);
  }, [pointerNode]);
  // Zoomed out, where cards draw shapes only, the hovered tensor peeks out
  // beside its card: its axes, statistics, and a first look at its values.
  function peekCard() {
    if (dragging || playing) return null;
    // A folded card, lingered on at any zoom, lists what it holds.
    const card = lingered ? byId.get(lingered) : undefined;
    const contents =
      card?.stage && !card.stage.operationIds.includes(selectedId ?? "")
        ? card
        : undefined;
    // Zoomed out, other cards peek at their tensor; a folded card waits to
    // show its contents instead of flashing its result first.
    const hovered = pointerNode ? byId.get(pointerNode) : undefined;
    const node =
      contents ?? (view.scale < 0.45 && !hovered?.stage ? hovered : undefined);
    const tensor = node?.tensors[0];
    if (!node || (!contents && !tensor)) return null;
    // Read the canvas's place only when a peek shows: reading it makes the
    // browser lay out the whole canvas in the middle of a render.
    const box = frame.current?.getBoundingClientRect();
    if (!box) return null;
    const left = box.left + node.x * view.scale + view.x;
    const right = left + NODE_WIDTH * view.scale;
    const top = box.top + node.y * view.scale + view.y;
    return (
      <div
        className="tensor-peek-layer canvas-peek"
        style={{
          left: right + 296 > box.right ? left - 292 : right + 12,
          top: Math.max(box.top + 8, Math.min(top - 8, box.bottom - 280)),
        }}
      >
        {contents && operations && tensors ? (
          <StageContents
            stage={contents.stage!}
            operations={operations}
            tensors={tensors}
          />
        ) : (
          tensor && <TensorPeek tensor={tensor} />
        )}
      </div>
    );
  }
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
  // x + f(x): a skip into an add whose other operand was computed from the
  // same tensor. A long edge into any add (a rotary sum) is not one.
  const residuals = useMemo(() => {
    const found = new Set<string>();
    for (const edge of graph.edges) {
      if (byId.get(edge.target)?.operation?.kind !== "add") continue;
      const others = graph.edges.filter(
        (other) =>
          other.target === edge.target &&
          other.id !== edge.id &&
          other.kind !== "storage",
      );
      if (
        others.some((other) => ancestors(graph, other.source).has(edge.source))
      )
        found.add(edge.id);
    }
    return found;
  }, [graph, byId]);
  // The flow lens: each node's statistic and its place on the colour scale.
  const lensInfo = useMemo(() => {
    if (!lens) return null;
    const values = new Map<string, number>();
    // Cost lenses measure steps: a folded card sums the steps it holds.
    const byOperation = new Map(
      (operations ?? []).map((operation) => [operation.id, operation]),
    );
    const cost = (operation: Operation | undefined) =>
      !operation || !tensors
        ? undefined
        : lens === "compute"
          ? stepFlops(operation, tensors)
          : lens === "time"
            ? stepTimes?.get(operation.id)
            : stepBytes(operation, tensors);
    // The whole run's cost: every recorded step, each pass of a loop too.
    const total =
      lens === "compute" || lens === "memory" || lens === "time"
        ? (operations ?? []).reduce(
            (sum, operation) => sum + (cost(operation) ?? 0),
            0,
          )
        : 0;
    // Broadcast reuse names what was stretched, per node.
    const texts = new Map<string, string>();
    for (const node of graph.nodes) {
      if (lens === "live") {
        const ids = node.stage
          ? (node.repOperationIds ?? node.stage.operationIds)
          : node.operation
            ? [node.operation.id]
            : [];
        const sizes = ids.flatMap((id) => liveBytes?.get(id) ?? []);
        if (sizes.length) values.set(node.id, Math.max(...sizes));
        continue;
      }
      if (lens === "broadcast") {
        const ids = node.stage
          ? (node.repOperationIds ?? node.stage.operationIds)
          : node.operation
            ? [node.operation.id]
            : [];
        let best: Reuse | null = null;
        for (const id of ids) {
          const operation = byOperation.get(id);
          const found =
            operation && tensors ? broadcastReuse(operation, tensors) : null;
          if (found && (!best || found.factor > best.factor)) best = found;
        }
        if (best) {
          values.set(node.id, best.factor);
          texts.set(node.id, reuseText(best));
        }
        continue;
      }
      if (lens === "compute" || lens === "memory" || lens === "time") {
        const ids = node.stage
          ? (node.repOperationIds ?? node.stage.operationIds)
          : node.operation
            ? [node.operation.id]
            : [];
        if (!ids.length) continue;
        const value = ids.reduce(
          (sum, id) => sum + (cost(byOperation.get(id)) ?? 0),
          0,
        );
        values.set(node.id, value);
        continue;
      }
      const value =
        lens === "gradient"
          ? gradients?.get(node.tensors[0]?.id ?? "")
          : lens === "change"
            ? node.stage
              ? Math.max(
                  ...node.stage.operationIds.map((id) => changes?.get(id) ?? 0),
                )
              : changes?.get(node.operation?.id ?? node.id)
            : lensValue(node.tensors[0], lens);
      if (value !== undefined) values.set(node.id, value);
    }
    const all = [...values.values()];
    const scale = lensScale(all, lens);
    return {
      values,
      scale,
      low: all.length ? Math.min(...all) : 0,
      high: all.length ? Math.max(...all) : 0,
      total,
      texts,
    };
  }, [
    graph,
    lens,
    changes,
    gradients,
    liveBytes,
    operations,
    tensors,
    stepTimes,
  ]);
  const lensOf = (id: string) => {
    const value = lensInfo?.values.get(id);
    return value === undefined || !lens
      ? null
      : {
          color: lensColor(lensInfo!.scale(value)),
          text: lensInfo!.texts.get(id) ?? lensText(value, lens),
        };
  };
  // Thicker connections carry more data.
  const widths = useMemo(() => edgeWidths(graph), [graph]);
  // Skip connections arc over the nodes they pass; computed once per layout.
  const routes = useMemo(() => {
    const boxes = visibleNodes.map(({ id, x, y }) => ({ id, x, y }));
    return new Map(
      visibleEdges.map((edge) => {
        const from = byId.get(edge.source)!;
        const to = byId.get(edge.target)!;
        const anchors = ports.get(edge.id)!;
        return [
          edge.id,
          routeEdge(
            from,
            to,
            from.x + NODE_WIDTH,
            from.y + anchors.sourceY,
            to.y + anchors.targetY,
            boxes,
          ),
        ];
      }),
    );
    // visibleThrough decides which nodes and edges exist.
  }, [graph, ports, byId, visibleThrough]);
  // The hovered node's main line of flow, as names and shapes.
  // Hovering previews a node's path; otherwise it follows the selected step
  // and works as a breadcrumb.
  const pinned = !hoverNode;
  const ribbon = useMemo(
    () =>
      hoverNode
        ? mainPath(graph, hoverNode)
        : selectedId && byId.has(selectedId)
          ? mainPath(graph, selectedId)
          : [],
    [graph, hoverNode, selectedId, byId],
  );
  const jumps = useMemo(() => spreadJumps(ribbon), [ribbon]);
  const loopFrames = frameLoops(graph, visibleNodes);
  // Which of a folded card's steps it shows: fixed, or flipping with the
  // clock over the card's one beat, from its first result to its last.
  const insideAt = useSyncExternalStore(
    playingInside?.clock?.subscribe ?? noSubscription,
    () =>
      !playingInside
        ? -1
        : playingInside.clock
          ? Math.min(
              playingInside.steps.length - 1,
              Math.floor(
                playingInside.clock.getSnapshot() * playingInside.steps.length,
              ),
            )
          : (playingInside.at ?? -1),
  );
  const insideStep =
    playingInside && insideAt >= 0 ? playingInside.steps[insideAt] : undefined;
  // Open calls and capsules are framed where they are drawn, to fold again.
  const stageFrames = useMemo(
    () =>
      openStages
        ? frameStages(
            openStages,
            visibleNodes,
            graph.sceneOperationId && graph.sceneNodeIds
              ? {
                  operationId: graph.sceneOperationId,
                  nodeIds: graph.sceneNodeIds,
                }
              : null,
          )
        : [],
    [openStages, visibleNodes, graph],
  );
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
  const sceneHeadroom =
    loopHeadroom + Math.max(0, captionDepth + 16 - CAPTION_ROOM);
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

  function fittedView(whole = false): Viewport {
    // Room kept clear at the top, such as for a failed run's error card.
    return overviewView(modelGraph, size, topInset, whole);
  }
  function fit() {
    if (!size.width || !size.height) return;
    manualNavigation();
    settlePan();
    setViewport((previous) =>
      fitCanvas(previous, structure, size, fittedView(true)),
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
  /** The last run's camera, moved so its card sits where it did. */
  function carriedView(): Viewport | undefined {
    const to = carry && byId.get(carry.to);
    if (!carry || !to) return undefined;
    const { scale } = carry.view;
    return {
      scale,
      x: carry.view.x + (carry.from.x - to.x) * scale,
      y: carry.view.y + (carry.from.y - to.y) * scale,
    };
  }
  function focus(node: JourneyNode) {
    manualNavigation();
    setView((previous) => focusedView(node, previous));
  }
  // A layout effect, so a new step's camera is set before the browser paints:
  // the step is never drawn from the old camera first, and the browser lays
  // the canvas out once rather than twice.
  useLayoutEffect(() => {
    const selected = graph.nodes.find((node) => node.id === selectedId);
    settlePan();
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
                  ? sceneView(graph, selected.id, size, current, sceneHeadroom)
                  : current
                : focusedView(selected, current)
          : focusedLoop
            ? (current) =>
                followPlayback || previous.frame?.selectionKey !== selectionKey
                  ? loopView(focusedLoop)
                  : current
            : undefined,
        carriedView(),
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

  // Folding or unfolding a stage keeps it in place: its card, or its first
  // step, lands where it was on screen, at the same scale. Positions from the
  // layout before are kept below, after this has read them.
  const placed = useRef<{
    nodes: Map<string, { x: number; y: number }>;
    view: Viewport;
  } | null>(null);
  const anchoredKey = useRef(0);
  useEffect(() => {
    const before = placed.current;
    if (!foldAnchor || foldAnchor.key === anchoredKey.current || !before)
      return;
    anchoredKey.current = foldAnchor.key;
    const leftmost = (positions: { x: number; y: number }[]) =>
      positions.sort((a, b) => a.x - b.x || a.y - b.y)[0];
    const from = leftmost(
      foldAnchor.ids.flatMap((id) => before.nodes.get(id) ?? []),
    );
    const to = leftmost(
      graph.nodes.filter((node) => foldAnchor.ids.includes(node.id)),
    );
    if (!from || !to) return;
    const scale = before.view.scale;
    setView({
      scale,
      x: before.view.x + from.x * scale - to.x * scale,
      y: before.view.y + from.y * scale - to.y * scale,
    });
  }, [structure, layoutStructure, foldAnchor?.key]);
  useEffect(() => {
    placed.current = {
      nodes: new Map(graph.nodes.map((node) => [node.id, node])),
      view,
    };
    if (memory) memory.current = { ...placed.current, size };
  });

  useEffect(() => {
    if (followPlayback && focusedLoop) setView(loopView(focusedLoop));
    else if (followPlayback && sceneMode && focusKey > 0 && selectedId)
      setView((previous) =>
        sceneView(graph, selectedId, size, previous, sceneHeadroom),
      );
    // Follow also reframes on resize; manual navigation keeps its own framing.
  }, [followPlayback, size.width, size.height, sceneHeadroom]);

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
        zoomAt(
          Math.exp(-event.deltaY * 0.008),
          event.clientX - bounds.left,
          event.clientY - bounds.top,
        );
      else panBy(-event.deltaX, -event.deltaY);
    }
    target.addEventListener("wheel", wheel, { passive: false });
    return () => target.removeEventListener("wheel", wheel);
  }, []);

  // The step caption owns the top centre; the ribbon then flows under it.
  // A step's scene, or a folded card: a capsule explains its axes, a call
  // what goes in and out, or the step playback is on inside it.
  const captionable =
    !!graph.sceneOperationId || (!!selectedId && !!byId.get(selectedId)?.stage);
  const captioned = !probe && !!(sceneMode && semantics && captionable);
  const ribbonView = (
    <ol
      className={`flow-ribbon ${pinned ? "flow-ribbon-pinned" : ""} ${captioned ? "flow-ribbon-in-caption" : ""}`}
      aria-label={
        pinned
          ? "Main path to the selected step"
          : "Main path to the hovered node"
      }
    >
      {(ribbon.length > 9
        ? [...ribbon.slice(0, 3), null, ...ribbon.slice(-5)]
        : ribbon
      ).map((node, i) =>
        node ? (
          <li key={node.id}>
            {i > 0 && <span className="flow-ribbon-arrow">→</span>}
            <span
              {...(pinned &&
              modelGraph.nodes.some((item) => item.id === node.id)
                ? {
                    role: "button",
                    tabIndex: 0,
                    title: "Go to this step",
                    onClick: () => onSelect(node.id),
                    onKeyDown: (event: React.KeyboardEvent) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        onSelect(node.id);
                      }
                    },
                  }
                : {})}
              className={node.id === selectedId ? "flow-ribbon-current" : ""}
            >
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
  );
  const peek = peekCard();
  return (
    <div
      ref={frame}
      className={`journey-canvas ${dragging ? "is-panning" : ""} ${scene.nodes.size ? "has-scene" : ""} ${playing ? "is-playing" : ""} ${sceneMode && followPlayback ? "is-following" : ""} ${probe?.result.status === "mapped" && !probe.result.truncated ? "has-cell-probe" : ""}${size.height && size.height < SHORT ? " is-short" : ""}${size.height && size.height < CRAMPED ? " is-cramped" : ""}`}
      role="region"
      aria-label="Tensor transformation canvas"
      aria-keyshortcuts="Space Enter Escape Home + - [ ]"
      tabIndex={0}
      style={{
        backgroundPosition: `${shown.x}px ${shown.y}px`,
        backgroundSize: `${28 * Math.max(0.8, shown.scale)}px ${28 * Math.max(0.8, shown.scale)}px`,
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
          event.key.startsWith("Arrow") &&
          selectedId &&
          !(event.target as Element).closest("input, select, textarea")
        ) {
          // Alt+↑/↓ move between the steps that read the same tensor.
          const next =
            event.key === "ArrowRight" || event.key === "ArrowLeft"
              ? flowNeighbor(
                  modelGraph,
                  selectedId,
                  event.key === "ArrowRight" ? "forward" : "back",
                )
              : flowSibling(
                  modelGraph,
                  selectedId,
                  event.key === "ArrowDown" ? "next" : "previous",
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
          panBy(event.clientX - previous.x, event.clientY - previous.y);
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
          moveCamera((p) => {
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
          transform: `translate(${shown.x}px, ${shown.y}px) scale(${shown.scale})`,
        }}
      >
        <svg
          className="journey-edges"
          width={graph.width}
          height={graph.height}
          overflow="visible"
          aria-hidden="true"
          // Lines and their labels stay legible at any zoom.
          style={{ ["--canvas-scale" as string]: shown.scale }}
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
          {stageFrames.map((frame) => (
            <rect
              key={`frame-${frame.stage.id}`}
              className={`stage-frame${frame.stage.layout ? " is-layout" : ""}`}
              x={frame.left}
              y={frame.top}
              width={frame.right - frame.left}
              height={frame.bottom - frame.top}
              rx={frame.stage.layout ? 14 : 22}
            />
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
            const route = routes.get(edge.id)!;
            const focused =
              sceneMode && scene.nodes.size
                ? scene.edges.has(edge.id)
                : directEdges.has(edge.id) ||
                  (highlighted.has(edge.source) &&
                    highlighted.has(edge.target));
            const carried = from.tensors.find(
              (tensor) => tensor.id === edge.tensorId,
            );
            const path = route.path;
            // A smaller tensor stretched to fit an elementwise result is
            // reused, not consumed: a bias, a positional table, a mask.
            const output = to.tensors[0];
            const reuse =
              carried &&
              output &&
              to.operation &&
              ELEMENTWISE_KINDS.has(to.operation.kind) &&
              carried.numel > 0 &&
              carried.numel < output.numel
                ? Math.round(output.numel / carried.numel)
                : 0;
            const residual = route.skip && !reuse && residuals.has(edge.id);
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
                className={`${focused ? "edge-highlighted" : ""} ${scene.edges.has(edge.id) ? "scene-edge" : ""} ${edge.kind === "storage" ? "edge-storage" : ""} ${probeEdge ? "probe-connection" : ""} ${flowOf(edge)} ${lineageOf(edge)} ${route.skip ? "edge-skip" : ""} ${reuse ? "edge-broadcast" : ""} ${lensOf(edge.source) ? "edge-lensed" : ""}`}
              >
                <title>{edgeDescriptions.get(edge.id)}</title>
                <path
                  className="journey-edge-hit"
                  d={path}
                  onClick={(event) => {
                    // A connection leads to the step that reads its tensor.
                    if (
                      modelGraph.nodes.some((item) => item.id === edge.target)
                    ) {
                      event.stopPropagation();
                      onSelect(edge.target);
                    }
                  }}
                />
                <path
                  className="journey-edge"
                  markerEnd={`url(#${marker})`}
                  d={path}
                  style={
                    {
                      "--edge-width": widths.get(edge.id) ?? 1.4,
                      ...(lensOf(edge.source)
                        ? { "--lens-edge": lensOf(edge.source)!.color }
                        : {}),
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
                      x={route.mid.x}
                      y={route.mid.y - 7}
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
                {view.scale >= 0.3 && !lineage && (residual || reuse > 1) && (
                  <text
                    className="edge-note"
                    x={route.mid.x}
                    y={route.mid.y - 7}
                    textAnchor="middle"
                  >
                    {residual ? "residual" : `reused ×${reuse}`}
                  </text>
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
        {placeFoldTabs(stageFrames, view.scale).map(({ stage, x, y }) => (
          <button
            key={`fold-${stage.id}`}
            type="button"
            className={`stage-frame-tab${stage.layout ? " is-layout" : ""}`}
            style={{
              left: x,
              top: y,
              ["--canvas-scale" as string]: shown.scale,
            }}
            aria-label={`Fold ${stageLabel(stage)}`}
            title={
              stage.layout
                ? `${stage.title}: [${stage.layout.from.join(", ")}] → [${stage.layout.to.join(", ")}], values unchanged. Fold these steps back into one card.`
                : `${stage.path}: fold this call back into one card`
            }
            onClick={(event) => {
              event.stopPropagation();
              onStageToggle(stage.id);
            }}
          >
            <FoldHorizontal size={11} aria-hidden="true" />
            <b>{stage.layout ? stage.title : stage.path || stage.title}</b>
            <small>
              {stage.operationIds.length}{" "}
              {stage.operationIds.length === 1 ? "step" : "steps"}
            </small>
          </button>
        ))}
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
          // A folded card that playback is inside shows each step's result
          // as it is made, so the tensor changes shape without unfolding.
          const tensor =
            (group && playingInside?.id === node.id && insideStep?.tensor) ||
            node.tensors[tensorChoice];
          const takesIn = group?.inputs?.[0]
            ? tensors?.[group.inputs[0]]
            : undefined;
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
          // A step the last save added, edited, or reshaped, or a card holding one.
          const edited =
            !!changed &&
            (changed.has(node.id) ||
              (group?.operationIds ?? []).some((id) => changed.has(id)));
          const rootLabel =
            tensor?.role === "input"
              ? "Input tensor"
              : tensor?.role === "parameter"
                ? "Parameter tensor"
                : "Captured tensor";
          return (
            <article
              key={node.id}
              className={`journey-node node-light-${light} ${threaded?.has(ownerId) ? "node-threaded" : ""} ${multiple ? "has-tensor-choices" : ""} ${group ? "journey-stage-node" : ""} ${selectedId === ownerId ? "node-selected" : ""} ${scene.nodes.has(node.id) ? "scene-actor" : ""} ${highlighted.has(ownerId) ? "node-connected" : ""} ${notContributing ? "node-not-contributing" : ""} ${operation?.status === "error" || group?.failed ? "node-error" : ""} ${lineage ? (lineage.up.has(node.id) || lineage.down.has(node.id) ? "node-in-lineage" : "node-off-lineage") : ""} category-node-${operation?.lesson.category ?? (group ? "layout" : "input")}${edited ? " node-changed" : ""}`}
              style={
                {
                  left: node.x,
                  top: node.y,
                  width: NODE_WIDTH,
                  height: NODE_HEIGHT,
                } as React.CSSProperties
              }
              onClick={inspectNode}
              onDoubleClick={(event) => {
                // Double-click folds and unfolds where you look: a folded
                // card opens, a step folds the innermost open call around it.
                if ((event.target as Element).closest("button")) return;
                const call = group
                  ? group
                  : openStages
                      ?.filter((stage) => stage.operationIds.includes(ownerId))
                      .sort(
                        (a, b) => a.operationIds.length - b.operationIds.length,
                      )[0];
                if (call) onStageToggle(call.id);
              }}
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
                  ? group.layout
                    ? `${group.title}: ${group.operationIds.length} steps that rearrange a tensor from ${group.layout.from.join(" × ")} to ${group.layout.to.join(" × ")}, values unchanged`
                    : `Stage: ${group.title}, ${group.operationIds.length} operations, ${group.path}`
                  : operation
                    ? `Step ${operation.index + 1}: ${kindName(operation.kind)}, ${tensor?.name ?? (operation.status === "error" ? "execution error" : "no tensor output")}, shape ${tensor ? tensor.shape.join(", ") || "scalar" : "none"}`
                    : `${rootLabel} ${tensor?.name}, shape ${tensor?.shape.join(", ") || "scalar"}`
              }
              data-node-id={node.id}
              data-owner-id={ownerId}
              data-tensor-id={tensor?.id}
              title={
                group
                  ? `${group.path || group.title} · ${group.operationIds.length} steps played as one. Select to focus it; double-click to unfold.`
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
                <span
                  title={
                    actorLabels.join(" · ") ||
                    (group && takesIn
                      ? `Takes ${takesIn.name} [${takesIn.shape.join(" × ")}]; its result is below`
                      : undefined)
                  }
                >
                  {actorLabel ??
                    (group
                      ? // A folded card reads as a function: what it takes
                        // in here, its result below.
                        takesIn
                        ? `in [${takesIn.shape.join(",")}]`
                        : `${group.operationIds.length} OPS`
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
                    <InkShape
                      shape={tensor.shape}
                      ink={inkFor?.(tensor)}
                      labels={symbolicOf?.(tensor.id)}
                    />
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
                  {group && playingInside?.id === node.id && insideStep
                    ? `${insideStep.text} · ${insideAt + 1} of ${playingInside.steps.length}`
                    : group?.layout
                      ? "values unchanged"
                      : group
                        ? // Played as one step: say so, not its recorded range.
                          `${group.operationIds.length} steps as one`
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
                          ? `L${placeLine(operation.source.file ?? null, operation.source.line) ?? "–"}`
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
                    title={
                      group.layout
                        ? "See the steps that rearrange it"
                        : "See inside this stage"
                    }
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
              // Too far out for every name: keep where playback is, and the
              // lens's hottest steps.
              const hot =
                !!lensInfo &&
                lensInfo.scale(lensInfo.values.get(node.id) ?? 0) >= 0.85;
              if (view.scale < 0.2 && !current && !hot) return null;
              const tensor = node.tensors[0];
              // Symbolic sizes where found (B, T), recorded ones otherwise.
              const symbols = tensor ? symbolicOf?.(tensor.id) : null;
              const sizes = tensor?.shape.map(
                (size, axis) => symbols?.[axis] ?? `${size}`,
              );
              const name =
                node.stage?.title ??
                tensor?.name ??
                (node.operation ? kindName(node.operation.kind) : "");
              const room = current ? 220 : (NODE_WIDTH + 70) * view.scale;
              // Far out, labels get narrow: shapes drop their spaces and the
              // type steps down, so names and sizes still read in full. A
              // long shape does so sooner (10px mono is about 6px a glyph,
              // and the label's padding and border take 18).
              const tight =
                room < 96 ||
                (!!tensor && `[${sizes!.join(", ")}]`.length * 6.1 + 18 > room);
              return (
                <div
                  key={node.id}
                  className={`overview-label overview-${lightOf(node)} ${current ? "overview-current" : ""}${tight ? " overview-tight" : ""}`}
                  style={{
                    left: node.x + NODE_WIDTH / 2,
                    top: node.y + NODE_HEIGHT / 2,
                    maxWidth: room,
                  }}
                >
                  <b>{name}</b>
                  {tensor && <small>[{sizes!.join(tight ? "," : ", ")}]</small>}
                  {lensOf(node.id) && (
                    <small
                      className="overview-lens"
                      style={{ color: lensOf(node.id)!.color }}
                    >
                      {lensOf(node.id)!.text}
                    </small>
                  )}
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
      {/* What reads over the canvas's top, below its toolbar, stacked so
          none covers another: the lens's scale and extras on the right, then
          the path to the step, its caption, or a traced cell. */}
      <div className="canvas-top">
        {lens && ((lensInfo && lensInfo.values.size > 0) || lensExtra) && (
          <div className="lens-hud">
            {lens && lensInfo && lensInfo.values.size > 0 && (
              <div className="lens-legend" aria-label="Flow lens scale">
                <span>{lensText(lensInfo.low, lens)}</span>
                <i aria-hidden="true" />
                <span>{lensText(lensInfo.high, lens)}</span>
                {(lens === "compute" ||
                  lens === "memory" ||
                  lens === "time") && (
                  <b title="Over every recorded step, each pass of a loop included">
                    Σ {lensText(lensInfo.total, lens)}
                  </b>
                )}
              </div>
            )}
            {lens && lensExtra && <div className="lens-extra">{lensExtra}</div>}
          </div>
        )}
        {ribbon.length > 1 && !captioned && ribbonView}
        {probe && onClearTrace ? (
          <CanvasCellProbe
            probe={probe}
            onInspect={() => onTensorInspect(probe.tensor.id, probe.index)}
            onClear={clearTrace}
          />
        ) : (
          sceneMode &&
          semantics &&
          captionable && (
            <div className="canvas-scene-caption" ref={caption}>
              <div className="canvas-scene-caption-text" aria-live="polite">
                <strong>{semantics.title}</strong>
                <span title={semantics.summary}>{semantics.summary}</span>
              </div>
              {semantics.wiring && <AxisWiring data={semantics.wiring} />}
              {semantics.einops && <EinopsLine call={semantics.einops} />}
              {ribbon.length > 1 && ribbonView}
            </div>
          )
        )}
      </div>
      {peek}
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
            {/* Open calls outlined, as the canvas frames them. */}
            {stageFrames.map((frame) => (
              <rect
                key={`minimap-frame-${frame.stage.id}`}
                className={`minimap-frame${frame.stage.layout ? " is-layout" : ""}`}
                x={frame.left}
                y={frame.top}
                width={frame.right - frame.left}
                height={frame.bottom - frame.top}
                rx="20"
              />
            ))}
            {visibleNodes.map((node) => (
              <rect
                style={
                  lensOf(node.id) ? { fill: lensOf(node.id)!.color } : undefined
                }
                className={`minimap-${lightOf(node)} ${node.stage ? `minimap-folded${node.stage.layout ? " is-layout" : ""}` : ""} ${changed && (changed.has(node.id) || node.stage?.operationIds.some((id) => changed.has(id))) ? "minimap-changed" : ""} ${threaded?.has(graph.actorOrigins?.[node.id] ?? node.id) ? "minimap-threaded" : ""} ${selectedId === node.id ? "minimap-selected" : ""}`}
                key={node.id}
                x={node.x}
                y={node.y}
                width={NODE_WIDTH}
                height={NODE_HEIGHT}
                rx="12"
              />
            ))}
            <rect
              ref={minimapView}
              className="minimap-viewport"
              x={-shown.x / shown.scale}
              y={-shown.y / shown.scale}
              width={size.width / shown.scale}
              height={size.height / shown.scale}
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

/** Elementwise steps, where a smaller operand broadcasts to the result. */
const ELEMENTWISE_KINDS = new Set([
  "add",
  "sub",
  "mul",
  "div",
  "true_divide",
  "pow",
  "maximum",
  "minimum",
  "where",
  "masked_fill",
  "__rsub__",
  "__rtruediv__",
]);

/**
 * Where each frame's fold tab sits: on the frame's lower edge, near its left.
 * Nested frames' edges nearly meet, so a tab that would cover one already
 * placed moves to its right, and the tabs read along the edge like a path
 * (GPT, blocks.0, attention). Sizes are in world units at the tab's own
 * counter-scale, which keeps it a fixed size on screen when zoomed out.
 */
function placeFoldTabs(
  frames: { stage: JourneyStage; left: number; bottom: number }[],
  scale: number,
) {
  const grow = Math.max(1, 1 / scale);
  const placed: { x0: number; x1: number; y0: number; y1: number }[] = [];
  return [...frames]
    .sort((a, b) => b.bottom - a.bottom || a.left - b.left)
    .map((frame) => {
      const text = frame.stage.layout
        ? frame.stage.title
        : frame.stage.path || frame.stage.title;
      // The tab sits across the edge. Zoomed out, it grows about its left
      // middle, so it moves down to keep clear of the cards the frame holds:
      // its top nears the edge as it grows, rather than reaching up.
      const width = (text.length * 7 + 72) * grow;
      const top = frame.bottom - 11 / grow;
      const y = top + 11 * grow - 11;
      const y0 = top,
        y1 = top + 22 * grow;
      let x = frame.left + 16;
      for (;;) {
        const hit = placed.find(
          (box) =>
            x < box.x1 && x + width > box.x0 && y0 < box.y1 && y1 > box.y0,
        );
        if (!hit) break;
        x = hit.x1 + 6 * grow;
      }
      placed.push({ x0: x, x1: x + width, y0, y1 });
      return { stage: frame.stage, x, y };
    });
}

const noSubscription = () => () => {};

/** A rearrangement as one einops call, to read or copy into code. */
function EinopsLine({ call }: { call: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="einops-line">
      <span>as einops</span>
      <code title={call}>{call}</code>
      <button
        type="button"
        className="text-button"
        onClick={() => {
          void navigator.clipboard?.writeText(call).then(() => {
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1500);
          });
        }}
      >
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}
