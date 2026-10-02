import {
  useContext,
  useLayoutEffect,
  useMemo,
  useState,
  type RefObject,
} from "react";
import type { Tensor } from "../api/client";
import { unravel } from "../tensors/coordinates";
import { roundedCellPath } from "../tensors/cellOutline";
import { NODE_HEIGHT, NODE_WIDTH, type JourneyGraph } from "./graph";
import type { CellMotionPlan } from "./cellMotion";
import { useSceneProgress, type SceneClock } from "./useSceneClock";
import { CellPaintContext } from "../tensors/InkShape";
import "./cellMotion.css";

type Point = { x: number; y: number };
type Faces = Point[][];
type Flight = {
  source: Faces;
  target: Faces;
  tensorId: string;
  index: number;
  label: string;
  /** Ink shade of the cell along the step's story axis, when traced. */
  tint: string | null;
};
type Props = {
  world: RefObject<HTMLDivElement | null>;
  graph: JourneyGraph;
  operationId: string;
  tensors: Record<string, Tensor>;
  plan: CellMotionPlan;
  clock: SceneClock;
  geometryKey: string;
  onInspect: (tensorId: string, index: number, keyboard?: boolean) => void;
};

/** Project the real rendered voxel faces back into the canvas world. */
function faces(
  cell: Element | undefined,
  bounds: DOMRect,
  scale: number,
): Faces | null {
  if (!cell) return null;
  const polygons = [
    ...cell.querySelectorAll<SVGPolygonElement>(".volume-cell-geometry"),
  ];
  if (!polygons.length) return null;
  const result: Faces = [];
  for (const polygon of polygons) {
    const transform = polygon.getScreenCTM();
    if (!transform) return null;
    const points: Point[] = [];
    for (let i = 0; i < polygon.points.numberOfItems; i++) {
      const point = polygon.points.getItem(i);
      const screen = new DOMPoint(point.x, point.y).matrixTransform(transform);
      points.push({
        x: (screen.x - bounds.x) / scale,
        y: (screen.y - bounds.y) / scale,
      });
    }
    result.push(points);
  }
  return result;
}

/** This component alone subscribes to frames; the full model graph stays still. */
export function CanvasCellMotion({
  world,
  graph,
  operationId,
  tensors,
  plan,
  clock,
  geometryKey,
  onInspect,
}: Props) {
  const progress = useSceneProgress(clock);
  const paint = useContext(CellPaintContext);
  const [flights, setFlights] = useState<Flight[]>([]);
  const [reduced, setReduced] = useState(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useLayoutEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  const sources = useMemo(
    () =>
      new Map(
        graph.edges
          .filter(
            (edge) => edge.target === operationId && edge.kind !== "storage",
          )
          .map((edge) => [edge.tensorId, edge.source]),
      ),
    [graph, operationId],
  );
  const targets = useMemo(
    () =>
      new Map(
        graph.nodes
          .filter(
            (node) =>
              node.id === operationId ||
              (graph.actorOrigins?.[node.id] === operationId &&
                graph.actorRoles?.[node.id]?.side === "output"),
          )
          .flatMap((node) =>
            node.tensors.map((tensor) => [tensor.id, node.id] as const),
          ),
      ),
    [graph, operationId],
  );
  useLayoutEffect(() => {
    const element = world.current;
    if (!element || !plan.movers.length) {
      setFlights([]);
      return;
    }
    const bounds = element.getBoundingClientRect();
    const scale = bounds.width / graph.width;
    if (!(scale > 0)) {
      setFlights([]);
      return;
    }
    const nodes = new Map(
      [
        ...element.querySelectorAll<HTMLElement>(".journey-node[data-node-id]"),
      ].map((node) => [node.dataset.nodeId, node]),
    );
    const cells = new Map<string, Map<number, Element>>();
    for (const [id, node] of nodes)
      cells.set(
        `${id}/${node.dataset.tensorId}`,
        new Map(
          [...node.querySelectorAll<SVGGElement>("[data-volume-index]")].map(
            (cell) => [Number(cell.dataset.volumeIndex), cell],
          ),
        ),
      );
    // A flying cell carries the paint of the cube it left.
    const tintOf = (from: Tensor, _to: Tensor, index: number) =>
      paint?.(from, index) ?? null;
    const next: Flight[] = [];
    for (const mover of plan.movers) {
      const source = faces(
        cells
          .get(`${sources.get(mover.sourceTensorId)}/${mover.sourceTensorId}`)
          ?.get(mover.sourceIndex),
        bounds,
        scale,
      );
      const target = faces(
        cells
          .get(`${targets.get(mover.targetTensorId)}/${mover.targetTensorId}`)
          ?.get(mover.targetIndex),
        bounds,
        scale,
      );
      if (
        !source ||
        !target ||
        source.length !== target.length ||
        source.some((face, i) => face.length !== target[i].length)
      )
        continue;
      const from = tensors[mover.sourceTensorId],
        to = tensors[mover.targetTensorId];
      next.push({
        source,
        target,
        tensorId: mover.targetTensorId,
        index: mover.targetIndex,
        label: `${from.name}[${unravel(mover.sourceIndex, from.shape).join(", ")}] → ${to.name}[${unravel(mover.targetIndex, to.shape).join(", ")}]`,
        tint: tintOf(from, to, mover.sourceIndex),
      });
    }
    setFlights(next);
  }, [
    plan,
    sources,
    targets,
    operationId,
    graph,
    geometryKey,
    tensors,
    world,
    paint,
  ]);

  const complete = progress >= 1;
  const phase = reduced ? 1 : progress;
  const results = graph.nodes.filter(
    (node) =>
      node.id === operationId ||
      (graph.actorOrigins?.[node.id] === operationId &&
        graph.actorRoles?.[node.id]?.side === "output"),
  );
  const resultNode = results.reduce<(typeof results)[number] | undefined>(
    (lowest, node) => (!lowest || node.y > lowest.y ? node : lowest),
    undefined,
  );
  const description = !plan.movers.length
    ? plan.caption
    : !flights.length
      ? "The matching cells are outside this view. Select the matching tensor output on each node, or open a tensor to inspect its cells."
      : `${plan.caption} ${flights.length} of ${plan.movers.length} coordinate paths are visible.`;
  return (
    <svg
      className="canvas-cell-motion"
      width={graph.width}
      height={graph.height}
      aria-label={
        flights.length
          ? `${flights.length} of ${plan.movers.length} recorded cell paths visible`
          : description
      }
      role="group"
      data-motion-progress={phase.toFixed(3)}
    >
      <title>{description}</title>
      {resultNode && (
        <text
          className="canvas-motion-summary"
          x={resultNode.x + NODE_WIDTH / 2}
          y={resultNode.y + NODE_HEIGHT + 17}
          textAnchor="middle"
        >
          <title>{description}</title>
          {!plan.movers.length
            ? "Recorded tensors · inspect cells"
            : !flights.length
              ? "Cell paths outside this view"
              : flights.length === plan.movers.length
                ? `${flights.length} cell paths`
                : `${flights.length} of ${plan.movers.length} paths · shown outputs`}
        </text>
      )}
      {flights.map((flight, order) => {
        const delay =
          flights.length > 1 ? (order / (flights.length - 1)) * 0.14 : 0;
        const p = Math.max(0, Math.min(1, (phase - delay) / 0.76));
        const eased = p * p * (3 - 2 * p);
        const lift = Math.sin(eased * Math.PI) * 20;
        const outlines = flight.source.map((face, fi) =>
          roundedCellPath(
            face.map((start, pi): [number, number] => {
              const end = flight.target[fi][pi];
              return [
                start.x + (end.x - start.x) * eased,
                start.y + (end.y - start.y) * eased - lift,
              ];
            }),
          ),
        );
        return (
          <g
            key={`${flight.tensorId}/${flight.index}/${order}`}
            className="canvas-cell-flight"
            role="button"
            aria-label={flight.label}
            tabIndex={order === 0 ? 0 : -1}
            data-settled={reduced || complete}
            onClick={(event) => {
              event.stopPropagation();
              onInspect(flight.tensorId, flight.index);
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                event.stopPropagation();
                onInspect(flight.tensorId, flight.index, true);
              } else if (
                ["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)
              ) {
                event.preventDefault();
                event.stopPropagation();
                const next =
                  event.key === "Home"
                    ? 0
                    : event.key === "End"
                      ? flights.length - 1
                      : Math.max(
                          0,
                          Math.min(
                            flights.length - 1,
                            order + (event.key === "ArrowLeft" ? -1 : 1),
                          ),
                        );
                event.currentTarget.parentElement
                  ?.querySelectorAll<SVGGElement>(".canvas-cell-flight")
                  [next]?.focus({ preventScroll: true });
              }
            }}
          >
            <title>{flight.label}</title>
            {outlines.map((outline, fi) => (
              <path
                key={fi}
                className={`canvas-cell-flight-face${flight.tint ? " inked" : ""}`}
                d={outline}
                fill={
                  flight.tint
                    ? undefined
                    : `hsl(${264 + (order % 4) * 4} 65% 80% / ${0.04 + fi * 0.012})`
                }
                // Inline, so hover and focus styles keep the ink.
                style={
                  flight.tint
                    ? { fill: flight.tint, fillOpacity: 0.42 + fi * 0.1 }
                    : undefined
                }
              />
            ))}
            <path
              className="canvas-cell-flight-rim"
              d={outlines.join(" ")}
              aria-hidden="true"
            />
          </g>
        );
      })}
    </svg>
  );
}
