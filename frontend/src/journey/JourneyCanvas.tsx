import { useEffect, useId, useRef, useState } from "react";
import { Focus, Maximize, Minus, Plus } from "lucide-react";
import type { JourneyGraph, JourneyNode } from "./graph";
import { NODE_HEIGHT, NODE_WIDTH } from "./graph";
import { TensorGlyph } from "./TensorGlyph";

type Viewport = { x: number; y: number; scale: number };
type Props = {
  graph: JourneyGraph;
  selectedId: string | null;
  highlighted: Set<string>;
  focusKey: number;
  visibleThrough?: number;
  onSelect: (id: string) => void;
  onOverview: () => void;
  onTensorInspect: (tensorId: string, index: number) => void;
};
const clamp = (value: number) => Math.max(0.001, Math.min(2, value));

export function JourneyCanvas({
  graph,
  selectedId,
  highlighted,
  focusKey,
  visibleThrough,
  onSelect,
  onOverview,
  onTensorInspect,
}: Props) {
  const frame = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const [view, setView] = useState<Viewport>({ x: 0, y: 0, scale: 1 });
  const [dragging, setDragging] = useState(false);
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const marker = useId().replace(/:/g, "");
  const byId = new Map(graph.nodes.map((node) => [node.id, node]));
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

  function fit() {
    const scale = Math.min(
      1,
      Math.max(
        0.001,
        Math.min(
          (size.width - 72) / graph.width,
          (size.height - 170) / graph.height,
        ),
      ),
    );
    setView({
      scale,
      x: (size.width - graph.width * scale) / 2,
      y: (size.height - graph.height * scale) / 2,
    });
  }
  function focus(node: JourneyNode) {
    setView((previous) => {
      const scale = Math.max(0.8, Math.min(1.15, previous.scale));
      return {
        scale,
        x: size.width / 2 - (node.x + NODE_WIDTH / 2) * scale,
        y: size.height / 2 - (node.y + NODE_HEIGHT / 2) * scale,
      };
    });
  }
  useEffect(() => {
    if (!size.width || !size.height) return;
    const selected = graph.nodes.find((node) => node.id === selectedId);
    if (selected && focusKey > 0) focus(selected);
    else fit();
    // Reframe on selection or available space changes, not while panning.
  }, [graph, focusKey, size.width, size.height]);

  function zoom(factor: number, x = size.width / 2, y = size.height / 2) {
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
      className={`journey-canvas ${dragging ? "is-panning" : ""}`}
      role="region"
      aria-label="Tensor transformation canvas"
      tabIndex={0}
      style={{
        backgroundPosition: `${view.x}px ${view.y}px`,
        backgroundSize: `${28 * Math.max(0.8, view.scale)}px ${28 * Math.max(0.8, view.scale)}px`,
      }}
      onKeyDown={(event) => {
        if (event.target !== event.currentTarget) return;
        const delta = {
          ArrowLeft: [70, 0],
          ArrowRight: [-70, 0],
          ArrowUp: [0, 70],
          ArrowDown: [0, -70],
        }[event.key];
        if (delta) {
          event.preventDefault();
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
            "button,select,input,[role=button],.journey-node",
          ) ||
          (event.button !== 0 && event.button !== 1)
        )
          return;
        event.preventDefault();
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
            const x = from.x + NODE_WIDTH;
            const y = from.y + NODE_HEIGHT / 2;
            const endY = to.y + NODE_HEIGHT / 2;
            const bend = Math.max(38, (to.x - x) * 0.5);
            const focused =
              highlighted.has(edge.source) && highlighted.has(edge.target);
            // Keep parallel operand connections distinguishable (e.g. x @ x).
            const offset = edge.inputIndex * 5;
            return (
              <g key={edge.id} className={focused ? "edge-highlighted" : ""}>
                <path
                  className="journey-edge"
                  markerEnd={`url(#${marker})`}
                  d={`M${x},${y + offset} C${x + bend},${y + offset} ${to.x - bend},${endY + offset} ${to.x - 3},${endY + offset}`}
                />
                {edge.inputIndex === 0 && (
                  <text
                    className="edge-label"
                    x={to.x - 48}
                    y={endY - 13}
                    textAnchor="middle"
                  >
                    {to.operation?.kind}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
        {visibleNodes.map((node) => {
          const operation = node.operation;
          const tensor = node.tensors[0];
          const rootLabel =
            tensor?.role === "input" ? "Input tensor" : "Captured tensor";
          return (
            <article
              key={node.id}
              className={`journey-node ${selectedId === node.id ? "node-selected" : ""} ${highlighted.has(node.id) ? "node-connected" : ""} ${operation?.status === "error" ? "node-error" : ""} category-node-${operation?.lesson.category ?? "input"}`}
              style={{
                left: node.x,
                top: node.y,
                width: NODE_WIDTH,
                height: NODE_HEIGHT,
              }}
              onClick={() => onSelect(node.id)}
              onFocus={(event) => {
                // Keyboard navigation must bring off-screen tensors into view.
                // Pointer focus waits for the click so the target doesn't move
                // between pressing and releasing the mouse.
                if ((event.target as Element).matches(":focus-visible")) {
                  frame.current?.scrollTo(0, 0);
                  focus(node);
                }
              }}
              aria-label={
                operation
                  ? `Step ${operation.index + 1}: ${operation.kind}, ${tensor?.name ?? "execution error"}, shape ${tensor ? tensor.shape.join(", ") || "scalar" : "none"}`
                  : `${rootLabel} ${tensor?.name}, shape ${tensor?.shape.join(", ") || "scalar"}`
              }
              data-node-id={node.id}
              title={
                operation?.lesson.summary ??
                (tensor?.role === "input"
                  ? "The original input tensor"
                  : "A tensor captured before its first recorded use")
              }
            >
              <div className="journey-node-top">
                <button
                  className="journey-node-select"
                  aria-label={
                    operation
                      ? `Step ${operation.index + 1}: ${operation.kind}, ${tensor?.name ?? "error"}, shape ${tensor?.shape.join(", ")}`
                      : `${rootLabel} ${tensor?.name}, shape ${tensor?.shape.join(", ")}`
                  }
                  aria-pressed={selectedId === node.id}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelect(node.id);
                  }}
                >
                  <b>{tensor?.name ?? operation?.kind}</b>
                </button>
                <span>
                  {node.terminal
                    ? "OUTPUT"
                    : !operation
                      ? tensor?.role === "input"
                        ? "INPUT"
                        : "CAPTURED"
                      : `${operation.index + 1}`.padStart(2, "0")}
                </span>
              </div>
              {tensor ? (
                <TensorGlyph
                  tensor={tensor}
                  onSelect={(index) => onTensorInspect(tensor.id, index)}
                />
              ) : (
                <div className="node-failure">Execution stopped</div>
              )}
              <div className="node-shape">
                {tensor
                  ? tensor.shape.length
                    ? `[${tensor.shape.join(", ")}]`
                    : "scalar · shape []"
                  : "No output"}
              </div>
              <div className="node-operation">
                <span className="operation-dot" />
                <span>{operation?.kind ?? rootLabel}</span>
                <small>
                  {node.tensors.length > 1
                    ? `${node.tensors.length} tensors`
                    : node.parameterCount
                      ? "+ weights"
                      : operation?.source
                        ? `L${operation.source.line}`
                        : ""}
                </small>
                {tensor && (
                  <button
                    className="node-enlarge"
                    aria-label={
                      operation
                        ? `Enlarge step ${operation.index + 1} tensor`
                        : `Enlarge ${rootLabel.toLowerCase()}`
                    }
                    title="Enlarge tensor in 3D"
                    onClick={(event) => {
                      event.stopPropagation();
                      onTensorInspect(tensor.id, 0);
                    }}
                  >
                    <Maximize size={12} />
                  </button>
                )}
              </div>
            </article>
          );
        })}
      </div>
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
        <button
          onClick={() => {
            const node = selectedId && byId.get(selectedId);
            if (node) onSelect(node.id);
            else if (graph.nodes[0]) onSelect(graph.nodes[0].id);
          }}
          aria-label="Focus selected tensor"
          title="Focus selected tensor"
        >
          <Focus size={17} />
        </button>
      </div>
      <div className="canvas-hint">
        Drag to pan <span>·</span> Pinch to zoom <span>·</span> Select a tensor
        to expand
      </div>
      <button
        className="canvas-minimap"
        title="Fit entire journey"
        aria-label="Journey overview — fit all"
        onClick={() => {
          onOverview();
          fit();
        }}
      >
        <svg viewBox={`0 0 ${graph.width} ${graph.height}`} aria-hidden="true">
          {graph.edges.map((edge) => {
            const a = byId.get(edge.source)!;
            const b = byId.get(edge.target)!;
            return (
              <path
                key={edge.id}
                d={`M${a.x + NODE_WIDTH / 2},${a.y + NODE_HEIGHT / 2}L${b.x + NODE_WIDTH / 2},${b.y + NODE_HEIGHT / 2}`}
              />
            );
          })}
          {graph.nodes.map((node) => (
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
    </div>
  );
}
