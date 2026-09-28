import { useEffect, useId, useMemo, useRef, useState } from "react";
import {
  RotateCcw,
  Rotate3D,
  ArrowLeft,
  ArrowRight,
  ChevronUp,
  ChevronDown,
} from "lucide-react";
import type { Tensor } from "../api/client";
import { formatCellValue, ravel, unravel } from "./coordinates";
import { useTensorValues } from "./useTensorValues";
import {
  INITIAL_CAMERA,
  rotate,
  volumeLayout,
  voxelFaces,
  type Camera,
  type Point3,
} from "./volume";

type Props = {
  tensor: Tensor;
  runId?: string;
  selected?: number;
  onSelect?: (index: number) => void;
  onGap?: (axis: number, first: number, last: number) => void;
  compact?: boolean;
  showValues?: boolean;
  isolatedAxis?: number;
};

/** Orthographic projection of indexed unit cells; every layer uses the same geometry. */
export function TensorVolume({
  tensor,
  runId,
  selected = 0,
  onSelect,
  onGap,
  compact = false,
  showValues = true,
  isolatedAxis,
}: Props) {
  const [camera, setCamera] = useState<Camera>(INITIAL_CAMERA);
  const [hover, setHover] = useState<number | null>(null);
  const clipId = useId().replace(/:/g, "");
  const svg = useRef<SVGSVGElement>(null);
  const keyboard = useRef(false);
  useEffect(() => {
    setHover(null);
    if (keyboard.current) {
      svg.current
        ?.querySelector<SVGGElement>(`[data-volume-index="${selected}"]`)
        ?.focus({ preventScroll: true });
      keyboard.current = false;
    }
  }, [selected]);
  const drag = useRef<{
    x: number;
    y: number;
    camera: Camera;
    moved: boolean;
  } | null>(null);
  const coords = tensor.numel
    ? unravel(Math.min(selected, tensor.numel - 1), tensor.shape)
    : tensor.shape.map(() => 0);
  const layout = useMemo(
    () => volumeLayout(tensor.shape, coords, isolatedAxis),
    [tensor.shape, coords.join(","), isolatedAxis],
  );
  const indices = layout.blocks.flatMap((block) =>
    block.voxels.map((v) => v.flat),
  );
  const data = useTensorValues(
    compact ? { ...tensor, value_source: "shape" } : tensor,
    compact ? undefined : runId,
    indices,
  );
  const unit = 28;
  // A camera-independent radius keeps rotation from resizing the scene.
  const radius =
    (Math.hypot(...layout.samples.map((s) => s.extent)) * unit) / 2;
  const boxWidth = Math.max(140, radius * 2 + 64),
    boxHeight = Math.max(130, radius * 2 + 72);
  const outer = layout.outerAxis;
  const gaps = layout.groups.gaps;
  const outerGap = 36;
  const width =
    Math.max(1, layout.blocks.length) * boxWidth + gaps.length * outerGap;
  const height = boxHeight + (outer !== null ? 28 : 0);
  const blockX = (i: number) =>
    i * boxWidth +
    gaps.filter((g) => g.last < (layout.blocks[i].index ?? 0)).length *
      outerGap;
  const axisName = (axis: number) => tensor.axes[axis] || `axis ${axis}`;
  const active = hover ?? selected;
  function turn(yaw: number, pitch: number) {
    setCamera((c) => ({
      yaw: c.yaw + yaw,
      pitch: Math.max(-80, Math.min(80, c.pitch + pitch)),
    }));
  }
  function inspectGap(axis: number, first: number, last: number) {
    if (onGap) onGap(axis, first, last);
    else
      onSelect?.(
        ravel(
          coords.map((v, i) => (i === axis ? first : v)),
          tensor.shape,
        ),
      );
  }
  return (
    <div className={`tensor-volume ${compact ? "volume-compact" : ""}`}>
      {!compact && (
        <div className="volume-toolbar" aria-label="3D viewing controls">
          <span>
            <Rotate3D size={14} /> Drag to rotate
          </span>
          <div>
            <button
              aria-label="Rotate left"
              title="Rotate left"
              onClick={() => turn(-20, 0)}
            >
              <ArrowLeft size={13} />
            </button>
            <button
              aria-label="Rotate right"
              title="Rotate right"
              onClick={() => turn(20, 0)}
            >
              <ArrowRight size={13} />
            </button>
            <button
              aria-label="Rotate up"
              title="Rotate up"
              onClick={() => turn(0, -15)}
            >
              <ChevronUp size={13} />
            </button>
            <button
              aria-label="Rotate down"
              title="Rotate down"
              onClick={() => turn(0, 15)}
            >
              <ChevronDown size={13} />
            </button>
            <button onClick={() => setCamera({ yaw: 0, pitch: 0 })}>
              Front
            </button>
            <button
              aria-label="Reset rotation"
              title="Reset rotation"
              onClick={() => setCamera(INITIAL_CAMERA)}
            >
              <RotateCcw size={13} />
            </button>
          </div>
        </div>
      )}
      <svg
        ref={svg}
        className="volume-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="group"
        aria-label={`${tensor.name} indexed ${tensor.shape.length}-dimensional tensor`}
        onPointerDown={(e) => {
          if (compact || e.button !== 0) return;
          e.stopPropagation();
          drag.current = { x: e.clientX, y: e.clientY, camera, moved: false };
        }}
        onPointerMove={(e) => {
          const start = drag.current;
          if (!start) return;
          const dx = e.clientX - start.x,
            dy = e.clientY - start.y;
          if (Math.abs(dx) + Math.abs(dy) < 4) return;
          start.moved = true;
          if (!e.currentTarget.hasPointerCapture(e.pointerId))
            e.currentTarget.setPointerCapture(e.pointerId);
          setCamera({
            yaw: start.camera.yaw + dx * 0.45,
            pitch: Math.max(-80, Math.min(80, start.camera.pitch + dy * 0.4)),
          });
        }}
        onPointerUp={(e) => {
          if (e.currentTarget.hasPointerCapture(e.pointerId))
            e.currentTarget.releasePointerCapture(e.pointerId);
        }}
        onPointerCancel={() => {
          drag.current = null;
        }}
        onClick={(e) => {
          if (drag.current?.moved) e.stopPropagation();
          drag.current = null;
        }}
        onMouseLeave={() => setHover(null)}
      >
        <title>
          Each cube is one indexed element. Missing ranges are labeled with
          dots. Geometry is compressed across gaps, not proportional to
          dimension size.
        </title>
        {tensor.numel === 0 && (
          <text x={width / 2} y={height / 2} textAnchor="middle">
            Empty tensor
          </text>
        )}
        {layout.blocks.map((block, bi) => {
          const xOffset = blockX(bi) + boxWidth / 2,
            yOffset = boxHeight / 2;
          const project = (p: Point3) => [
            xOffset + p[0] * unit,
            yOffset + p[1] * unit,
          ];
          return (
            <g key={block.index ?? "volume"}>
              {outer !== null && (
                <text
                  className="volume-block-label"
                  x={xOffset}
                  y={18}
                  textAnchor="middle"
                >
                  {axisName(outer)} [{block.index}]
                </text>
              )}
              {[...block.voxels]
                .sort(
                  (a, b) =>
                    rotate(a.center, camera)[2] - rotate(b.center, camera)[2],
                )
                .map((voxel) => {
                  const faces = voxelFaces(voxel.center, camera);
                  const value = data.valueAt(voxel.flat);
                  const chosen = voxel.flat === selected;
                  const title = `[${voxel.coords.join(", ")}]${tensor.value_source === "shape" ? " · shape only" : value === undefined ? " · expand to inspect values" : ` = ${value}`}`;
                  const mainFace = faces.reduce(
                    (best, face) =>
                      face.visibility > best.visibility ? face : best,
                    faces[0],
                  );
                  return (
                    <g
                      key={voxel.flat}
                      className={`volume-cell ${chosen ? "volume-selected" : ""}`}
                      data-volume-index={voxel.flat}
                      role={onSelect ? "button" : undefined}
                      tabIndex={
                        onSelect && !compact ? (chosen ? 0 : -1) : undefined
                      }
                      aria-label={`${tensor.name} cell ${title}`}
                      aria-pressed={onSelect ? chosen : undefined}
                      onClick={(e) => {
                        if (!onSelect || drag.current?.moved) return;
                        e.stopPropagation();
                        onSelect(voxel.flat);
                      }}
                      onKeyDown={(e) => {
                        if (!onSelect) return;
                        const movement: Record<string, [number, number]> = {
                          ArrowLeft: [1, -1],
                          ArrowRight: [1, 1],
                          ArrowUp: [2, -1],
                          ArrowDown: [2, 1],
                          PageUp: [3, -1],
                          PageDown: [3, 1],
                        };
                        if (movement[e.key]) {
                          e.preventDefault();
                          e.stopPropagation();
                          const [offset, delta] = movement[e.key],
                            axis = tensor.shape.length - offset;
                          if (axis >= 0) {
                            const next = [...voxel.coords];
                            next[axis] = Math.max(
                              0,
                              Math.min(
                                tensor.shape[axis] - 1,
                                next[axis] + delta,
                              ),
                            );
                            keyboard.current = true;
                            onSelect(ravel(next, tensor.shape));
                          }
                        } else if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          e.stopPropagation();
                          onSelect(voxel.flat);
                        }
                      }}
                      onMouseEnter={() => !compact && setHover(voxel.flat)}
                    >
                      <title>{title}</title>
                      {faces.map((face, fi) => (
                        <polygon
                          key={fi}
                          points={face.points
                            .map((p) => project(p).join(","))
                            .join(" ")}
                          style={{
                            fill: `color-mix(in srgb, ${chosen ? "#c5b0ef" : "#8d75b5"} ${Math.round(25 + face.visibility * 30)}%, #20192c)`,
                          }}
                        />
                      ))}
                      {!compact &&
                        showValues &&
                        value !== undefined &&
                        mainFace &&
                        (() => {
                          const center = mainFace.points.reduce(
                            (sum, p) =>
                              sum.map((n, i) => n + p[i] / 4) as Point3,
                            [0, 0, 0] as Point3,
                          );
                          const [x, y] = project(center);
                          const label = formatCellValue(value, 5);
                          const clip = `${clipId}-${voxel.flat}`;
                          return (
                            <>
                              <defs>
                                <clipPath id={clip}>
                                  <polygon
                                    points={mainFace.points
                                      .map((p) => project(p).join(","))
                                      .join(" ")}
                                  />
                                </clipPath>
                              </defs>
                              <text
                                x={x}
                                y={y}
                                clipPath={`url(#${clip})`}
                                textAnchor="middle"
                                dominantBaseline="central"
                                fontSize={Math.min(
                                  7,
                                  21 / (label.length * 0.65),
                                )}
                              >
                                {label}
                              </text>
                            </>
                          );
                        })()}
                    </g>
                  );
                })}
              {(!compact || bi === 0) &&
                layout.spatialAxes.map((axis, si) => {
                  if (axis === null) return null;
                  // Put index labels on an outer silhouette edge instead of
                  // projecting the rear corner's labels over foreground cells.
                  const direction = [
                    [0, 1],
                    [-1, 0],
                    [0, -1],
                  ][si];
                  const base = layout.samples.map((s, i) => {
                    if (i === si) return 0;
                    const unitAxis: Point3 = [0, 0, 0];
                    unitAxis[i] = 1;
                    const projected = rotate(unitAxis, camera);
                    const sign =
                      Math.sign(
                        projected[0] * direction[0] +
                          projected[1] * direction[1],
                      ) || 1;
                    return sign * (s.extent / 2 + 0.7);
                  }) as Point3;
                  return (
                    <g key={axis} className="volume-axis">
                      {layout.samples[si].entries.map((entry) => {
                        const p = [...base] as Point3;
                        p[si] =
                          entry.position - (layout.samples[si].extent - 1) / 2;
                        const [x, y] = project(rotate(p, camera));
                        return (
                          <text
                            key={entry.index}
                            x={x}
                            y={y}
                            textAnchor="middle"
                            style={{
                              fontSize: Math.min(
                                9,
                                24 / (String(entry.index).length * 0.65),
                              ),
                            }}
                          >
                            {entry.index}
                          </text>
                        );
                      })}
                      {layout.samples[si].gaps.map((gap) => {
                        const p = [...base] as Point3;
                        p[si] =
                          gap.position - (layout.samples[si].extent - 1) / 2;
                        const [x, y] = project(rotate(p, camera));
                        return (
                          <g
                            key={gap.first}
                            role="button"
                            tabIndex={compact ? -1 : 0}
                            aria-label={`${axisName(axis)} omitted indices ${gap.first} to ${gap.last}`}
                            className="volume-gap"
                            onPointerDown={(e) => e.stopPropagation()}
                            onClick={(e) => {
                              e.stopPropagation();
                              inspectGap(axis, gap.first, gap.last);
                            }}
                            onKeyDown={(e) => {
                              if (e.key === "Enter" || e.key === " ") {
                                e.preventDefault();
                                e.stopPropagation();
                                inspectGap(axis, gap.first, gap.last);
                              }
                            }}
                          >
                            <title>
                              {axisName(axis)}: indices {gap.first}–{gap.last}.
                              Select to inspect this range.
                            </title>
                            <rect
                              x={x - 9}
                              y={y - 10}
                              width={18}
                              height={18}
                              rx={4}
                            />
                            <text x={x} y={y} textAnchor="middle">
                              …
                            </text>
                          </g>
                        );
                      })}
                    </g>
                  );
                })}
            </g>
          );
        })}
        {outer !== null &&
          gaps.map((gap) => {
            const next = layout.blocks.findIndex((b) => b.index! > gap.last);
            const x = blockX(next) - outerGap / 2;
            return (
              <g
                key={gap.first}
                className="volume-gap"
                role={onGap || onSelect ? "button" : undefined}
                tabIndex={onGap || onSelect ? (compact ? -1 : 0) : undefined}
                aria-label={`${axisName(outer)} omitted indices ${gap.first} to ${gap.last}`}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  inspectGap(outer, gap.first, gap.last);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    e.stopPropagation();
                    inspectGap(outer, gap.first, gap.last);
                  }
                }}
              >
                <title>
                  {axisName(outer)}: {gap.first}–{gap.last} omitted
                </title>
                <rect
                  x={x - 14}
                  y={height / 2 - 18}
                  width={28}
                  height={32}
                  rx={6}
                />
                <text x={x} y={height / 2} textAnchor="middle" fontSize={18}>
                  …
                </text>
              </g>
            );
          })}
      </svg>
      {!compact && (
        <>
          <div className="volume-axis-legend">
            {layout.spatialAxes.map((axis, i) =>
              axis === null ? null : (
                <span key={axis}>
                  <b>{["X", "Y", "Z"][i]}</b> {axisName(axis)} ·{" "}
                  {tensor.shape[axis].toLocaleString()}
                </span>
              ),
            )}
          </div>
          <div className="volume-readout">
            <span>{hover === null ? "Selected" : "Hover"}</span>
            <code>
              [{tensor.numel ? unravel(active, tensor.shape).join(", ") : ""}]
            </code>
            <strong>
              {tensor.value_source === "shape"
                ? "Shape only"
                : String(
                    data.valueAt(active) ??
                      (data.loading ? "Loading…" : "Value unavailable"),
                  )}
            </strong>
          </div>
          <p className="volume-caption">
            {indices.length.toLocaleString()} indexed cells shown of{" "}
            {tensor.numel.toLocaleString()} · gaps compress omitted ranges.
            Select a cell or … to inspect it.
          </p>
          {data.error && (
            <p role="alert" className="tensor-load-error">
              {data.error} <button onClick={data.retry}>Retry</button>
            </p>
          )}
        </>
      )}
    </div>
  );
}
