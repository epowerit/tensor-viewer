import { Fragment, useEffect, useId, useMemo, useRef, useState } from "react";
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
import { thumbnailFrame } from "./volumeProjection";
import { roundedCellPath } from "./cellOutline";
import { visibleCellLabels } from "./cellLabelVisibility";
import type { Plane } from "./plane";
import { useAxisInk, useCellPaint } from "./InkShape";
import "./tensorGlass.css";
import "./volumeHighlights.css";
import "./volumeInk.css";
import {
  INITIAL_CAMERA,
  THUMBNAIL_CELL_LIMIT,
  VOLUME_CELL_LIMIT,
  moveVolumeSelection,
  rotate,
  turnCamera,
  volumeLayout,
  voxelFaces,
  type Camera,
  type Point3,
} from "./volume";
import { ValueSpread } from "./ValueSpread";

/**
 * How a tensor is lit. Pending: its step has not run, so it has no values yet
 * and stays unlit. Receiving: its step is running. Active: the step's result,
 * on fire. Lit: it holds values from an earlier step.
 */
export type TensorLight = "pending" | "receiving" | "active" | "lit";

type Props = {
  tensor: Tensor;
  runId?: string;
  /** The chosen cell, which burns; none is chosen unless one is given. */
  selected?: number;
  light?: TensorLight;
  highlights?: readonly number[];
  keyboardNavigation?: boolean;
  onSelect?: (index: number) => void;
  onGap?: (axis: number, first: number, last: number) => void;
  compact?: boolean;
  showValues?: boolean;
  isolatedAxis?: number;
  plane?: Plane;
};

/** Orthographic projection of indexed unit cells; every layer uses the same geometry. */
export function TensorVolume({
  tensor,
  runId,
  selected: chosenCell,
  light = "lit",
  highlights,
  keyboardNavigation = false,
  onSelect,
  onGap,
  compact = false,
  showValues = true,
  isolatedAxis,
  plane,
}: Props) {
  const [camera, setCamera] = useState<Camera>(INITIAL_CAMERA);
  // Geometry and keyboard focus start at the first cell; only a chosen cell
  // of a tensor that holds values burns.
  const selected = chosenCell ?? 0;
  const burning = chosenCell !== undefined && light !== "pending";
  // Slice inspection uses a front view without changing the saved turntable.
  const viewCamera = plane ? { yaw: 0, pitch: 0 } : camera;
  const [hover, setHover] = useState<number | null>(null);
  const clipId = useId().replace(/:/g, "");
  // Index ticks and labels take each axis's ink, as shapes do everywhere else.
  const ink = useAxisInk(tensor);
  // Glass takes the color of where each value came from; selection burns.
  const paint = useCellPaint();
  const inkFill = (axis: number | null) =>
    axis !== null && ink?.[axis] ? { fill: ink[axis]!.colors[0] } : undefined;
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
  }, [selected, plane?.row, plane?.column]);
  const drag = useRef<{
    pointerId: number;
    x: number;
    y: number;
    camera: Camera;
    moved: boolean;
  } | null>(null);
  const suppressClick = useRef(false);
  const coords = tensor.numel
    ? unravel(Math.min(selected, tensor.numel - 1), tensor.shape)
    : tensor.shape.map(() => 0);
  const layout = useMemo(
    () =>
      volumeLayout(
        tensor.shape,
        coords,
        isolatedAxis,
        compact ? THUMBNAIL_CELL_LIMIT : VOLUME_CELL_LIMIT,
        plane,
      ),
    [
      tensor.shape,
      coords.join(","),
      isolatedAxis,
      compact,
      plane?.row,
      plane?.column,
    ],
  );
  // Glass faces reveal the volume, but values behind other cells must not
  // compete with the readable surface values. Keep all cells selectable and
  // use Slice/the exact readout for interior elements.
  const projectedBlocks = useMemo(
    () =>
      layout.blocks.map((block) => {
        const cells = block.voxels
          .map((voxel) => {
            const faces = voxelFaces(voxel.center, viewCamera, 0.96);
            const mainFace = faces.reduce(
              (best, face) => (face.visibility > best.visibility ? face : best),
              faces[0],
            );
            const labelCenter = mainFace.points.reduce(
              (sum, point) =>
                sum.map((n, axis) => n + point[axis] / 4) as Point3,
              [0, 0, 0] as Point3,
            );
            return {
              voxel,
              faces,
              mainFace,
              labelCenter,
              depth: rotate(voxel.center, viewCamera)[2],
            };
          })
          .sort((a, b) => a.depth - b.depth);
        const visibleLabels =
          !compact && showValues
            ? visibleCellLabels(
                cells.map(({ voxel, faces, labelCenter }) => ({
                  index: voxel.flat,
                  faces: faces.map((face) =>
                    face.points.map(
                      (point) => [point[0], point[1]] as [number, number],
                    ),
                  ),
                  labelPoint: [labelCenter[0], labelCenter[1]],
                })),
              )
            : new Set<number>();
        return { ...block, cells, visibleLabels };
      }),
    [layout, viewCamera.yaw, viewCamera.pitch, compact, showValues],
  );
  const indices = layout.blocks.flatMap((block) =>
    block.voxels.map((v) => v.flat),
  );
  // Highlight only existing sampled cells. Contributor coordinates must never
  // increase the geometry budget or pull additional slices into this view.
  const contributors = useMemo(
    () =>
      new Set(
        layout.blocks.flatMap((block) =>
          block.voxels
            .filter((voxel) => highlights?.includes(voxel.flat))
            .map((voxel) => voxel.flat),
        ),
      ),
    [layout, highlights],
  );
  const hasHighlights = !!highlights?.length;
  const data = useTensorValues(
    compact ? { ...tensor, value_source: "shape" } : tensor,
    compact ? undefined : runId,
    indices,
  );
  const unit = 28;
  // A camera-independent radius keeps rotation from resizing the scene.
  const radius =
    (Math.hypot(...layout.samples.map((s) => s.extent)) * unit) / 2;
  const thumbnail = thumbnailFrame(
    layout.samples.map((sample) => sample.extent),
    viewCamera,
    unit,
    tensor.shape.length > 0,
  );
  const boxWidth =
      compact || plane ? thumbnail.width : Math.max(140, radius * 2 + 64),
    boxHeight =
      compact || plane
        ? thumbnail.height + (layout.outerAxis !== null ? 24 : 0)
        : Math.max(130, radius * 2 + 72);
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
  function turn(right: number, down: number) {
    setCamera((c) => turnCamera(c, right, down));
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
    <div
      className={`tensor-volume volume-light-${light} ${compact ? "volume-compact" : ""} ${plane ? "volume-slice" : ""} ${hasHighlights ? "volume-contributions" : ""}`}
    >
      {!compact && !plane && (
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
        aria-label={
          plane
            ? `${tensor.name} selected 2D slice of ${tensor.shape.length === 0 ? "a scalar" : tensor.shape.length === 1 ? "a vector" : `${tensor.shape.length}-dimensional tensor`}`
            : `${tensor.name} indexed ${tensor.shape.length}-dimensional tensor`
        }
        onPointerDown={(e) => {
          if (compact || plane || e.button !== 0 || !e.isPrimary) return;
          e.stopPropagation();
          suppressClick.current = false;
          drag.current = {
            pointerId: e.pointerId,
            x: e.clientX,
            y: e.clientY,
            camera,
            moved: false,
          };
        }}
        onPointerMove={(e) => {
          if (plane) return;
          const start = drag.current;
          if (!start || start.pointerId !== e.pointerId) return;
          // A short press can leave the SVG before the drag captures the pointer.
          // Re-entering after release must never resume that abandoned gesture.
          if (e.buttons === 0) {
            drag.current = null;
            return;
          }
          const dx = e.clientX - start.x,
            dy = e.clientY - start.y;
          if (!start.moved && Math.abs(dx) + Math.abs(dy) < 4) return;
          start.moved = true;
          if (!e.currentTarget.hasPointerCapture(e.pointerId))
            e.currentTarget.setPointerCapture(e.pointerId);
          setCamera(turnCamera(start.camera, dx * 0.45, dy * 0.45));
        }}
        onPointerUp={(e) => {
          if (drag.current?.pointerId !== e.pointerId) return;
          suppressClick.current = drag.current.moved;
          drag.current = null;
          if (e.currentTarget.hasPointerCapture(e.pointerId))
            e.currentTarget.releasePointerCapture(e.pointerId);
        }}
        onPointerCancel={(e) => {
          if (drag.current?.pointerId !== e.pointerId) return;
          drag.current = null;
          suppressClick.current = false;
        }}
        onLostPointerCapture={(e) => {
          if (drag.current?.pointerId !== e.pointerId) return;
          suppressClick.current = drag.current.moved;
          drag.current = null;
        }}
        onClickCapture={(e) => {
          if (suppressClick.current && e.detail > 0) {
            e.preventDefault();
            e.stopPropagation();
          }
          suppressClick.current = false;
        }}
        onMouseLeave={() => setHover(null)}
      >
        <title>
          {tensor.numel === 0
            ? `Empty tensor. No cells are available${plane ? " in this selected 2D slice" : ""}.`
            : (plane
                ? "Selected 2D slice. Each square is one indexed element; undisplayed axes stay at the selected coordinate. "
                : "Each cube is one indexed element. ") +
              (layout.hasGaps
                ? "Dots mark omitted ranges in this large or dense view. Geometry is compressed only across those gaps."
                : "No indices are omitted within the displayed axes.")}
        </title>
        <defs>
          {/* Shared face gradients keep the glass tint consistent without
              adding filters or extra geometry for individual cells. */}
          <linearGradient
            id={`${clipId}-fire`}
            x1="0%"
            y1="0%"
            x2="70%"
            y2="100%"
          >
            <stop offset="0%" stopColor="#fff1a8" />
            <stop offset="45%" stopColor="#ffb347" />
            <stop offset="100%" stopColor="#f0542c" />
          </linearGradient>
          {["rest", "muted", "contributor", "selected"].flatMap((state) =>
            [0, 1, 2].map((light) => (
              <linearGradient
                id={`${clipId}-face-${state}-${light}`}
                key={`${state}-${light}`}
                x1="0%"
                y1="0%"
                x2="85%"
                y2="100%"
              >
                <stop
                  offset="0%"
                  style={{
                    stopColor: `color-mix(in srgb, var(--volume-${state}, #b4a0d4) ${42 + light * 8}%, #211c2c)`,
                  }}
                />
                <stop
                  offset="100%"
                  style={{
                    stopColor: `color-mix(in srgb, var(--volume-${state}, #b4a0d4) ${25 + light * 8}%, #211c2c)`,
                  }}
                />
              </linearGradient>
            )),
          )}
        </defs>
        {tensor.numel === 0 && (
          <text x={width / 2} y={height / 2} textAnchor="middle">
            Empty tensor
          </text>
        )}
        {projectedBlocks.map((block, bi) => {
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
                  style={inkFill(outer)}
                >
                  {axisName(outer)} [{block.index}]
                </text>
              )}
              {block.cells.map(({ voxel, faces, mainFace, labelCenter }) => {
                const facePoints = faces.map((face) =>
                  face.points.map((p) => project(p) as [number, number]),
                );
                const value = data.valueAt(voxel.flat);
                const focused = voxel.flat === selected;
                const chosen = burning && focused;
                const contributing = contributors.has(voxel.flat);
                const tint =
                  chosen || light === "pending"
                    ? null
                    : paint?.(tensor, voxel.flat);
                const title = `[${voxel.coords.join(", ")}]${tensor.value_source === "shape" ? " · shape only" : value === undefined ? " · expand to inspect values" : ` = ${value}`}`;
                return (
                  <g
                    key={voxel.flat}
                    className={`volume-cell ${chosen ? "volume-selected" : ""} ${tint ? "volume-inked" : ""}`}
                    style={
                      tint
                        ? ({ "--cell-ink": tint } as React.CSSProperties)
                        : undefined
                    }
                    data-volume-index={voxel.flat}
                    data-contributor={contributing ? true : undefined}
                    role={onSelect ? "button" : undefined}
                    tabIndex={
                      onSelect && (!compact || keyboardNavigation)
                        ? focused
                          ? 0
                          : -1
                        : undefined
                    }
                    aria-label={`${tensor.name} cell ${title}`}
                    aria-pressed={onSelect ? chosen : undefined}
                    onClick={(e) => {
                      if (!onSelect) return;
                      e.stopPropagation();
                      onSelect(voxel.flat);
                    }}
                    onKeyDown={(e) => {
                      if (!onSelect) return;
                      const next = moveVolumeSelection(
                        voxel.flat,
                        tensor.shape,
                        e.key,
                        plane,
                      );
                      if (next !== null) {
                        e.preventDefault();
                        e.stopPropagation();
                        keyboard.current = true;
                        onSelect(next);
                      } else if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        e.stopPropagation();
                        onSelect(voxel.flat);
                      }
                    }}
                    onMouseEnter={() => !compact && setHover(voxel.flat)}
                    onMouseLeave={() => !compact && setHover(null)}
                  >
                    <title>{title}</title>
                    {faces.map((face, fi) => (
                      <Fragment key={fi}>
                        <polygon
                          className="volume-cell-geometry"
                          aria-hidden="true"
                          points={facePoints[fi]
                            .map((p) => p.join(","))
                            .join(" ")}
                        />
                        <path
                          className="volume-cell-face"
                          d={roundedCellPath(facePoints[fi])}
                          fill={
                            chosen
                              ? `url(#${clipId}-fire)`
                              : `url(#${clipId}-face-${contributing ? "contributor" : hasHighlights ? "muted" : "rest"}-${Math.min(2, Math.floor(face.visibility * 3))})`
                          }
                          style={
                            {
                              "--face-light":
                                0.7 +
                                0.15 *
                                  Math.min(2, Math.floor(face.visibility * 3)),
                            } as React.CSSProperties
                          }
                        />
                      </Fragment>
                    ))}
                    <path
                      className="volume-cell-rim"
                      aria-hidden="true"
                      d={facePoints
                        .map((points) => roundedCellPath(points))
                        .join(" ")}
                    />
                    {!compact &&
                      showValues &&
                      value !== undefined &&
                      block.visibleLabels.has(voxel.flat) &&
                      mainFace &&
                      (() => {
                        const [x, y] = project(labelCenter);
                        const label = formatCellValue(value, 5);
                        const clip = `${clipId}-${voxel.flat}`;
                        return (
                          <>
                            <defs>
                              <clipPath id={clip}>
                                <path
                                  d={roundedCellPath(
                                    mainFace.points.map(
                                      (p) => project(p) as [number, number],
                                    ),
                                  )}
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
                                5.25,
                                18 / (label.length * 0.58),
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
                    const projected = rotate(unitAxis, viewCamera);
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
                        const [x, y] = project(rotate(p, viewCamera));
                        return (
                          <text
                            key={entry.index}
                            x={x}
                            y={y}
                            textAnchor="middle"
                            style={{
                              ...inkFill(axis),
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
                        const [x, y] = project(rotate(p, viewCamera));
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
                <span
                  key={axis}
                  style={
                    ink?.[axis] ? { color: ink[axis]!.colors[0] } : undefined
                  }
                  title={ink?.[axis]?.text}
                >
                  <b>{plane ? ["Columns", "Rows"][i] : ["X", "Y", "Z"][i]}</b>{" "}
                  {axisName(axis)} · {tensor.shape[axis].toLocaleString()}
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
          {tensor.value_source !== "shape" && (
            <ValueSpread
              tensor={tensor}
              value={data.valueAt(active) as number | string | undefined}
            />
          )}
          <p className="volume-caption">
            {indices.length.toLocaleString()} indexed cells shown of{" "}
            {tensor.numel.toLocaleString()}
            {tensor.numel === 0
              ? ` · Empty tensor.${plane ? " No cells in the selected 2D slice." : ""}`
              : plane
                ? ` · Selected 2D slice.${layout.hasGaps ? " … marks omitted ranges. Select a cell or gap to inspect it." : " Select a visible cell to inspect it."}`
                : layout.hasGaps
                  ? " · … marks omitted ranges. Select a cell or gap to inspect it."
                  : indices.length === tensor.numel
                    ? " · Complete tensor. Select a visible cell to inspect it."
                    : " · Selected slice. Select a visible cell to inspect it."}
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
