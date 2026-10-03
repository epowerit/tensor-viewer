import { useEffect, useMemo, useRef } from "react";
import type { Tensor } from "../api/client";
import { sliceThumbnails, type Thumbnails } from "./sliceThumbnails";
import { heatFill, heatLevel } from "./heat";
import { asNumber } from "./margins";
import type { Plane } from "./plane";
import { useTensorQuery } from "./useTensorQuery";

/** Thumbnails are about this wide, whatever their block count. */
const THUMB = 24;

/**
 * Every slice of a hidden axis at a glance, such as each channel of a
 * feature map: a small averaged picture per index, shaded on one scale.
 * Choosing one moves the grid to that slice. Large tensors are pictured by
 * the backend.
 */
export function Filmstrip({
  tensor,
  runId,
  plane,
  coords,
  axis,
  axisName,
  onPick,
}: {
  tensor: Tensor;
  runId?: string;
  plane: Plane;
  coords: number[];
  axis: number;
  axisName: string;
  onPick: (index: number) => void;
}) {
  // The pictures depend only on the other hidden coordinates.
  const others = coords
    .map((coordinate, i) =>
      i === axis || i === plane.row || i === plane.column ? 0 : coordinate,
    )
    .join(",");
  const local = useMemo(
    () => sliceThumbnails(tensor, plane, others.split(",").map(Number), axis),
    [tensor, plane, others, axis],
  );
  const remote = useTensorQuery<{
    rows: number;
    columns: number;
    total: number;
    thumbs: (number | string)[][];
  }>(
    runId,
    tensor.id,
    "thumbnails",
    !local && tensor.value_source === "paged"
      ? {
          row: plane.row ?? -1,
          column: plane.column ?? -1,
          axis,
          fixed: others,
        }
      : null,
  );
  const film: Thumbnails | null =
    local ??
    (remote.data && {
      ...remote.data,
      thumbs: remote.data.thumbs.map((thumb) => thumb.map(asNumber)),
    });
  const chosen = coords[axis];
  const strip = useRef<HTMLDivElement>(null);
  useEffect(() => {
    strip.current
      ?.querySelector(`[data-slice="${chosen}"]`)
      ?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [chosen, film]);
  if (!film)
    return remote.error ? (
      <p className="filmstrip-note">{remote.error}</p>
    ) : null;
  const cell = Math.max(
    3,
    Math.floor(THUMB / Math.max(film.rows, film.columns)),
  );
  const finite = film.thumbs.flat().filter(Number.isFinite);
  const low = Math.min(...finite),
    high = Math.max(...finite);
  return (
    <div className="filmstrip">
      <div
        className="filmstrip-strip"
        ref={strip}
        role="group"
        aria-label={`Every ${axisName} slice`}
      >
        {film.thumbs.map((thumb, index) => (
          <button
            key={index}
            type="button"
            data-slice={index}
            className="filmstrip-slice"
            aria-pressed={index === chosen}
            aria-label={`${axisName} ${index}`}
            title={`${axisName} ${index}`}
            onClick={() => onPick(index)}
          >
            <svg
              width={film.columns * cell}
              height={film.rows * cell}
              aria-hidden="true"
            >
              {thumb.map((value, block) => {
                const level = heatLevel(value, low, high);
                return (
                  <rect
                    key={block}
                    x={(block % film.columns) * cell}
                    y={Math.floor(block / film.columns) * cell}
                    width={cell}
                    height={cell}
                    fill={
                      level === null
                        ? "var(--danger, #efa99b)"
                        : heatFill(level)
                    }
                  />
                );
              })}
            </svg>
            <span>{index}</span>
          </button>
        ))}
      </div>
      {film.total > film.thumbs.length && (
        <p className="filmstrip-note">
          The first {film.thumbs.length} of {film.total.toLocaleString()}{" "}
          {axisName} slices.
        </p>
      )}
    </div>
  );
}
