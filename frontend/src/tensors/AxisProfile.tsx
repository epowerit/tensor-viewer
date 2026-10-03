import { useMemo, useState } from "react";
import type { Tensor } from "../api/client";
import { formatValue } from "./coordinates";
import { asNumber } from "./margins";
import { axisProfile, defaultProfileAxis, type ProfileStat } from "./profile";
import { useTensorQuery } from "./useTensorQuery";

/** Axes longer than this are drawn as a line rather than one bar each. */
const BARS = 512;

const STATS: [ProfileStat, string][] = [
  ["mean", "Mean"],
  ["std", "σ"],
  ["min", "Min"],
  ["max", "Max"],
];

/**
 * One statistic for each index of an axis, over every other axis: a
 * per-channel mean, say, as a bar chart. Choosing a bar moves the selection
 * to that index. Large tensors are profiled by the backend.
 */
export function AxisProfileChart({
  tensor,
  runId,
  coords,
  axisName,
  onPick,
}: {
  tensor: Tensor;
  runId?: string;
  coords: number[];
  axisName: (axis: number) => string;
  onPick: (axis: number, index: number) => void;
}) {
  const [axis, setAxis] = useState(() => defaultProfileAxis(tensor));
  const [stat, setStat] = useState<ProfileStat>("mean");
  const inline = tensor.values.length === tensor.numel;
  const local = useMemo(
    () => (axis !== null && inline ? axisProfile(tensor, axis) : null),
    [tensor, axis, inline],
  );
  const remote = useTensorQuery<Record<ProfileStat, (number | string)[]>>(
    runId,
    tensor.id,
    "axis",
    axis !== null && !inline && tensor.value_source === "paged"
      ? { axis }
      : null,
  );
  if (axis === null) return null;
  const values =
    local?.[stat] ?? remote.data?.[stat].map((value) => asNumber(value));
  const finite = values?.filter(Number.isFinite) ?? [];
  const low = Math.min(0, ...finite),
    high = Math.max(0, ...finite);
  const width = 220,
    height = 40;
  const span = high - low || 1;
  const y = (value: number) => height - ((value - low) / span) * height;
  const step = values ? width / values.length : 0;
  return (
    <div className="axis-profile">
      <div className="axis-profile-controls">
        <span>Per index of</span>
        <select
          aria-label="Axis to profile"
          value={axis}
          onChange={(event) => setAxis(Number(event.target.value))}
        >
          {tensor.shape.map((size, index) => (
            <option key={index} value={index}>
              {axisName(index)} · {size}
            </option>
          ))}
        </select>
        <select
          aria-label="Statistic"
          value={stat}
          onChange={(event) => setStat(event.target.value as ProfileStat)}
        >
          {STATS.map(([value, label]) => (
            <option key={value} value={value}>
              {label}
            </option>
          ))}
        </select>
      </div>
      {!values ? (
        <p className="axis-profile-note">
          {remote.error ||
            (remote.loading
              ? "Profiling every value…"
              : "Values load by window, so this tensor cannot be profiled.")}
        </p>
      ) : (
        <>
          <svg
            width={width}
            height={height}
            viewBox={`0 0 ${width} ${height}`}
            role="group"
            aria-label={`${STATS.find(([value]) => value === stat)![1]} of each ${axisName(axis)} index`}
          >
            {low < 0 && high > 0 && (
              <line
                className="axis-profile-zero"
                x1={0}
                x2={width}
                y1={y(0)}
                y2={y(0)}
              />
            )}
            {values.length > BARS ? (
              // Too many indices for bars: a line, and a click picks the
              // index under the pointer.
              <>
                <polyline
                  className="axis-profile-line"
                  points={values
                    .map((value, index) =>
                      Number.isFinite(value)
                        ? `${((index + 0.5) * step).toFixed(1)},${y(value).toFixed(1)}`
                        : null,
                    )
                    .filter(Boolean)
                    .join(" ")}
                />
                <line
                  className="axis-profile-chosen-line"
                  x1={(coords[axis] + 0.5) * step}
                  x2={(coords[axis] + 0.5) * step}
                  y1={0}
                  y2={height}
                />
                <rect
                  className="axis-profile-hit"
                  width={width}
                  height={height}
                  onClick={(event) => {
                    const box = event.currentTarget.getBoundingClientRect();
                    onPick(
                      axis,
                      Math.min(
                        values.length - 1,
                        Math.floor(
                          ((event.clientX - box.left) / box.width) *
                            values.length,
                        ),
                      ),
                    );
                  }}
                />
              </>
            ) : (
              values.map((value, index) => {
                const broken = !Number.isFinite(value);
                const top = broken ? 0 : Math.min(y(value), y(0));
                const barHeight = broken
                  ? height
                  : Math.max(1, Math.abs(y(value) - y(0)));
                return (
                  <rect
                    key={index}
                    className={`${index === coords[axis] ? "axis-profile-chosen" : ""}${broken ? " axis-profile-broken" : ""}`}
                    x={index * step + (step > 3 ? 0.5 : 0)}
                    y={top}
                    width={Math.max(0.6, step - (step > 3 ? 1 : 0))}
                    height={barHeight}
                    onClick={() => onPick(axis, index)}
                  >
                    <title>{`${axisName(axis)} ${index}: ${formatValue(value)}`}</title>
                  </rect>
                );
              })
            )}
          </svg>
          <span className="axis-profile-axis">
            <span>
              {finite.length ? formatValue(Math.min(...finite)) : "—"}
            </span>
            <span>
              {axisName(axis)} {coords[axis]}:{" "}
              {formatValue(values[coords[axis]])}
            </span>
            <span>
              {finite.length ? formatValue(Math.max(...finite)) : "—"}
            </span>
          </span>
        </>
      )}
    </div>
  );
}
