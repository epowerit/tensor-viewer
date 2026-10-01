import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Box, Grid2X2, X } from "lucide-react";
import type { Tensor } from "../api/client";
import { useInspectionActive } from "../journey/InspectionActivity";
import { ravel, unravel } from "./coordinates";
import { CoordinateJump, IndexControl } from "./TensorNavigation";
import { TensorVolume } from "./TensorVolume";
import { changePlane, defaultPlane, hiddenAxes, safeIndex } from "./plane";
import "./tensorSlice.css";

export function TensorVolumeDialog({
  tensor,
  runId,
  initialIndex = 0,
  onSelect,
  onClose,
}: {
  tensor: Tensor;
  runId?: string;
  initialIndex?: number;
  onSelect?: (index: number) => void;
  onClose: () => void;
}) {
  const active = useInspectionActive();
  const dialog = useRef<HTMLDialogElement>(null);
  const dismissed = useRef(false);
  const latestClose = useRef(onClose);
  latestClose.current = onClose;
  const [selected, setSelected] = useState(() =>
    safeIndex(initialIndex, tensor.shape),
  );
  const [view, setView] = useState<"volume" | "slice">("volume");
  const [plane, setPlane] = useState(() => defaultPlane(tensor.shape.length));
  const [isolated, setIsolated] = useState<number | undefined>();
  const [gap, setGap] = useState<{
    axis: number;
    first: number;
    last: number;
  } | null>(null);
  const [showValues, setShowValues] = useState(true);
  const coords = tensor.numel
    ? unravel(selected, tensor.shape)
    : tensor.shape.map(() => 0);
  const axisName = (axis: number) => tensor.axes[axis] || `axis ${axis}`;
  const fixedAxes = hiddenAxes(tensor.shape, plane);
  useEffect(() => {
    if (!active) {
      if (!dismissed.current) {
        dismissed.current = true;
        latestClose.current();
      }
      return;
    }
    const element = dialog.current;
    if (!dismissed.current && element && !element.open) element.showModal();
    return () => {
      if (element?.open) element.close();
    };
  }, [active]);
  function select(index: number) {
    setGap(null);
    setSelected(index);
    onSelect?.(index);
  }
  function show(next: "volume" | "slice") {
    setView(next);
    setGap(null);
  }
  if (!active || dismissed.current) return null;
  return createPortal(
    <dialog
      ref={dialog}
      className="volume-dialog"
      aria-labelledby="volume-dialog-title"
      onCancel={onClose}
      onKeyDown={(event) => {
        if (event.key === "Escape") event.stopPropagation();
      }}
    >
      <header className="volume-dialog-heading">
        <div>
          <Box size={19} />
          <h2 id="volume-dialog-title">{tensor.name}</h2>
          <code>[{tensor.shape.join(", ") || "scalar"}]</code>
        </div>
        <div className="volume-dialog-actions">
          <div
            className="volume-view-switch"
            role="group"
            aria-label="Tensor view"
          >
            <button
              aria-pressed={view === "volume"}
              aria-label="3D volume view"
              onClick={() => show("volume")}
              title="Explore the tensor as a volume"
            >
              <Box size={14} /> Volume
            </button>
            <button
              aria-pressed={view === "slice"}
              aria-label="2D slice view"
              onClick={() => show("slice")}
              title="Reveal the selected cell in a flat slice"
            >
              <Grid2X2 size={14} /> Slice
            </button>
          </div>
          <button
            className="icon-button"
            onClick={onClose}
            aria-label="Close tensor volume"
          >
            <X size={20} />
          </button>
        </div>
      </header>
      <div className="volume-dialog-body">
        <div className="volume-inspection">
          {view === "slice" && (
            <div className="volume-plane-controls">
              {(["row", "column"] as const).map((side) => (
                <label key={side}>
                  {side === "row" ? "Rows" : "Columns"}
                  <select
                    aria-label={`Slice ${side} axis`}
                    value={plane[side] ?? "none"}
                    disabled={plane[side] === null || tensor.shape.length < 2}
                    onChange={(event) => {
                      setPlane(
                        changePlane(plane, side, Number(event.target.value)),
                      );
                      setGap(null);
                    }}
                  >
                    {plane[side] === null && (
                      <option value="none">
                        {side === "row" ? "One row" : "One column"}
                      </option>
                    )}
                    {tensor.shape.map((size, axis) => (
                      <option value={axis} key={axis}>
                        {axisName(axis)} · {size.toLocaleString()}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
            </div>
          )}
          {view === "slice" && !!tensor.numel && !!fixedAxes.length && (
            <div
              className="volume-slice-context"
              aria-label="Fixed slice coordinates"
            >
              <span>Fixed</span>
              {fixedAxes.map((axis) => (
                <code key={axis}>
                  {axisName(axis)} = {coords[axis]}
                </code>
              ))}
            </div>
          )}
          <TensorVolume
            tensor={tensor}
            runId={runId}
            selected={selected}
            onSelect={select}
            showValues={showValues}
            isolatedAxis={isolated}
            plane={view === "slice" ? plane : undefined}
            onGap={(axis, first, last) => {
              if (view === "volume") setIsolated(axis);
              select(
                ravel(
                  coords.map((v, i) => (i === axis ? first : v)),
                  tensor.shape,
                ),
              );
              setGap({ axis, first, last });
            }}
          />
        </div>
        <aside className="volume-controls" aria-label="Tensor coordinates">
          <h3>{view === "slice" ? "Explore a slice" : "Inspect the tensor"}</h3>
          <p>
            {view === "slice"
              ? "Choose the axes to display. Other coordinates select the slice; the tensor stays unchanged."
              : "Every visible cube is one element. Rotate to explore, or switch to Slice to reveal the selected cell inside."}
          </p>
          {tensor.value_source === "shape" ? (
            <span className="shape-preview-label">
              Shape metadata · no numeric values
            </span>
          ) : (
            <label className="volume-values">
              <input
                type="checkbox"
                checked={showValues}
                onChange={(e) => setShowValues(e.target.checked)}
              />{" "}
              Show cell values
            </label>
          )}
          {view === "volume" && (
            <label className="volume-isolate">
              Layers
              <select
                aria-label="Isolate tensor axis"
                value={isolated ?? "all"}
                onChange={(e) => {
                  setIsolated(
                    e.target.value === "all"
                      ? undefined
                      : Number(e.target.value),
                  );
                  setGap(null);
                }}
              >
                <option value="all">All sampled layers</option>
                {tensor.shape.map((_, i) => (
                  <option value={i} key={i}>
                    Isolate {tensor.axes[i] || `axis ${i}`}
                  </option>
                ))}
              </select>
            </label>
          )}
          {gap && (
            <p className="volume-gap-note" role="status">
              Inspecting {axisName(gap.axis)} {coords[gap.axis]} from omitted
              range {gap.first}–{gap.last}. Choose any index below.
            </p>
          )}
          <div className="volume-coordinate-controls">
            {!!tensor.numel &&
              tensor.shape.map((size, axis) => (
                <IndexControl
                  key={axis}
                  label={tensor.axes[axis] || `axis ${axis}`}
                  name={`Volume ${tensor.axes[axis] || `axis ${axis}`} index`}
                  value={coords[axis]}
                  size={size}
                  onChange={(value) =>
                    select(
                      ravel(
                        coords.map((v, i) => (i === axis ? value : v)),
                        tensor.shape,
                      ),
                    )
                  }
                />
              ))}
          </div>
          {!!tensor.numel && !!tensor.shape.length && (
            <CoordinateJump
              label="Volume"
              shape={tensor.shape}
              coords={coords}
              onSelect={select}
            />
          )}
          {view === "volume" && tensor.shape.length > 4 && (
            <p>
              Axes before{" "}
              {tensor.axes[tensor.shape.length - 4] ||
                `axis ${tensor.shape.length - 4}`}{" "}
              are fixed at the selected coordinate.
            </p>
          )}
          <p className="volume-convention">
            Small dimensions are shown in full. Larger or denser views use …
            only where indices are omitted, retaining the first two, the final
            index, and your selected coordinate. Arrow keys move between cells
            on the displayed axes. Indexing starts at zero.
          </p>
        </aside>
      </div>
    </dialog>,
    document.body,
  );
}
