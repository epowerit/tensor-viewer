import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Box, X } from "lucide-react";
import type { Tensor } from "../api/client";
import { useInspectionActive } from "../journey/InspectionActivity";
import { ravel, unravel } from "./coordinates";
import { CoordinateJump, IndexControl } from "./TensorNavigation";
import { TensorVolume } from "./TensorVolume";

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
  const [selected, setSelected] = useState(initialIndex);
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
    setSelected(index);
    onSelect?.(index);
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
        <button
          className="icon-button"
          onClick={onClose}
          aria-label="Close tensor volume"
        >
          <X size={20} />
        </button>
      </header>
      <div className="volume-dialog-body">
        <div className="volume-inspection">
          <TensorVolume
            tensor={tensor}
            runId={runId}
            selected={selected}
            onSelect={select}
            showValues={showValues}
            isolatedAxis={isolated}
            onGap={(axis, first, last) => {
              setGap({ axis, first, last });
              setIsolated(axis);
              select(
                ravel(
                  coords.map((v, i) => (i === axis ? first : v)),
                  tensor.shape,
                ),
              );
            }}
          />
        </div>
        <aside className="volume-controls" aria-label="Tensor coordinates">
          <h3>Inspect the tensor</h3>
          <p>
            Every visible cube is one element. Rotate the volume, or isolate a
            slice to see inside.
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
          <label className="volume-isolate">
            Layers
            <select
              aria-label="Isolate tensor axis"
              value={isolated ?? "all"}
              onChange={(e) => {
                setIsolated(
                  e.target.value === "all" ? undefined : Number(e.target.value),
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
          {gap && (
            <p className="volume-gap-note" role="status">
              Inspecting {tensor.axes[gap.axis] || `axis ${gap.axis}`}{" "}
              {coords[gap.axis]} from omitted range {gap.first}–{gap.last}.
              Choose any index below.
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
          {tensor.shape.length > 4 && (
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
            index, and your selected coordinate. Isolate a layer to see more
            detail. Indexing starts at zero.
          </p>
        </aside>
      </div>
    </dialog>,
    document.body,
  );
}
