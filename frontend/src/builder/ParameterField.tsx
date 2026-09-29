import type { ToolboxItem } from "../api/client";

/** Use the actual incoming axes instead of asking learners to guess dimension numbers. */
export function ParameterField({
  parameter,
  value,
  title,
  kind,
  shape,
  axes,
  onChange,
}: {
  parameter: ToolboxItem["parameters"][number];
  value: number;
  title: string;
  kind: string;
  shape?: number[];
  axes: string[];
  onChange: (value: number) => void;
}) {
  const name = `${title} ${parameter.label}`;
  if (parameter.key === "keepdim")
    return (
      <label className="parameter-toggle">
        <span>
          <b>Keep reduced axis</b>
          <small>Retain a dimension of size 1</small>
        </span>
        <input
          type="checkbox"
          role="switch"
          aria-label={`${title} Keep reduced axis`}
          checked={value === 1}
          onChange={(e) => onChange(e.target.checked ? 1 : 0)}
        />
      </label>
    );
  if (parameter.key.startsWith("axis") && shape) {
    const insertion = kind === "unsqueeze" || kind === "stack_join";
    const size = shape.length + Number(insertion);
    const current = value < 0 ? size + value : value;
    return (
      <label>
        {parameter.label}
        <select
          aria-label={name}
          value={current}
          onChange={(e) => onChange(Number(e.target.value))}
        >
          {(current < 0 || current >= size) && (
            <option value={current}>Invalid axis ({value})</option>
          )}
          {Array.from({ length: size }, (_, i) => (
            <option key={i} value={i}>
              {insertion
                ? `${i} · ${i === shape.length ? "After the last axis" : `Before ${axes[i] || `axis ${i}`}`}`
                : `${i} · ${axes[i] || `axis ${i}`} (${shape[i]})`}
            </option>
          ))}
        </select>
        <small className="field-hint">
          {insertion
            ? "Position of the new dimension"
            : "Axis numbers start at zero"}
        </small>
      </label>
    );
  }
  return (
    <label>
      {parameter.label}
      <input
        type="number"
        step={1}
        min={parameter.min}
        max={parameter.max}
        aria-label={name}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}
