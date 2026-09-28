export function ValuesToggle({
  checked,
  onChange,
  shapeOnly = false,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  shapeOnly?: boolean;
}) {
  if (shapeOnly)
    return (
      <span className="shape-preview-label">
        Shapes only · no numeric values
      </span>
    );
  return (
    <label
      className="values-toggle"
      title="Cell shading shows magnitude; paged tensors use the visible window's scale. Cell text is rounded; select a cell for the recorded value."
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
      Show values
    </label>
  );
}
