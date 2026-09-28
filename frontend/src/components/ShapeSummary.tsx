/** Exact dimensions, with semantic axis labels that stay readable at large sizes. */
export function ShapeSummary({
  shape,
  axes = [],
  compact = false,
}: {
  shape: number[];
  axes?: string[];
  compact?: boolean;
}) {
  if (!shape.length)
    return (
      <div className="shape-scalar">
        Scalar <span>one value</span>
      </div>
    );
  return (
    <div
      className={`shape-summary ${compact ? "compact" : ""}`}
      aria-label={`Tensor shape [${shape.join(", ")}]`}
      style={{
        gridTemplateColumns: `repeat(${Math.min(shape.length, 4)}, minmax(0, 1fr))`,
      }}
    >
      {shape.map((size, index) => (
        <div
          className="shape-dimension"
          key={index}
          title={`Axis ${index} · ${axes[index] || `axis ${index}`} · ${size.toLocaleString("en-US")}`}
        >
          <b style={{ fontSize: String(size).length > 6 ? "10px" : undefined }}>
            {size}
          </b>
          <span>{axes[index] || `axis ${index}`}</span>
        </div>
      ))}
    </div>
  );
}
