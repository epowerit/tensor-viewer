/** A tiny drawing of a shape: rank as stacked layers, proportions from the last two sizes. */
export function ShapeGlyph({
  shape,
  failed = false,
}: {
  shape: number[];
  failed?: boolean;
}) {
  const rank = shape.length;
  const scale = (size: number) => Math.min(1, Math.log2(size + 1) / 5);
  const width = rank ? 5 + 9 * scale(shape[rank - 1]) : 4;
  const height = rank > 1 ? 4 + 7 * scale(shape[rank - 2]) : rank ? 4 : 4;
  const layers = Math.min(Math.max(rank - 2, 0), 3);
  return (
    <svg
      className={`shape-glyph ${failed ? "failed" : ""}`}
      viewBox="0 0 20 16"
      width="20"
      height="16"
      aria-hidden="true"
    >
      {Array.from({ length: layers + 1 }, (_, i) => {
        const offset = (layers - i) * 2;
        return (
          <rect
            key={i}
            x={1 + offset}
            y={15 - height - offset}
            width={width}
            height={height}
            rx={rank ? 1 : 2}
            opacity={i === layers ? 1 : 0.45}
          />
        );
      })}
    </svg>
  );
}
