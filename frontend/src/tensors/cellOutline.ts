type Point = readonly [number, number];

/** Round a projected cell outline without changing its indexed geometry.
 * Corner cuts stay on the original edges, so their quadratic curves remain
 * inside a convex face. Short edges bound the cuts even at edge-on angles.
 */
export function roundedCellPath(
  points: readonly Point[],
  radius = 1.6,
): string {
  if (!points.length || points.some((p) => p.some((n) => !Number.isFinite(n))))
    return "";
  const format = (point: Point) => `${point[0]} ${point[1]}`;
  if (points.length < 3)
    return `M ${format(points[0])}${points
      .slice(1)
      .map((point) => ` L ${format(point)}`)
      .join("")} Z`;

  const cut = Number.isFinite(radius) ? Math.max(0, radius) : 1.6;
  const between = (from: Point, to: Point, fraction: number): Point => [
    from[0] * (1 - fraction) + to[0] * fraction,
    from[1] * (1 - fraction) + to[1] * fraction,
  ];
  const corners = points.map((point, index) => {
    const previous = points[(index + points.length - 1) % points.length];
    const next = points[(index + 1) % points.length];
    const previousLength = Math.hypot(
      previous[0] - point[0],
      previous[1] - point[1],
    );
    const nextLength = Math.hypot(next[0] - point[0], next[1] - point[1]);
    const distance = Math.min(cut, previousLength * 0.12, nextLength * 0.12);
    // A finite coordinate span can overflow when subtracted. A zero cut in
    // that direction preserves the corner instead of emitting Infinity/NaN.
    return {
      point,
      incoming: between(
        point,
        previous,
        previousLength ? distance / previousLength : 0,
      ),
      outgoing: between(point, next, nextLength ? distance / nextLength : 0),
    };
  });

  return `M ${format(corners[0].outgoing)}${corners
    .map((_, index) => {
      const corner = corners[(index + 1) % corners.length];
      return ` L ${format(corner.incoming)} Q ${format(corner.point)} ${format(corner.outgoing)}`;
    })
    .join("")} Z`;
}
