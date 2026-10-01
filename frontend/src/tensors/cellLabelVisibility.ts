type Point = readonly [number, number];
type Face = readonly Point[];
type Cell = {
  index: number;
  faces: readonly Face[];
  labelPoint: Point;
};
type Bounds = { left: number; right: number; top: number; bottom: number };

function bounds(points: Face): Bounds {
  return points.reduce<Bounds>(
    (box, [x, y]) => ({
      left: Math.min(box.left, x),
      right: Math.max(box.right, x),
      top: Math.min(box.top, y),
      bottom: Math.max(box.bottom, y),
    }),
    { left: Infinity, right: -Infinity, top: Infinity, bottom: -Infinity },
  );
}
function within([x, y]: Point, box: Bounds): boolean {
  return x >= box.left && x <= box.right && y >= box.top && y <= box.bottom;
}

/** Boundary points count as covered so text cannot collide with a nearer rim. */
function contains([x, y]: Point, face: Face): boolean {
  let inside = false;
  for (let i = 0, j = face.length - 1; i < face.length; j = i++) {
    const [ax, ay] = face[j],
      [bx, by] = face[i];
    const dx = bx - ax,
      dy = by - ay;
    const cross = dx * (y - ay) - dy * (x - ax);
    const tolerance =
      Number.EPSILON *
      8 *
      Math.max(1, Math.abs(dx * (y - ay)), Math.abs(dy * (x - ax)));
    if (
      Math.abs(cross) <= tolerance &&
      x >= Math.min(ax, bx) &&
      x <= Math.max(ax, bx) &&
      y >= Math.min(ay, by) &&
      y <= Math.max(ay, by)
    )
      return true;
    if (ay > y !== by > y && x < ax + ((y - ay) * dx) / dy) inside = !inside;
  }
  return inside;
}

/** Cells arrive in back-to-front painter order. Glass keeps every face visible,
 * but only unoccluded value labels remain, without changing cell hit targets.
 * Precomputed cell/face bounds avoid polygon work for separated cells/gutters. */
export function visibleCellLabels(cells: readonly Cell[]): Set<number> {
  const geometry = cells.map((cell) => {
    const faces = cell.faces
      .filter((face) => face.length >= 3)
      .map((points) => ({ points, bounds: bounds(points) }));
    return {
      cell,
      faces,
      bounds: bounds(faces.flatMap((face) => face.points)),
    };
  });
  const visible = new Set<number>();
  for (let i = 0; i < geometry.length; i++) {
    const { cell } = geometry[i];
    let covered = false;
    for (let j = i + 1; j < geometry.length; j++) {
      const nearer = geometry[j];
      if (!within(cell.labelPoint, nearer.bounds)) continue;
      if (
        nearer.faces.some(
          (face) =>
            within(cell.labelPoint, face.bounds) &&
            contains(cell.labelPoint, face.points),
        )
      ) {
        covered = true;
        break;
      }
    }
    if (!covered) visible.add(cell.index);
  }
  return visible;
}
