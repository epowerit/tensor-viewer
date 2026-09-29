import { ravel } from "./coordinates";

export type AxisEntry = { index: number; position: number };
export type AxisGap = { first: number; last: number; position: number };
export type AxisSample = {
  entries: AxisEntry[];
  gaps: AxisGap[];
  extent: number;
};
export type Point3 = [number, number, number];
export type Camera = { yaw: number; pitch: number };
export const INITIAL_CAMERA: Camera = { yaw: -32, pitch: 24 };
export const FULL_AXIS_LIMIT = 16;
export const VOLUME_CELL_LIMIT = 1024;
export const THUMBNAIL_CELL_LIMIT = 512;

/** Turntable controls in screen directions: right/down are positive.
 * SVG y grows downward, so pitch must decrease to move the near surface down.
 * Keep the vertical axis upright and stop short of the poles. */
export function turnCamera(
  camera: Camera,
  rightDegrees: number,
  downDegrees: number,
): Camera {
  return {
    yaw: ((((camera.yaw + rightDegrees + 180) % 360) + 360) % 360) - 180,
    pitch: Math.max(-80, Math.min(80, camera.pitch - downDegrees)),
  };
}

/** The final index is always present. Gaps are real omitted index intervals. */
export function sampleAxis(
  size: number,
  selected?: number,
  isolate = false,
  expand = true,
): AxisSample {
  if (size <= 0) return { entries: [], gaps: [], extent: 0 };
  const valid =
    selected !== undefined &&
    Number.isInteger(selected) &&
    selected >= 0 &&
    selected < size;
  const indices =
    isolate && valid
      ? [selected]
      : size <= 3 || (expand && size <= FULL_AXIS_LIMIT)
        ? Array.from({ length: size }, (_, i) => i)
        : [...new Set([0, 1, ...(valid ? [selected] : []), size - 1])].sort(
            (a, b) => a - b,
          );
  const entries: AxisEntry[] = [],
    gaps: AxisGap[] = [];
  let position = 0;
  indices.forEach((index, i) => {
    const previous = indices[i - 1];
    if (i && index > previous + 1) {
      gaps.push({
        first: previous + 1,
        last: index - 1,
        position,
      });
      position += 1;
    }
    entries.push({ index, position });
    position += 1;
  });
  return { entries, gaps, extent: Math.max(1, position) };
}

export type Voxel = { flat: number; coords: number[]; center: Point3 };
export type VolumeBlock = { index: number | null; voxels: Voxel[] };

/** Last three axes form a volume. The preceding axis forms separate volumes.
 * Earlier axes remain explicitly fixed at the selected coordinate. Never flatten
 * unrelated batch/head axes into an unlabeled stack. Expand small axes first;
 * condense only as needed to respect the total visible-cell budget. */
export function volumeLayout(
  shape: number[],
  coordinates: number[],
  isolatedAxis?: number,
  cellLimit = VOLUME_CELL_LIMIT,
) {
  const rank = shape.length;
  const spatialAxes = [rank - 1, rank - 2, rank - 3].map((axis) =>
    axis >= 0 ? axis : null,
  );
  const outerAxis = rank >= 4 ? rank - 4 : null;
  const visibleAxes = [...spatialAxes, outerAxis].filter(
    (axis): axis is number => axis !== null,
  );
  const expanded = new Set(
    visibleAxes.filter((axis) => shape[axis] <= FULL_AXIS_LIMIT),
  );
  // Reserve room for an interior selection so clicking a cell never changes
  // which other dimensions are compressed. On ties, condense the outer axis.
  const count = () =>
    visibleAxes.reduce(
      (total, axis) =>
        total *
        (axis === isolatedAxis
          ? 1
          : expanded.has(axis)
            ? shape[axis]
            : Math.min(shape[axis], 4)),
      1,
    );
  const candidates = [...expanded]
    .filter((axis) => axis !== isolatedAxis && shape[axis] > 4)
    .sort((a, b) => shape[b] - shape[a] || a - b);
  for (const axis of candidates) {
    if (count() <= cellLimit) break;
    expanded.delete(axis);
  }
  const samples = spatialAxes.map((axis) =>
    axis === null
      ? sampleAxis(1)
      : sampleAxis(
          shape[axis],
          coordinates[axis],
          isolatedAxis === axis,
          expanded.has(axis),
        ),
  );
  const groups =
    outerAxis === null
      ? sampleAxis(1)
      : sampleAxis(
          shape[outerAxis],
          coordinates[outerAxis],
          isolatedAxis === outerAxis,
          expanded.has(outerAxis),
        );
  const blocks: VolumeBlock[] = groups.entries.map((group) => {
    const voxels: Voxel[] = [];
    if (shape.some((n) => n === 0)) return { index: group.index, voxels };
    for (const z of samples[2].entries)
      for (const y of samples[1].entries)
        for (const x of samples[0].entries) {
          const coords = [...coordinates];
          if (outerAxis !== null) coords[outerAxis] = group.index;
          [x, y, z].forEach((entry, i) => {
            const axis = spatialAxes[i];
            if (axis !== null) coords[axis] = entry.index;
          });
          voxels.push({
            flat: ravel(coords, shape),
            coords,
            center: [
              x.position - (samples[0].extent - 1) / 2,
              y.position - (samples[1].extent - 1) / 2,
              z.position - (samples[2].extent - 1) / 2,
            ],
          });
        }
    return { index: outerAxis === null ? null : group.index, voxels };
  });
  const hasGaps =
    groups.gaps.length > 0 || samples.some((sample) => sample.gaps.length > 0);
  return { spatialAxes, samples, outerAxis, groups, blocks, hasGaps };
}

export function rotate([x, y, z]: Point3, camera: Camera): Point3 {
  const yaw = (camera.yaw * Math.PI) / 180,
    pitch = (camera.pitch * Math.PI) / 180;
  const rx = x * Math.cos(yaw) + z * Math.sin(yaw);
  const rz = -x * Math.sin(yaw) + z * Math.cos(yaw);
  return [
    rx,
    y * Math.cos(pitch) - rz * Math.sin(pitch),
    y * Math.sin(pitch) + rz * Math.cos(pitch),
  ];
}

const FACES: { normal: Point3; corners: Point3[] }[] = [
  {
    normal: [0, 0, 1],
    corners: [
      [-0.5, -0.5, 0.5],
      [0.5, -0.5, 0.5],
      [0.5, 0.5, 0.5],
      [-0.5, 0.5, 0.5],
    ],
  },
  {
    normal: [0, 0, -1],
    corners: [
      [0.5, -0.5, -0.5],
      [-0.5, -0.5, -0.5],
      [-0.5, 0.5, -0.5],
      [0.5, 0.5, -0.5],
    ],
  },
  {
    normal: [1, 0, 0],
    corners: [
      [0.5, -0.5, 0.5],
      [0.5, -0.5, -0.5],
      [0.5, 0.5, -0.5],
      [0.5, 0.5, 0.5],
    ],
  },
  {
    normal: [-1, 0, 0],
    corners: [
      [-0.5, -0.5, -0.5],
      [-0.5, -0.5, 0.5],
      [-0.5, 0.5, 0.5],
      [-0.5, 0.5, -0.5],
    ],
  },
  {
    normal: [0, -1, 0],
    corners: [
      [-0.5, -0.5, -0.5],
      [0.5, -0.5, -0.5],
      [0.5, -0.5, 0.5],
      [-0.5, -0.5, 0.5],
    ],
  },
  {
    normal: [0, 1, 0],
    corners: [
      [-0.5, 0.5, 0.5],
      [0.5, 0.5, 0.5],
      [0.5, 0.5, -0.5],
      [-0.5, 0.5, -0.5],
    ],
  },
];

export function voxelFaces(center: Point3, camera: Camera) {
  return FACES.map((face) => ({
    visibility: rotate(face.normal, camera)[2],
    points: face.corners.map((p) =>
      rotate(p.map((v, i) => v + center[i]) as Point3, camera),
    ),
  })).filter((face) => face.visibility > 0.001);
}
