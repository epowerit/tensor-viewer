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

/** The final index is always present. Gaps are real omitted index intervals. */
export function sampleAxis(
  size: number,
  selected?: number,
  isolate = false,
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
      : size <= 3
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
        position: position + 0.25,
      });
      position += 1;
    }
    entries.push({ index, position });
    position += 1.16;
  });
  return { entries, gaps, extent: Math.max(1, position - 0.16) };
}

export type Voxel = { flat: number; coords: number[]; center: Point3 };
export type VolumeBlock = { index: number | null; voxels: Voxel[] };

/** Last three axes form a volume. The preceding axis forms separate volumes.
 * Earlier axes remain explicitly fixed at the selected coordinate. Never flatten
 * unrelated batch/head axes into an unlabeled stack. At most 4^4 = 256 cells. */
export function volumeLayout(
  shape: number[],
  coordinates: number[],
  isolatedAxis?: number,
) {
  const rank = shape.length;
  const spatialAxes = [rank - 1, rank - 2, rank - 3].map((axis) =>
    axis >= 0 ? axis : null,
  );
  const samples = spatialAxes.map((axis) =>
    axis === null
      ? sampleAxis(1)
      : sampleAxis(shape[axis], coordinates[axis], isolatedAxis === axis),
  );
  const outerAxis = rank >= 4 ? rank - 4 : null;
  const groups =
    outerAxis === null
      ? sampleAxis(1)
      : sampleAxis(
          shape[outerAxis],
          coordinates[outerAxis],
          isolatedAxis === outerAxis,
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
  return { spatialAxes, samples, outerAxis, groups, blocks };
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
