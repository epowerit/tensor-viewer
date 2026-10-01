import { rotate, type Camera, type Point3 } from "./volume";

/** Compact tensors cannot rotate, so frame their visible projection rather
 * than reserving the sphere needed by the rotatable inspector. The extra
 * 0.7 units encloses the silhouette's index labels and gap hit targets.
 */
export function thumbnailFrame(
  extents: number[],
  camera: Camera,
  unit: number,
  hasAxes = true,
) {
  const half = extents.map((extent) => extent / 2 + (hasAxes ? 0.7 : 0));
  let x = 0;
  let y = 0;
  for (const sx of [-1, 1])
    for (const sy of [-1, 1])
      for (const sz of [-1, 1]) {
        const p = rotate(
          [sx * half[0], sy * half[1], sz * half[2]] as Point3,
          camera,
        );
        x = Math.max(x, Math.abs(p[0]));
        y = Math.max(y, Math.abs(p[1]));
      }
  return { width: x * unit * 2 + 24, height: y * unit * 2 + 24 };
}
