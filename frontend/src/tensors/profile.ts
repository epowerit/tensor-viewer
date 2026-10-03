import type { Tensor } from "../api/client";
import { asNumber } from "./margins";

export type ProfileStat = "mean" | "std" | "min" | "max";
export type AxisProfile = Record<ProfileStat, number[]>;

/**
 * Mean, σ, min and max of each index along `axis`, over every other axis,
 * when every value was recorded inline. NaN spreads into its index, as in
 * torch and the backend's profile.
 */
export function axisProfile(
  tensor: Pick<Tensor, "shape" | "values" | "numel">,
  axis: number,
): AxisProfile | null {
  if (tensor.values.length !== tensor.numel || !tensor.numel) return null;
  const size = tensor.shape[axis];
  const inner = tensor.shape.slice(axis + 1).reduce((a, b) => a * b, 1);
  const groups = Array.from({ length: size }, () => [] as number[]);
  tensor.values.forEach((value, flat) => {
    groups[Math.floor(flat / inner) % size].push(asNumber(value));
  });
  const profile: AxisProfile = { mean: [], std: [], min: [], max: [] };
  for (const group of groups) {
    const broken = group.some(Number.isNaN);
    const mean = group.reduce((sum, value) => sum + value, 0) / group.length;
    const variance =
      group.reduce((sum, value) => sum + (value - mean) ** 2, 0) / group.length;
    profile.mean.push(broken ? NaN : mean);
    profile.std.push(broken ? NaN : Math.sqrt(variance));
    profile.min.push(broken ? NaN : Math.min(...group));
    profile.max.push(broken ? NaN : Math.max(...group));
  }
  return profile;
}

/** The axis worth profiling first: channels or features, else axis 1. */
export function defaultProfileAxis(
  tensor: Pick<Tensor, "shape" | "axes">,
): number | null {
  if (tensor.shape.length < 2) return null;
  const named = tensor.axes.findIndex((name) =>
    /channel|feature|head|hidden|embed/i.test(name ?? ""),
  );
  return named >= 0 ? named : 1;
}
