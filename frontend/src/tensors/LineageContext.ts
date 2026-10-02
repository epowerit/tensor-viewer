import { createContext, useContext } from "react";
import type { Tensor } from "../api/client";
import { explainedLineage, shortAxis, type AxisStory } from "./axisLineage";

/** Axis stories of the displayed run, provided where a whole trace is known. */
export const LineageContext = createContext<
  ((tensorId: string) => AxisStory[]) | null
>(null);

/** Where this tensor's axes come from, or null when they are simply its own. */
export function useAxisOrigins(tensor: Tensor): AxisStory[] | null {
  const lineage = useContext(LineageContext);
  return explainedLineage(tensor, lineage?.(tensor.id));
}

/**
 * A tensor's axis labels: its own names, and for an unnamed axis the origin
 * the run traced ("x.rows", "x.columns[1/2]") before falling back to "axis 2".
 */
export function useAxisNames(tensor: Tensor): (axis: number) => string {
  const origins = useAxisOrigins(tensor);
  return (axis) => {
    const own = tensor.axes[axis];
    if (own && !/^axis \d+$/.test(own)) return own;
    const origin = origins?.[axis];
    return (origin && shortAxis(origin)) || own || `axis ${axis}`;
  };
}
