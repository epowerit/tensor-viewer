import { createContext, useContext } from "react";
import type { Tensor } from "../api/client";
import { explainedLineage, type AxisStory } from "./axisLineage";

/** Axis stories of the displayed run, provided where a whole trace is known. */
export const LineageContext = createContext<
  ((tensorId: string) => AxisStory[]) | null
>(null);

/** Where this tensor's axes come from, or null when they are simply its own. */
export function useAxisOrigins(tensor: Tensor): AxisStory[] | null {
  const lineage = useContext(LineageContext);
  return explainedLineage(tensor, lineage?.(tensor.id));
}
