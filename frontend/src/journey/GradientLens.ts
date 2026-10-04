import { createContext } from "react";

/** A value whose gradient the gradient lens shows: a cell, or a whole sum. */
export type GradientTarget = { tensorId: string; index: number | null };

/** Lets a tensor card point the gradient lens at its selected cell. */
export const GradientLensContext = createContext<{
  show: (target: GradientTarget) => void;
  target: GradientTarget | null;
} | null>(null);
