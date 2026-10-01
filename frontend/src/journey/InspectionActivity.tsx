import { createContext, useContext } from "react";

// Standalone inspectors stay active; retained panels provide their visibility.
export const InspectionActivityContext = createContext(true);

export function useInspectionActive(): boolean {
  return useContext(InspectionActivityContext);
}
