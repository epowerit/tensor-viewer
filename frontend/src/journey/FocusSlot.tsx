import { createContext, useContext, type ReactNode } from "react";
import { createPortal } from "react-dom";

/** The focused view's header row, where a step can put its controls. */
export const FocusSlotContext = createContext<HTMLElement | null>(null);

/**
 * Controls that belong in the focus header when there is one, so they do
 * not take a row of the stage; elsewhere they render where they are.
 */
export function FocusSlot({ children }: { children: ReactNode }) {
  const slot = useContext(FocusSlotContext);
  return slot ? createPortal(children, slot) : <>{children}</>;
}
