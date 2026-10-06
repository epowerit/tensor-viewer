import type { ReactNode } from "react";
import { LoaderCircle } from "lucide-react";

/** A tab still working something out: the same spinner and tone in each. */
export function PanelLoading({ children }: { children: ReactNode }) {
  return (
    <p className="panel-empty panel-loading" role="status">
      <LoaderCircle size={12} className="spin" aria-hidden="true" />
      <span>{children}</span>
    </p>
  );
}
