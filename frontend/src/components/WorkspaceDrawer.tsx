import { useEffect, useRef, type ReactNode } from "react";
import { X } from "lucide-react";

/** Settings stay mounted so closing the drawer preserves unfinished fields. */
export function WorkspaceDrawer({
  active,
  className,
  label,
  title,
  eyebrow,
  closeLabel,
  returnLabel,
  onClose,
  children,
  footer,
}: {
  active: boolean;
  className: string;
  label: string;
  title: string;
  eyebrow: string;
  closeLabel: string;
  returnLabel: string;
  onClose: () => void;
  children: ReactNode;
  footer?: ReactNode;
}) {
  const panel = useRef<HTMLElement>(null);
  useEffect(() => {
    // A validation or source-navigation effect may already have selected a
    // specific field. Keep that focus instead of replacing it on drawer open.
    if (active && !panel.current?.contains(document.activeElement))
      panel.current?.focus({ preventScroll: true });
  }, [active]);
  return (
    <aside
      ref={panel}
      className={`workspace-drawer ${className}`}
      hidden={!active}
      aria-label={label}
      tabIndex={-1}
      onKeyDown={(event) => {
        if (
          event.key !== "Escape" ||
          event.defaultPrevented ||
          event.nativeEvent.isComposing ||
          (event.target instanceof Element &&
            event.target.closest("dialog[open]"))
        )
          return;
        event.preventDefault();
        event.stopPropagation();
        onClose();
      }}
    >
      <header className="drawer-heading">
        <div>
          <span className="eyebrow">{eyebrow}</span>
          <h2>{title}</h2>
        </div>
        <button
          className="icon-button"
          aria-label={closeLabel}
          title={`${returnLabel} (Esc)`}
          onClick={onClose}
        >
          <X size={18} />
        </button>
      </header>
      <div className="drawer-scroll">{children}</div>
      {footer}
    </aside>
  );
}
