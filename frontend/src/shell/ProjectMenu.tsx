import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { Copy, Pencil, Pin, PinOff, RotateCcw, Trash2 } from "lucide-react";

export type ProjectAction = "rename" | "duplicate" | "pin" | "reset" | "delete";

const ACTIONS: {
  action: ProjectAction;
  label: string;
  icon: typeof Copy;
  keys: string;
}[] = [
  { action: "rename", label: "Rename", icon: Pencil, keys: "F2" },
  { action: "duplicate", label: "Duplicate", icon: Copy, keys: "" },
  { action: "pin", label: "Pin to top", icon: Pin, keys: "" },
  { action: "reset", label: "Reset to original", icon: RotateCcw, keys: "" },
  { action: "delete", label: "Delete", icon: Trash2, keys: "Del" },
];

/**
 * A project's actions, opened from its row's ⋯ button or by right-clicking
 * it: rename, duplicate, pin to the top of the list, reset a library model
 * to its original, delete. Arrow keys move through them, Enter picks
 * one, and Escape or a click elsewhere closes the menu.
 */
export function ProjectMenu({
  name,
  at,
  canDelete,
  pinned = false,
  library = false,
  onAction,
  onClose,
}: {
  name: string;
  /** Pinned to the top of the list: the pin item unpins. */
  pinned?: boolean;
  /** A library model, which can be reset to its original. */
  library?: boolean;
  /** Where the menu opens, in viewport pixels. */
  at: { x: number; y: number };
  /** The last project cannot be deleted: the workspace keeps one open. */
  canDelete: boolean;
  onAction: (action: ProjectAction) => void;
  onClose: () => void;
}) {
  const menu = useRef<HTMLDivElement>(null);
  useEffect(() => {
    menu.current?.querySelector<HTMLButtonElement>("button")?.focus();
    const away = (event: PointerEvent) => {
      if (!menu.current?.contains(event.target as Node)) onClose();
    };
    window.addEventListener("pointerdown", away, true);
    return () => window.removeEventListener("pointerdown", away, true);
  }, []);
  // Kept on screen: a menu opened near the bottom or right edge opens inward.
  const left = Math.min(at.x, window.innerWidth - 196);
  const top = Math.min(at.y, window.innerHeight - 200);
  return createPortal(
    <div
      ref={menu}
      className="project-menu"
      role="menu"
      aria-label={`Actions for ${name}`}
      style={{ left, top }}
      onKeyDown={(event) => {
        const items = [
          ...(menu.current?.querySelectorAll<HTMLButtonElement>(
            "button:not(:disabled)",
          ) ?? []),
        ];
        const at = items.indexOf(document.activeElement as HTMLButtonElement);
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault();
          const step = event.key === "ArrowDown" ? 1 : -1;
          items[(at + step + items.length) % items.length]?.focus();
        } else if (event.key === "Escape" || event.key === "Tab") {
          event.preventDefault();
          onClose();
        }
      }}
    >
      {ACTIONS.filter(({ action }) => action !== "reset" || library).map(
        ({ action, label, icon, keys }) => {
          const unpin = action === "pin" && pinned;
          const Icon = unpin ? PinOff : icon;
          return (
            <button
              key={action}
              type="button"
              role="menuitem"
              className={
                action === "delete" ? "project-menu-danger" : undefined
              }
              disabled={action === "delete" && !canDelete}
              title={
                action === "delete" && !canDelete
                  ? "The workspace keeps at least one project"
                  : undefined
              }
              onClick={() => {
                onClose();
                onAction(action);
              }}
            >
              <Icon size={13} aria-hidden="true" />
              <span>{unpin ? "Unpin" : label}</span>
              {keys && <kbd>{keys}</kbd>}
            </button>
          );
        },
      )}
    </div>,
    document.body,
  );
}
