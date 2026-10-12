import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Search } from "lucide-react";
import { filterCommands, type Command } from "./commands";

type Props = { commands: Command[]; onClose: () => void };

/** Jump to any step, tensor, project, or action by typing part of its name. */
export function CommandPalette({ commands, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null);
  const list = useRef<HTMLDivElement>(null);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const matches = useMemo(
    () => filterCommands(commands, query),
    [commands, query],
  );
  // Open without showModal: a modal dialog makes the rest of the page inert,
  // which restyled every element of the workspace on opening and again on
  // closing (some 50 ms each with a model on the canvas). The backdrop, the
  // Escape key and the kept focus stand in for it.
  useEffect(() => {
    const before = document.activeElement as HTMLElement | null;
    return () => {
      // A command that moved focus keeps it; otherwise it returns.
      const now = document.activeElement;
      if (!now || now === document.body || dialog.current?.contains(now))
        before?.focus?.({ preventScroll: true });
    };
  }, []);
  useEffect(() => setActive(0), [query]);
  useEffect(() => {
    list.current
      ?.querySelector('[aria-selected="true"]')
      ?.scrollIntoView({ block: "nearest" });
  }, [active]);
  function execute(command: Command | undefined) {
    if (!command || command.disabled) return;
    onClose();
    command.run();
  }
  // Above the whole workspace, as the top layer held a modal dialog.
  return createPortal(
    <>
      <div
        className="command-palette-backdrop"
        aria-hidden="true"
        onClick={onClose}
      />
      <dialog
        ref={dialog}
        open
        aria-modal="true"
        className="command-palette"
        aria-label="Command palette"
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            event.stopPropagation();
            onClose();
          } else if (event.key === "Tab") event.preventDefault();
        }}
      >
        <label className="command-search">
          <Search size={16} />
          <input
            autoFocus
            value={query}
            placeholder="Go to a step, tensor, project, or action"
            role="combobox"
            aria-expanded="true"
            aria-controls="command-results"
            aria-activedescendant={
              matches[active] ? `command-${matches[active].id}` : undefined
            }
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                event.preventDefault();
                const step = event.key === "ArrowDown" ? 1 : -1;
                if (matches.length)
                  setActive((active + step + matches.length) % matches.length);
              } else if (event.key === "Enter") {
                event.preventDefault();
                execute(matches[active]);
              }
            }}
          />
        </label>
        <div
          ref={list}
          id="command-results"
          className="command-results"
          role="listbox"
          aria-label="Matching commands"
        >
          {matches.map((command, i) => (
            <div
              key={command.id}
              id={`command-${command.id}`}
              role="option"
              aria-selected={i === active}
              aria-disabled={command.disabled || undefined}
              onMouseMove={() => setActive(i)}
              onClick={() => execute(command)}
            >
              {(i === 0 || matches[i - 1].group !== command.group) && (
                <span className="command-group">{command.group}</span>
              )}
              <span className="command-label">{command.label}</span>
              {command.detail && (
                <span className="command-detail">{command.detail}</span>
              )}
              {command.shortcut && <kbd>{command.shortcut}</kbd>}
            </div>
          ))}
          {!matches.length && (
            <p className="command-empty">Nothing matches “{query}”.</p>
          )}
        </div>
      </dialog>
    </>,
    // In the workspace root, which holds the theme's variables.
    document.querySelector(".ide") ?? document.body,
  );
}
