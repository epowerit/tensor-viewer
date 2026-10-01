import { useEffect, useMemo, useRef, useState } from "react";
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
  useEffect(() => {
    dialog.current?.showModal();
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
  return (
    <dialog
      ref={dialog}
      className="command-palette"
      aria-label="Command palette"
      onCancel={onClose}
      onClick={(event) => {
        if (event.target === dialog.current) onClose();
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
  );
}
