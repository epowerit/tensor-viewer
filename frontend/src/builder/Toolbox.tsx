import { useEffect, useRef, useState } from "react";
import {
  ArrowRightLeft,
  Blend,
  Box,
  Boxes,
  CircleDot,
  Grid2X2,
  Layers3,
  Maximize,
  Minimize,
  MoveHorizontal,
  Network,
  Plus,
  Search,
  SlidersHorizontal,
  SquareStack,
  Waves,
  X,
  type LucideIcon,
} from "lucide-react";
import type { ToolboxItem } from "../api/client";

export const TOOL_GROUPS = [
  { id: "All", label: "All tools", icon: Grid2X2 },
  { id: "Models", label: "Models", icon: Network },
  { id: "Spatial", label: "Spatial", icon: SquareStack },
  { id: "Sequence", label: "Sequence", icon: Waves },
  { id: "Layers", label: "Layers", icon: Layers3 },
  { id: "Activations", label: "Activations", icon: Blend },
  { id: "Shape adapters", label: "Shape", icon: ArrowRightLeft },
] as const;
export type ToolGroup = (typeof TOOL_GROUPS)[number]["id"];
const ICONS: Record<string, LucideIcon> = {
  attention: Network,
  transformer: Boxes,
  patch_embedding: Grid2X2,
  mlp: Layers3,
  conv2d: SquareStack,
  conv1d: Waves,
  rnn: Waves,
  linear: MoveHorizontal,
  maxpool2d: Minimize,
  avgpool2d: Minimize,
  adaptiveavgpool2d: Minimize,
  layernorm: SlidersHorizontal,
  batchnorm2d: SlidersHorizontal,
  relu: Blend,
  gelu: Blend,
  sigmoid: Blend,
  tanh: Waves,
  softmax: CircleDot,
  tokens: Grid2X2,
  flatten: MoveHorizontal,
  transpose: ArrowRightLeft,
  split_axis: Maximize,
  merge_axes: Minimize,
  unsqueeze: Plus,
  squeeze: Minimize,
  unfold: SquareStack,
  contiguous: Box,
  mean: Blend,
};
export function ComponentIcon({
  kind,
  size = 18,
}: {
  kind: string;
  size?: number;
}) {
  const Icon = ICONS[kind] ?? Box;
  return <Icon size={size} />;
}

export function ToolboxRail({
  active,
  onSelect,
  disabled,
}: {
  active: ToolGroup | null;
  onSelect: (group: ToolGroup) => void;
  disabled: boolean;
}) {
  return (
    <div className="rail-tool-section" aria-label="Toolbox categories">
      <span className="rail-section-label">TOOLS</span>
      {TOOL_GROUPS.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          className={`rail-tool ${active === id ? "active" : ""}`}
          aria-label={`Toolbox: ${label}`}
          title={label}
          aria-expanded={active === id}
          aria-controls="component-toolbox"
          disabled={disabled}
          onClick={() => onSelect(id)}
        >
          <Icon size={18} />
          <span className="rail-tooltip">{label}</span>
        </button>
      ))}
    </div>
  );
}

export function ToolboxPanel({
  group,
  catalog,
  onAdd,
  onInput,
  onClose,
  limit,
  error,
  onRetry,
}: {
  group: ToolGroup;
  catalog: ToolboxItem[];
  onAdd: (item: ToolboxItem) => void;
  onInput: () => void;
  onClose: () => void;
  limit: boolean;
  error: string;
  onRetry: () => void;
}) {
  const [query, setQuery] = useState("");
  const search = useRef<HTMLInputElement>(null);
  const panel = useRef<HTMLElement>(null);
  const showInput =
    (group === "All" && !query) ||
    (!!query.trim() &&
      "input tensor dimensions values".includes(query.toLowerCase().trim()));
  useEffect(() => {
    setQuery("");
    search.current?.focus();
  }, [group]);
  const matching = catalog.filter(
    (c) =>
      (query || group === "All" || c.group === group) &&
      `${c.title} ${c.description} ${c.group}`
        .toLowerCase()
        .includes(query.toLowerCase().trim()),
  );
  return (
    <aside
      ref={panel}
      id="component-toolbox"
      className="component-toolbox"
      aria-label="Component toolbox"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.stopPropagation();
          onClose();
        }
        if (e.key === "ArrowDown" || e.key === "ArrowUp") {
          const items = Array.from(
            panel.current?.querySelectorAll<HTMLButtonElement>(
              ".toolbox-item:not(:disabled)",
            ) ?? [],
          );
          const index = items.indexOf(
            document.activeElement as HTMLButtonElement,
          );
          if (document.activeElement === search.current || index >= 0) {
            e.preventDefault();
            const next =
              index < 0
                ? e.key === "ArrowDown"
                  ? 0
                  : items.length - 1
                : index + (e.key === "ArrowDown" ? 1 : -1);
            if (next < 0) search.current?.focus();
            else items[Math.min(next, items.length - 1)]?.focus();
          }
        }
      }}
    >
      <header className="toolbox-title">
        <div>
          <span className="eyebrow">COMPONENT LIBRARY</span>
          <h2>
            {query ? "Search results" : group === "All" ? "All tools" : group}
          </h2>
        </div>
        <button
          className="icon-button"
          aria-label="Close toolbox"
          onClick={onClose}
        >
          <X size={16} />
        </button>
      </header>
      <div className="toolbox-search-wrap">
        <Search size={15} />
        <input
          ref={search}
          aria-label="Search components"
          className="toolbox-search"
          placeholder="Search all components…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        {query && (
          <button
            className="icon-button"
            aria-label="Clear component search"
            onClick={() => {
              setQuery("");
              search.current?.focus();
            }}
          >
            <X size={13} />
          </button>
        )}
      </div>
      <div className="toolbox-results">
        {showInput && (
          <button className="toolbox-item input-tool" onClick={onInput}>
            <span className="tool-icon">
              <Box size={18} />
            </span>
            <div>
              <b>Input tensor</b>
              <small>Set dimensions and starting values</small>
            </div>
            <Plus size={14} />
          </button>
        )}
        {TOOL_GROUPS.filter((g) => g.id !== "All").map((g) => {
          const items = matching.filter((c) => c.group === g.id);
          return items.length ? (
            <section key={g.id}>
              <h3>
                {g.id}
                <span>{items.length}</span>
              </h3>
              {items.map((item) => (
                <button
                  className="toolbox-item"
                  key={item.kind}
                  aria-label={`Add ${item.title}`}
                  disabled={limit}
                  onClick={() => onAdd(item)}
                >
                  <span className="tool-icon">
                    <ComponentIcon kind={item.kind} />
                  </span>
                  <div>
                    <b>{item.title}</b>
                    <small>{item.description}</small>
                  </div>
                  <Plus size={13} />
                </button>
              ))}
            </section>
          ) : null;
        })}
        {!catalog.length && !error && (
          <p className="toolbox-note" role="status">
            Loading components…
          </p>
        )}
        {error && (
          <div className="toolbox-note" role="alert">
            {error}
            <button className="secondary-button" onClick={onRetry}>
              Retry library
            </button>
          </div>
        )}
        {!!catalog.length && !matching.length && !showInput && (
          <div className="toolbox-no-results">
            <Search size={24} />
            <b>No matching components</b>
            <p>Try “pool”, “attention”, or “axis”.</p>
          </div>
        )}
      </div>
      <footer className="toolbox-footer">
        {limit
          ? "Sequence limit reached · remove a component to add another."
          : `${catalog.length} components · click to add to your sequence`}
        <div className="toolbox-key-hint">
          <span>
            <kbd>↑ ↓</kbd> Navigate
          </span>
          <span>
            <kbd>Enter</kbd> Add
          </span>
          <span>
            <kbd>Esc</kbd> Close
          </span>
        </div>
      </footer>
    </aside>
  );
}
