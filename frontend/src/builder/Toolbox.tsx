import { useEffect, useRef, useState } from "react";
import {
  ArrowRightLeft,
  ArrowLeft,
  Blend,
  Box,
  Boxes,
  CircleDot,
  ChevronRight,
  Grid2X2,
  Layers3,
  Maximize,
  Minimize,
  MoveHorizontal,
  Network,
  Plus,
  Puzzle,
  Search,
  SlidersHorizontal,
  SquareStack,
  Waves,
  X,
  type LucideIcon,
} from "lucide-react";
import type { ToolboxItem } from "../api/client";
import "./toolbox.css";
import { filterToolboxItems, matchesInputTool } from "./toolboxSearch";

export const TOOL_GROUPS = [
  { id: "All", label: "All tools", icon: Grid2X2 },
  { id: "Custom", label: "Custom", icon: Puzzle },
  { id: "Models", label: "Models", icon: Network },
  { id: "Spatial", label: "Spatial", icon: SquareStack },
  { id: "Sequence", label: "Sequence", icon: Waves },
  { id: "Layers", label: "Layers", icon: Layers3 },
  { id: "Activations", label: "Activations", icon: Blend },
  { id: "Shape adapters", label: "Shape", icon: ArrowRightLeft },
] as const;
export type ToolGroup = (typeof TOOL_GROUPS)[number]["id"];
const CATEGORY_DESCRIPTIONS: Record<Exclude<ToolGroup, "All">, string> = {
  Models: "Attention, transformers, patch embedding",
  Spatial: "Convolution, pooling, windows",
  Sequence: "Recurrent layers and token tools",
  Layers: "Linear, normalization, branch connections",
  Activations: "ReLU, GELU, softmax and more",
  "Shape adapters": "Flatten, split, merge and reorder axes",
  Custom: "Your saved PyTorch components",
};
const BROWSE_GROUPS = [
  ...TOOL_GROUPS.filter((g) => g.id !== "All" && g.id !== "Custom"),
  TOOL_GROUPS.find((g) => g.id === "Custom")!,
];
const ICONS: Record<string, LucideIcon> = {
  custom: Puzzle,
  attention: Network,
  transformer: Boxes,
  vit: Boxes,
  hierarchical_vit: Layers3,
  spatial_embedding: Grid2X2,
  window_partition: Grid2X2,
  window_reverse: SquareStack,
  window_attention: Network,
  shifted_window: Network,
  window_pair: Layers3,
  add_join: Network,
  concat_join: Network,
  stack_join: Layers3,
  patch_merging: Minimize,
  spatial_readout: CircleDot,
  token_preparation: Plus,
  class_readout: CircleDot,
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

export function ComponentToolbar({
  active,
  onSelect,
  disabled,
}: {
  active: ToolGroup | null;
  onSelect: (group: ToolGroup) => void;
  disabled: boolean;
}) {
  return (
    <div
      className="component-toolbar toolbox-toolbar"
      role="toolbar"
      aria-label="Component tools"
      onKeyDown={(event) => {
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key))
          return;
        const buttons = Array.from(
          event.currentTarget.querySelectorAll<HTMLButtonElement>(
            "button:not(:disabled)",
          ),
        );
        const index = buttons.indexOf(
          document.activeElement as HTMLButtonElement,
        );
        if (index < 0) return;
        event.preventDefault();
        const next =
          event.key === "Home"
            ? 0
            : event.key === "End"
              ? buttons.length - 1
              : (index +
                  (event.key === "ArrowRight" ? 1 : -1) +
                  buttons.length) %
                buttons.length;
        buttons[next]?.focus();
      }}
    >
      <span className="toolbar-caption">Add to canvas</span>
      {TOOL_GROUPS.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          className={`component-tool ${active === id ? "active" : ""}`}
          aria-label={`Toolbox: ${label}`}
          title={label}
          aria-expanded={active === id}
          aria-controls="component-toolbox"
          disabled={disabled}
          onClick={() => onSelect(id)}
        >
          <Icon size={18} />
          <span>{label}</span>
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
  onCreateCustom,
  onGroupSelect,
}: {
  group: ToolGroup;
  catalog: ToolboxItem[];
  onAdd: (item: ToolboxItem) => void;
  onInput: () => void;
  onClose: () => void;
  limit: boolean;
  error: string;
  onRetry: () => void;
  onCreateCustom: () => void;
  onGroupSelect: (group: ToolGroup) => void;
}) {
  const [query, setQuery] = useState("");
  const search = useRef<HTMLInputElement>(null);
  const panel = useRef<HTMLElement>(null);
  const hasQuery = !!query.trim();
  const browsing = group === "All" && !hasQuery;
  const showInput = browsing || matchesInputTool(query);
  useEffect(() => {
    setQuery("");
    search.current?.focus();
  }, [group]);
  const matching = filterToolboxItems(catalog, query).filter(
    (item) => hasQuery || group === "All" || item.group === group,
  );
  const resultCount = matching.length + (showInput ? 1 : 0);
  return (
    <aside
      ref={panel}
      id="component-toolbox"
      className="component-toolbox toolbox-glass"
      aria-label="Component toolbox"
      onKeyDown={(e) => {
        if (e.key === "Escape") {
          e.stopPropagation();
          onClose();
        }
        if (e.key === "ArrowDown" || e.key === "ArrowUp") {
          const items = Array.from(
            panel.current?.querySelectorAll<HTMLButtonElement>(
              ".toolbox-results button:not(:disabled)",
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
          <span className="eyebrow">Component library</span>
          <h2>
            {hasQuery
              ? "Search results"
              : group === "All"
                ? "All tools"
                : group}
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
      {group !== "All" && !hasQuery && (
        <button className="toolbox-back" onClick={() => onGroupSelect("All")}>
          <ArrowLeft size={13} /> All tools
        </button>
      )}
      <div className="toolbox-results">
        {group === "Custom" && !hasQuery && (
          <button
            className="toolbox-custom-create"
            onClick={onCreateCustom}
            disabled={limit}
          >
            <Puzzle size={17} />
            <span>
              New custom component<small>Bring your own PyTorch module</small>
            </span>
            <Plus size={15} />
          </button>
        )}
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
        {browsing && (
          <div className="toolbox-categories">
            {BROWSE_GROUPS.map(({ id, icon: Icon }) => (
              <button
                className="toolbox-category"
                key={id}
                onClick={() => onGroupSelect(id)}
                aria-label={`Browse ${id}`}
              >
                <Icon size={17} />
                <span>
                  <b>{id}</b>
                  <small>{CATEGORY_DESCRIPTIONS[id]}</small>
                </span>
                <span className="toolbox-category-count">
                  {catalog.filter((item) => item.group === id).length}
                </span>
                <ChevronRight size={13} />
              </button>
            ))}
          </div>
        )}
        {!browsing &&
          TOOL_GROUPS.filter((g) => g.id !== "All").map((g) => {
            const items = matching.filter((c) => c.group === g.id);
            return items.length ? (
              <section key={g.id}>
                {hasQuery && (
                  <h3>
                    {g.id}
                    <span>{items.length}</span>
                  </h3>
                )}
                {items.map((item) => (
                  <button
                    className="toolbox-item"
                    key={item.custom?.id ?? item.kind}
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
                      {item.custom && (
                        <small className="custom-library-version">
                          {item.custom.class_name} ·{" "}
                          {item.custom.id.slice(0, 8)}
                        </small>
                      )}
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
            <b>
              {group === "Custom" && !hasQuery
                ? "Your library starts here"
                : "No matching components"}
            </b>
            <p>
              {group === "Custom" && !hasQuery
                ? "Save a module once. Use it across your experiments."
                : "Try “pool”, “attention”, or “axis”."}
            </p>
            {hasQuery && (
              <button
                className="secondary-button"
                onClick={() => {
                  setQuery("");
                  search.current?.focus();
                }}
              >
                Clear search
              </button>
            )}
          </div>
        )}
      </div>
      <footer className="toolbox-footer">
        <span role="status" aria-live="polite" aria-atomic="true">
          {limit
            ? "Sequence limit reached · remove a component to add another."
            : hasQuery
              ? `${resultCount} result${resultCount === 1 ? "" : "s"} across all tools`
              : browsing
                ? `${catalog.length} components · choose a category`
                : `${matching.length} component${matching.length === 1 ? "" : "s"} · select to add`}
        </span>
        <div className="toolbox-key-hint">
          <span>
            <kbd>↑ ↓</kbd> Navigate
          </span>
          <span>
            <kbd>Enter</kbd> Select
          </span>
          <span>
            <kbd>Esc</kbd> Close
          </span>
        </div>
      </footer>
    </aside>
  );
}
