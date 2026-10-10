import {
  memo,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { GitCompare, Link2, Pin, Search, X } from "lucide-react";
import type { Run, Tensor } from "../api/client";
import { ShapeGlyph } from "../editor/ShapeGlyph";
import {
  describeAxis,
  explainedLineage,
  type AxisStory,
  lineageOf,
  originSummary,
} from "../tensors/axisLineage";
import { AxisInkContext, InkShape } from "../tensors/InkShape";
import { formatBytes, storageBytes, tensorBytes } from "../tensors/memory";
import {
  loadComparison,
  loadRun,
  stateChange,
  summaryChange,
  type StateChange,
} from "../tensors/diff";
import { heatFill, heatLevel, heatRangeOf, useDiff } from "../tensors/heat";
import { planeThumbnail } from "../tensors/sliceThumbnails";
import { usePins } from "./pins";
import { ContractContext } from "../editor/ContractContext";
import {
  filterShelf,
  keepUnchanged,
  pinFirst,
  sortShelf,
  variables,
  type ShelfSort,
  type Variable,
} from "./variables";
import "./tensorShelf.css";

type Props = {
  run: Run | null;
  selected: string | null;
  /** A folded card's steps, played as one: the shelf stands after the last. */
  range?: string[];
  /** Steps the last save added, edited, or reshaped, lit for a moment. */
  changed?: ReadonlySet<string> | null;
  onSelect: (nodeId: string) => void;
  /** Hovering or focusing a card follows its name: every node that wrote it. */
  onThread?: (nodeIds: string[] | null) => void;
  /** The run recorded before this one, for Diff. */
  previousRunId?: string | null;
};

const formatValue = (value: number) =>
  Math.abs(value) >= 1000 || (value !== 0 && Math.abs(value) < 0.01)
    ? value.toExponential(1)
    : value.toFixed(2);

/**
 * Every named tensor of the displayed run, like a debugger's locals: as it
 * stands at the selected step (or at the end of the run), with its range,
 * type and shared storage. Names assigned later wait, unlit.
 */
export function VariablesPanel({
  run,
  selected,
  range,
  changed,
  onSelect,
  onThread,
  previousRunId = null,
}: Props) {
  // A folded card played as one step stands after its last step, and every
  // name it wrote is just written.
  const indexOf = (id: string | null | undefined) =>
    run?.trace.operations.find((op) => op.id === id)?.index;
  const through = range?.length ? indexOf(range.at(-1)) : indexOf(selected);
  const since = range?.length ? (indexOf(range[0]) ?? 0) - 1 : undefined;
  // Each step lists every name afresh; a name whose state it did not change
  // keeps its object, so only the cards the step touches are drawn again.
  const listed = useRef<Variable[]>([]);
  const items = useMemo(() => {
    const next = keepUnchanged(
      listed.current,
      run ? variables(run.trace, through ?? Infinity, since) : [],
    );
    listed.current = next;
    return next;
  }, [run, through, since]);
  const lineage = useMemo(() => (run ? lineageOf(run.trace) : null), [run]);
  // The cards' handlers keep one identity and call the latest props.
  const latest = useRef({ onSelect, onThread });
  latest.current = { onSelect, onThread };
  const selectCard = useCallback(
    (nodeId: string) => latest.current.onSelect(nodeId),
    [],
  );
  const threadCard = useCallback(
    (nodeIds: string[] | null) => latest.current.onThread?.(nodeIds),
    [],
  );
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<ShelfSort>("order");
  // Diff, shared with every grid: how each name changed since the run before.
  const [diffOn, setDiff] = useDiff();
  const [earlier, setEarlier] = useState<Run | null>(null);
  useEffect(() => {
    if (!diffOn || !previousRunId || earlier?.id === previousRunId) return;
    let current = true;
    loadRun(previousRunId)
      .then((found) => current && setEarlier(found))
      .catch(() => {});
    return () => {
      current = false;
    };
  }, [diffOn, previousRunId, earlier?.id]);
  const local = useMemo(() => {
    if (!diffOn || !run || !earlier || earlier.id !== previousRunId)
      return null;
    return new Map(
      items
        .filter((item) => !item.pending)
        .map((item) => [
          item.name,
          stateChange(run.trace, earlier.trace, item.tensor.id),
        ]),
    );
  }, [diffOn, run, earlier, previousRunId, items]);
  // Tensors whose values stay in snapshots are compared by the backend, in
  // one request for the whole shelf.
  const pairs = useMemo(
    () =>
      local
        ? items.flatMap((item) => {
            const change = local.get(item.name);
            return change?.kind === "unknown"
              ? [[item.tensor.id, change.before] as [string, string]]
              : [];
          })
        : [],
    [local, items],
  );
  const [remote, setRemote] = useState<Map<string, StateChange | null>>(
    new Map(),
  );
  useEffect(() => {
    if (!pairs.length || !run || !earlier) return;
    let current = true;
    loadComparison(run.id, earlier.id, pairs)
      .then((results) => {
        if (current)
          setRemote(
            new Map(
              pairs.map(([id], i) => [
                `${run.id}/${id}`,
                summaryChange(results[i]),
              ]),
            ),
          );
      })
      .catch(() => {});
    return () => {
      current = false;
    };
  }, [pairs, run, earlier]);
  const changes = useMemo(
    () =>
      local &&
      new Map(
        [...local].map(([name, change]) => {
          const item = items.find((other) => other.name === name);
          const answered =
            change.kind === "unknown" && item && run
              ? remote.get(`${run.id}/${item.tensor.id}`)
              : null;
          return [name, answered ?? change];
        }),
      ),
    [local, remote, items, run],
  );
  const changeShare = useMemo(
    () =>
      changes
        ? new Map(
            [...changes].map(([name, change]) => [
              name,
              change.kind === "changed"
                ? change.changed / change.compared
                : change.kind === "same"
                  ? 0
                  : NaN,
            ]),
          )
        : undefined,
    [changes],
  );
  const order = sort === "change" && !changes ? "order" : sort;
  const [pins, togglePin] = usePins(run?.project_id ?? null);
  const shown = useMemo(
    () =>
      pinFirst(sortShelf(filterShelf(items, query), order, changeShare), pins),
    // Pins are compared by content: a new Set each render is the same pins.
    [items, query, order, changeShare, [...pins].join("\n")],
  );
  const held = items.filter((item) => !item.pending);
  // Names the run writes only after the playback step, still unlit.
  const waiting = items.length - held.length;
  const memory = storageBytes(held.map((item) => item.tensor));
  if (!items.length)
    return (
      <p className="tensor-shelf-empty">Tensors appear here after a run.</p>
    );
  return (
    <div className="tensor-shelf">
      <div className="tensor-shelf-toolbar">
        <label className="tensor-shelf-filter">
          <Search size={12} aria-hidden="true" />
          <input
            type="search"
            value={query}
            placeholder="Name, shape or dtype"
            aria-label="Filter tensors"
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Escape" && query) {
                event.preventDefault();
                event.stopPropagation();
                setQuery("");
              }
            }}
          />
        </label>
        <label className="tensor-shelf-sort">
          <span>Sort</span>
          <select
            value={sort}
            aria-label="Sort tensors"
            onChange={(event) => setSort(event.target.value as ShelfSort)}
          >
            <option value="order">Order of use</option>
            <option value="size">Size</option>
            <option value="magnitude">Largest value</option>
            <option value="name">Name</option>
            {changes && <option value="change">Most changed</option>}
          </select>
        </label>
        {previousRunId && (
          <button
            type="button"
            className="tensor-view-toggle"
            aria-pressed={diffOn}
            title="Compare every tensor with the run before; grids show each cell's change"
            onClick={() => setDiff(!diffOn)}
          >
            <GitCompare size={13} aria-hidden="true" />
            <span>Diff</span>
          </button>
        )}
        <span
          className="tensor-shelf-count"
          title="Views of one storage are counted once. Parameters are not included."
        >
          {shown.length === items.length
            ? `${items.length} ${items.length === 1 ? "tensor" : "tensors"}`
            : `${shown.length} of ${items.length}`}
          {memory > 0 && ` · ${formatBytes(memory)} of values`}
          {waiting > 0 && ` · ${waiting} not computed yet`}
        </span>
      </div>
      {!shown.length ? (
        <p className="tensor-shelf-empty">
          No tensor matches “{query.trim()}”.{" "}
          <button
            type="button"
            className="text-button"
            onClick={() => setQuery("")}
          >
            <X size={11} aria-hidden="true" /> Clear filter
          </button>
        </p>
      ) : (
        <ShelfGrid>
          {shown.map((item) => (
            <ShelfCard
              key={item.name}
              item={item}
              run={run}
              lineage={lineage}
              selected={item.nodeId === selected}
              changed={!!changed?.has(item.nodeId)}
              pinned={pins.has(item.name)}
              change={changes?.get(item.name)}
              onSelect={selectCard}
              onThread={threadCard}
              onTogglePin={togglePin}
            />
          ))}
        </ShelfGrid>
      )}
    </div>
  );
}

/**
 * One name's card. A step redraws only the cards whose state, selection or
 * pin it changes: the rest keep their props, handlers included.
 */
const ShelfCard = memo(function ShelfCard({
  item,
  run,
  lineage,
  selected,
  changed,
  pinned,
  change,
  onSelect,
  onThread,
  onTogglePin,
}: {
  item: Variable;
  run: Run | null;
  lineage: ((tensorId: string) => AxisStory[] | null) | null;
  selected: boolean;
  changed: boolean;
  pinned: boolean;
  change?: StateChange;
  onSelect: (nodeId: string) => void;
  onThread: (nodeIds: string[] | null) => void;
  onTogglePin: (name: string) => void;
}) {
  const inkFor = useContext(AxisInkContext);
  const contracts = useContext(ContractContext);
  const shape = item.tensor.shape.length
    ? `[${item.tensor.shape.join(", ")}]`
    : "scalar";
  const shared = item.sharedWith.join(", ");
  const stories = explainedLineage(item.tensor, lineage?.(item.tensor.id));
  // Axes that only restate their own names are left out of the line.
  const origin = stories && originSummary(stories, item.tensor.axes);
  const originFull = stories?.map(describeAxis).join(" · ");
  const originDetail = stories
    ?.map(
      (story, axis) =>
        `${item.tensor.axes[axis] ?? `axis ${axis}`}: ${describeAxis(story)}`,
    )
    .join("\n");
  const { minimum, maximum } = item.tensor;
  const range =
    !item.pending && typeof minimum === "number" && typeof maximum === "number"
      ? `${formatValue(minimum)} … ${formatValue(maximum)}`
      : null;
  const broken = item.pending ? 0 : (item.tensor.histogram?.non_finite ?? 0);
  const step = run?.trace.operations.find((op) => op.id === item.nodeId);
  const bytes = tensorBytes(item.tensor);
  return (
    <li className="tensor-shelf-item">
      <button
        type="button"
        className={`tensor-shelf-card${item.anonymous ? " tensor-shelf-anonymous" : ""}${item.pending ? " tensor-shelf-pending" : ""}${item.fresh ? " tensor-shelf-fresh" : ""}${changed ? " tensor-shelf-changed" : ""}`}
        aria-pressed={selected}
        aria-label={`Inspect ${item.name}, shape ${shape}, ${item.tensor.dtype}${item.pending ? `, not computed yet at this step` : ""}${item.fresh ? ", just written" : ""}${broken ? `, ${broken} NaN or infinite values` : ""}${range ? `, values from ${range}` : ""}${originFull ? `, axes from ${originFull}` : ""}${shared ? `, shares storage with ${shared}` : ""}`}
        title={`${item.name} · ${shape}${item.line ? ` · line ${item.line}` : ""}`}
        onClick={() => onSelect(item.nodeId)}
        onKeyDown={(event) => {
          // P pins or unpins the focused card, as its pin button does.
          if (
            event.key.toLowerCase() === "p" &&
            !event.ctrlKey &&
            !event.metaKey &&
            !event.altKey
          ) {
            event.preventDefault();
            onTogglePin(item.name);
          }
        }}
        onMouseEnter={() => onThread(item.history.map((state) => state.nodeId))}
        onMouseLeave={() => onThread(null)}
        onFocus={() => onThread(item.history.map((state) => state.nodeId))}
        onBlur={() => onThread(null)}
      >
        <span className="tensor-shelf-glyph" aria-hidden="true">
          {item.pending ? (
            <ShapeGlyph shape={item.tensor.shape} />
          ) : (
            <PlaneThumb tensor={item.tensor} />
          )}
        </span>
        <span className="tensor-shelf-identity">
          <span className="tensor-shelf-name">{item.name}</span>
          <code className="tensor-shelf-shape">
            <InkShape
              shape={item.tensor.shape}
              ink={item.pending ? null : inkFor?.(item.tensor)}
            />
          </code>
          <span className="tensor-shelf-type">
            {item.tensor.dtype}
            {change && <ChangeNote change={change} />}
            {broken > 0 && (
              <span
                className="tensor-shelf-broken"
                title={`${broken.toLocaleString()} NaN or infinite values`}
              >
                {broken.toLocaleString()} NaN/∞
              </span>
            )}
            {!item.pending &&
              contracts?.byTensor.get(item.tensor.id) &&
              (() => {
                const check = contracts.byTensor.get(item.tensor.id)!;
                return (
                  <span
                    className={
                      check.ok ? "tensor-shelf-contract" : "tensor-shelf-broken"
                    }
                    title={`Line ${check.line}: ${check.text}. ${check.message}`}
                  >
                    {check.ok ? "✓" : "✗"} contract
                  </span>
                );
              })()}
            {bytes !== null && (
              <span title={`${bytes.toLocaleString()} bytes`}>
                {formatBytes(bytes)}
              </span>
            )}
            {item.history.length > 1 && (
              <span
                title={`${item.history.length} recorded states with this name${item.pending ? "" : `; showing state ${item.shown + 1}`}`}
              >
                {item.pending
                  ? `${item.history.length} states`
                  : `state ${item.shown + 1} of ${item.history.length}`}
              </span>
            )}
            {item.history.length > 1 && <StateStrip item={item} />}
          </span>
          {origin && (
            <span
              className="tensor-shelf-lineage"
              title={`Where each axis comes from\n${originDetail}`}
            >
              ← {origin}
            </span>
          )}
        </span>
        {item.pending ? (
          <span className="tensor-shelf-storage">
            <span>
              Not computed yet
              {step ? ` · step ${step.index + 1}` : ""}
            </span>
          </span>
        ) : (
          <span
            className={`tensor-shelf-storage${shared ? " tensor-shelf-shared" : ""}`}
            title={
              shared
                ? `Shares recorded storage ${item.tensor.storage_id} with ${shared}. Shared storage does not necessarily mean the views overlap.`
                : `No other tensor in this shelf shares recorded storage ${item.tensor.storage_id}.`
            }
          >
            {shared ? (
              <>
                <Link2 size={12} aria-hidden="true" />
                <span>Shared with {shared}</span>
              </>
            ) : (
              <span>
                {item.fresh
                  ? "Just written"
                  : item.tensor.contiguous
                    ? "Contiguous"
                    : "Strided"}
                {range && (
                  <code
                    className="tensor-shelf-range"
                    title="Smallest and largest value"
                  >
                    {range}
                  </code>
                )}
              </span>
            )}
          </span>
        )}
      </button>
      <button
        type="button"
        className="tensor-shelf-pin"
        aria-pressed={pinned}
        aria-label={`${pinned ? "Unpin" : "Pin"} ${item.name}`}
        title={
          pinned
            ? "Unpin: return to its place on the shelf"
            : "Pin to the front of the shelf, for this project"
        }
        onClick={() => onTogglePin(item.name)}
      >
        <Pin size={11} aria-hidden="true" />
      </button>
    </li>
  );
});

/**
 * The cards as one keyboard stop: arrows move between them in reading order
 * (up and down by a row), Home and End go to the first and last.
 */
function ShelfGrid({ children }: { children: React.ReactNode }) {
  return (
    <ul
      className="tensor-shelf-grid"
      aria-label="Recorded tensors"
      onKeyDown={(event) => {
        const keys = [
          "ArrowLeft",
          "ArrowRight",
          "ArrowUp",
          "ArrowDown",
          "Home",
          "End",
        ];
        if (!keys.includes(event.key)) return;
        const cards = [
          ...event.currentTarget.querySelectorAll<HTMLButtonElement>(
            ".tensor-shelf-card",
          ),
        ];
        const at = cards.indexOf(document.activeElement as HTMLButtonElement);
        if (at < 0) return;
        const top = cards[0].offsetTop;
        const columns = Math.max(
          1,
          cards.filter((card) => card.offsetTop === top).length,
        );
        const next =
          event.key === "Home"
            ? 0
            : event.key === "End"
              ? cards.length - 1
              : at +
                ({ ArrowLeft: -1, ArrowRight: 1, ArrowUp: -columns }[
                  event.key
                ] ?? columns);
        if (next < 0 || next >= cards.length) return;
        event.preventDefault();
        cards[next].focus();
        cards[next].scrollIntoView({ block: "nearest" });
      }}
    >
      {children}
    </ul>
  );
}

/**
 * A watch on one name across the run: a bar per recorded state, spanning its
 * value range on a shared scale. The state at the playback position is lit,
 * earlier ones are quiet, and later ones are not computed yet.
 */
function StateStrip({ item }: { item: Variable }) {
  const ranges = item.history.map(({ tensor }) =>
    typeof tensor.minimum === "number" &&
    typeof tensor.maximum === "number" &&
    Number.isFinite(tensor.minimum) &&
    Number.isFinite(tensor.maximum)
      ? { min: tensor.minimum, max: tensor.maximum }
      : null,
  );
  if (ranges.length > 48 || ranges.some((range) => !range)) return null;
  const known = ranges as { min: number; max: number }[];
  const low = Math.min(...known.map((range) => range.min));
  const high = Math.max(...known.map((range) => range.max));
  const span = high - low || 1;
  const step = 6,
    height = 16;
  const y = (value: number) => 1.5 + (1 - (value - low) / span) * (height - 3);
  return (
    <svg
      className="tensor-shelf-history"
      width={known.length * step + 2}
      height={height}
      role="img"
      aria-label={`${item.name} across the run: ${known
        .map(
          (range, i) =>
            `state ${i + 1} from ${formatValue(range.min)} to ${formatValue(range.max)}${i === item.shown ? " (shown)" : i > item.shown ? " (not computed yet)" : ""}`,
        )
        .join("; ")}`}
    >
      {known.map((range, i) => (
        <line
          key={i}
          className={
            i === item.shown ? "shown" : i > item.shown ? "later" : undefined
          }
          x1={3 + i * step}
          x2={3 + i * step}
          y1={y(range.max)}
          y2={Math.max(y(range.min), y(range.max) + 1.5)}
        />
      ))}
    </svg>
  );
}

/**
 * A card's picture of its tensor: the first plane averaged down to a few
 * cells and shaded as the grids shade it, with a layer behind for more axes.
 * A tensor whose values stay with the backend keeps the shape's outline.
 */
function PlaneThumb({ tensor }: { tensor: Tensor }) {
  const thumb = useMemo(() => planeThumbnail(tensor), [tensor]);
  if (!thumb) return <ShapeGlyph shape={tensor.shape} />;
  const finite = thumb.picture.filter(Number.isFinite);
  const range = heatRangeOf(tensor) ?? {
    low: Math.min(...finite),
    high: Math.max(...finite),
  };
  const layered = tensor.shape.length > 2 ? 3 : 0;
  const cell = Math.min(
    (34 - layered) / thumb.columns,
    (28 - layered) / thumb.rows,
  );
  const width = cell * thumb.columns,
    height = cell * thumb.rows;
  const left = (34 - layered - width) / 2,
    top = layered + (28 - layered - height) / 2;
  return (
    <svg className="tensor-thumb" viewBox="0 0 34 28" width="34" height="28">
      {layered > 0 && (
        <rect
          className="tensor-thumb-layer"
          x={left + layered}
          y={top - layered}
          width={width}
          height={height}
          rx="1.5"
        />
      )}
      {thumb.picture.map((value, i) => {
        const level = heatLevel(value, range.low, range.high);
        return (
          <rect
            key={i}
            x={left + (i % thumb.columns) * cell}
            y={top + Math.floor(i / thumb.columns) * cell}
            width={cell}
            height={cell}
            style={{
              fill: level === null ? "var(--danger, #e5484d)" : heatFill(level),
            }}
          />
        );
      })}
      <rect
        className="tensor-thumb-frame"
        x={left}
        y={top}
        width={width}
        height={height}
        rx="1.5"
      />
    </svg>
  );
}

/** A card's change since the run before, in a few characters. */
function ChangeNote({ change }: { change: StateChange }) {
  if (change.kind === "changed")
    return (
      <span
        className="tensor-shelf-change"
        title={`${change.changed.toLocaleString()} of ${change.compared.toLocaleString()} values changed since the run before`}
      >
        Δ {change.changed.toLocaleString()}/{change.compared.toLocaleString()}
      </span>
    );
  if (change.kind === "same")
    return (
      <span
        className="tensor-shelf-unchanged"
        title="Every value is the same as in the run before"
      >
        unchanged
      </span>
    );
  return (
    <span
      className="tensor-shelf-unchanged"
      title={
        change.kind === "new"
          ? "No matching state in the run before: the step, shape or dtype differs"
          : "The run before did not record every value, so this was not compared"
      }
    >
      {change.kind === "new" ? "new" : "not compared"}
    </span>
  );
}
