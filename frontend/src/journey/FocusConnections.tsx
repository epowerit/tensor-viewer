import { useId, useMemo, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  GitBranch,
} from "lucide-react";
import type { Run } from "../api/client";
import type { JourneyGraph, JourneyNode } from "./graph";
import {
  journeyConnections,
  returnedTensorIds,
  type JourneyConnection,
} from "./connections";

type Props = {
  run: Run;
  graph: JourneyGraph;
  node: JourneyNode;
  open: boolean;
  onOpen: (open: boolean) => void;
  onSelect: (id: string) => void;
};

/** The same recorded dataflow navigation for inputs, lessons, and module stages. */
export function FocusConnections({
  run,
  graph,
  node,
  open,
  onOpen,
  onSelect,
}: Props) {
  const connections = useMemo(
    () => journeyConnections(graph, node.id, run.trace),
    [graph, node.id, run.trace],
  );
  const sources = new Set(connections.incoming.map((item) => item.nodeId)).size;
  const uses = connections.outgoing.length;
  const returned = useMemo(
    () =>
      returnedTensorIds(graph, node.id, run.trace)
        .map((id) => run.trace.tensors[id])
        .filter(Boolean),
    [graph, node.id, run.trace],
  );
  const stopped = node.operation?.status === "error" || node.stage?.failed;
  const usesParameters = node.stage
    ? run.trace.operations
        .slice(node.stage.start_index, node.stage.end_index)
        .some((operation) =>
          operation.inputs.some(
            (id) => run.trace.tensors[id]?.role === "parameter",
          ),
        )
    : node.parameterCount > 0;

  return (
    <details
      className="focus-connections"
      open={open}
      onToggle={(event) => onOpen(event.currentTarget.open)}
    >
      <summary>
        <GitBranch size={14} aria-hidden="true" />
        <span>Follow connections</span>
        <small>
          {sources} {sources === 1 ? "source" : "sources"} · {uses}{" "}
          {uses === 1 ? "use" : "uses"}
        </small>
        <ChevronDown
          className="connections-chevron"
          size={14}
          aria-hidden="true"
        />
      </summary>
      {open && (
        <div className="connections-content">
          <div className="connections-columns">
            <ConnectionList
              direction="incoming"
              items={connections.incoming}
              graph={graph}
              run={run}
              onSelect={onSelect}
              empty={
                !node.operation && !node.stage
                  ? run.trace.input_ids.some((id) =>
                      node.tensors.some((tensor) => tensor.id === id),
                    )
                    ? "Passed to forward as an input."
                    : "Captured before its first recorded use."
                  : usesParameters
                    ? "Uses captured parameters; no incoming tensor connection is shown."
                    : "No incoming tensor connection in this diagram."
              }
            />
            <ConnectionList
              direction="outgoing"
              items={connections.outgoing}
              graph={graph}
              run={run}
              onSelect={onSelect}
              empty={
                stopped
                  ? "Execution stopped here."
                  : "No later use was recorded."
              }
            />
          </div>
          {!!returned.length && (
            <p className="connections-returned">
              Returned by the model:{" "}
              {returned.map((tensor) => tensor.name).join(", ")}
            </p>
          )}
          <p className="connections-note">
            Follow recorded tensor connections. Playback follows execution
            order.
            {usesParameters &&
              (node.stage
                ? " Open an individual step to inspect its weights."
                : " Weights are available in Tensor details.")}
          </p>
        </div>
      )}
    </details>
  );
}

const PAGE_SIZE = 6;

function ConnectionList({
  direction,
  items,
  graph,
  run,
  onSelect,
  empty,
}: {
  direction: "incoming" | "outgoing";
  items: JourneyConnection[];
  graph: JourneyGraph;
  run: Run;
  onSelect: (id: string) => void;
  empty: string;
}) {
  const [page, setPage] = useState(0);
  const titleId = useId();
  const start = Math.min(
    page * PAGE_SIZE,
    Math.max(0, Math.ceil(items.length / PAGE_SIZE) - 1) * PAGE_SIZE,
  );
  const end = Math.min(items.length, start + PAGE_SIZE);
  const title = direction === "incoming" ? "Comes from" : "Used by";
  const pageLabel = direction === "incoming" ? "sources" : "uses";
  const Arrow = direction === "incoming" ? ArrowLeft : ArrowRight;
  return (
    <nav className="connections-group" aria-labelledby={titleId}>
      <h3 id={titleId}>{title}</h3>
      {items.length ? (
        <ul>
          {items.slice(start, end).map((item) => {
            const neighbor = graph.nodes.find(
              (entry) => entry.id === item.nodeId,
            )!;
            const tensor = run.trace.tensors[item.tensorId];
            const destination = neighbor.stage
              ? `${neighbor.stage.title} · steps ${neighbor.stage.start_index + 1}–${neighbor.stage.end_index}`
              : neighbor.operation
                ? `${neighbor.operation.kind} · step ${neighbor.operation.index + 1}`
                : tensor?.role === "input"
                  ? "Input tensor"
                  : "Captured tensor";
            const shape = tensor ? `[${tensor.shape.join(", ")}]` : "";
            const name = tensor?.name ?? "Tensor";
            return (
              <li key={item.key}>
                <button
                  onClick={() => onSelect(item.nodeId)}
                  aria-label={`${title}: ${name} ${shape}, ${destination}, ${item.operandRole}`}
                  title={`${name} ${shape} · ${destination} · ${item.operandRole}`}
                >
                  <Arrow size={14} aria-hidden="true" />
                  <span className="connection-label">
                    <span>
                      <b>{name}</b>
                      <code>{shape}</code>
                    </span>
                    <small>{destination}</small>
                  </span>
                  <span
                    className={`connection-role ${item.kind === "storage" ? "is-storage" : ""}`}
                  >
                    {item.operandRole}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="connections-empty">{empty}</p>
      )}
      {items.length > PAGE_SIZE && (
        <div className="connections-pages">
          <span>
            {start + 1}–{end} of {items.length}
          </span>
          <button
            aria-label={`Previous ${pageLabel}`}
            disabled={start === 0}
            onClick={() => setPage(Math.max(0, start / PAGE_SIZE - 1))}
          >
            <ChevronLeft size={14} />
          </button>
          <button
            aria-label={`Next ${pageLabel}`}
            disabled={end === items.length}
            onClick={() => setPage(start / PAGE_SIZE + 1)}
          >
            <ChevronRight size={14} />
          </button>
        </div>
      )}
    </nav>
  );
}
