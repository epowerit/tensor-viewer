import { useContext, useSyncExternalStore } from "react";
import type { Tensor } from "../api/client";
import { KnockoutControls, knockedStep } from "./KnockoutControls";
import { SensitivityMap, asksSensitivity } from "./SensitivityMap";
import { TensorUseContext } from "./TensorUseContext";
import { LearnControls, WhatIfContext, canTrain } from "./WhatIf";

export type Experiment = "inputs" | "train" | "knockout";

const KEY = "tensorviewer.cell-experiment";
const listeners = new Set<() => void>();

function stored(): Experiment | null {
  try {
    const value = localStorage.getItem(KEY);
    return value === "inputs" || value === "train" || value === "knockout"
      ? value
      : null;
  } catch {
    return null;
  }
}

let open: Experiment | null = stored();

/** Opens one experiment for every card, or closes them all. */
export function openExperiment(next: Experiment | null) {
  open = next;
  try {
    if (next) localStorage.setItem(KEY, next);
    else localStorage.removeItem(KEY);
  } catch {
    // Without storage the choice lasts until the page reloads.
  }
  listeners.forEach((listener) => listener());
}

function useOpenExperiment() {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => open,
    // Rendered on a server, every experiment starts closed.
    () => null,
  );
}

/**
 * What can be asked of a selected cell, one at a time: which input cells move
 * it, what training on it changes, and what knocking out its step changes.
 * The open experiment stays open from card to card; choosing it again closes
 * it.
 */
export function CellExperiments({
  tensor,
  index,
  coords,
}: {
  tensor: Tensor;
  index: number;
  coords: number[];
}) {
  const flow = useContext(TensorUseContext);
  const control = useContext(WhatIfContext);
  const chosen = useOpenExperiment();
  const tabs: { id: Experiment; label: string; title: string }[] = [];
  if (asksSensitivity(flow, tensor))
    tabs.push({
      id: "inputs",
      label: "What moves it",
      title:
        "Which input cells move this value, and the gradient at every step",
    });
  if (canTrain(control, tensor))
    tabs.push({
      id: "train",
      label: "Train on it",
      title: "Training steps on every weight toward this value",
    });
  if (control && knockedStep(flow, tensor))
    tabs.push({
      id: "knockout",
      label: "Knock out",
      title:
        "Replace this step's result as it is made, or sweep its slices, and see what changes",
    });
  if (!tabs.length) return null;
  const active = tabs.some((tab) => tab.id === chosen) ? chosen : null;
  return (
    <section className="cell-experiments" aria-label="Experiments on this cell">
      <div className="cell-experiments-tabs" role="tablist">
        <span className="cell-experiments-title">Ask</span>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={active === tab.id}
            title={tab.title}
            onClick={() => openExperiment(active === tab.id ? null : tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {active === "inputs" && <SensitivityMap tensor={tensor} index={index} />}
      {active === "train" && <LearnControls tensor={tensor} index={index} />}
      {active === "knockout" && (
        <KnockoutControls tensor={tensor} coords={coords} />
      )}
    </section>
  );
}
