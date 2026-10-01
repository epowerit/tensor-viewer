import { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Layers3, X } from "lucide-react";
import type { JourneyStage } from "./stages";

type Props = {
  stages: JourneyStage[];
  collapsed: Set<string>;
  disabled: boolean;
  onToggle: (id: string) => void;
  onOverview: () => void;
  onExpandAll: () => void;
};

export function StageControls({
  stages,
  collapsed,
  disabled,
  onToggle,
  onOverview,
  onExpandAll,
}: Props) {
  const [open, setOpen] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (!container.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [open]);
  if (!stages.length) return null;
  function rows(parent: string | null, depth = 0): React.ReactNode {
    return stages
      .filter((stage) => stage.parentStageId === parent)
      .map((stage) => (
        <div key={stage.id}>
          <button
            className="stage-tree-row"
            style={{ paddingLeft: 12 + depth * 14 }}
            aria-expanded={!collapsed.has(stage.id)}
            aria-label={`${collapsed.has(stage.id) ? "Expand" : "Collapse"} stage ${stage.title}, ${stage.path}, ${stage.id}`}
            disabled={disabled}
            onClick={(event) => {
              const button = event.currentTarget;
              onToggle(stage.id);
              requestAnimationFrame(() =>
                button.focus({ preventScroll: true }),
              );
            }}
          >
            {collapsed.has(stage.id) ? (
              <ChevronRight size={14} />
            ) : (
              <ChevronDown size={14} />
            )}
            <span>
              <b>{stage.title}</b>
              <small>
                {stage.path} · steps {stage.start_index + 1}–{stage.end_index}
              </small>
            </span>
            <em>{stage.operationIds.length}</em>
          </button>
          {!collapsed.has(stage.id) && rows(stage.id, depth + 1)}
        </div>
      ));
  }
  return (
    <div
      className="stage-controls"
      ref={container}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.stopPropagation();
          setOpen(false);
          trigger.current?.focus();
        }
      }}
    >
      <button
        ref={trigger}
        className="stage-menu-trigger"
        aria-label="Recorded stages"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
      >
        <Layers3 size={14} /> Stages <ChevronDown size={12} />
      </button>
      {open && (
        <section className="stage-menu" aria-label="Recorded module stages">
          <header>
            <div>
              <b>Recorded stages</b>
              <small>Expand a call to see its operations.</small>
            </div>
            <button
              className="icon-button"
              aria-label="Close stages"
              onClick={() => {
                setOpen(false);
                trigger.current?.focus();
              }}
            >
              <X size={15} />
            </button>
          </header>
          <div
            className="stage-view-actions"
            role="group"
            aria-label="Diagram detail"
          >
            <button
              disabled={disabled}
              onClick={() => {
                onOverview();
                setOpen(false);
                trigger.current?.focus();
              }}
            >
              Group into stages
            </button>
            <button
              disabled={disabled}
              onClick={() => {
                onExpandAll();
                setOpen(false);
                trigger.current?.focus();
              }}
            >
              Show every operation
            </button>
          </div>
          {disabled && (
            <p>Turn off Reveal steps in playback options to group stages.</p>
          )}
          <div className="stage-tree">{rows(null)}</div>
          <footer>
            Groups follow actual module calls. No operations are removed.
          </footer>
        </section>
      )}
    </div>
  );
}
