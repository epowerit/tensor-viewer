import { useEffect, useId, useRef, useState } from "react";
import { Ellipsis, RotateCcw } from "lucide-react";

type Props = {
  active: boolean;
  disabled: boolean;
  reveal: boolean;
  onReveal: (value: boolean) => void;
  onRestart: () => void;
};

/** Occasional playback actions stay out of the everyday step controls. */
export function PlaybackOptions({
  active,
  disabled,
  reveal,
  onReveal,
  onRestart,
}: Props) {
  const [open, setOpen] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const id = useId();
  useEffect(() => {
    if (!active || disabled) setOpen(false);
  }, [active, disabled]);
  useEffect(() => {
    if (!open) return;
    const outside = (event: PointerEvent) => {
      if (!container.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [open]);
  return (
    <div
      className="playback-options"
      ref={container}
      onKeyDown={(event) => {
        if (event.key === "Escape" && open) {
          event.stopPropagation();
          setOpen(false);
          trigger.current?.focus();
        }
      }}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setOpen(false);
      }}
    >
      <button
        ref={trigger}
        className="playback-options-trigger"
        aria-label="Playback options"
        title={
          reveal ? "Playback options · Reveal steps is on" : "Playback options"
        }
        aria-expanded={open}
        aria-controls={id}
        disabled={disabled}
        onClick={() => setOpen(!open)}
      >
        <Ellipsis size={18} />
        {reveal && (
          <span className="playback-option-indicator" aria-hidden="true" />
        )}
      </button>
      {open && (
        <section
          id={id}
          className="playback-options-panel"
          aria-label="Playback options"
        >
          <label className="playback-reveal">
            <input
              type="checkbox"
              aria-label="Reveal steps"
              aria-describedby={`${id}-reveal-description`}
              checked={reveal}
              onChange={(event) => onReveal(event.target.checked)}
            />
            <span>
              <b>Reveal steps</b>
              <small id={`${id}-reveal-description`}>
                Show the diagram gradually during playback.
              </small>
            </span>
          </label>
          <button
            onClick={() => {
              setOpen(false);
              trigger.current?.focus();
              onRestart();
            }}
          >
            <RotateCcw size={14} /> Start from the first step
          </button>
        </section>
      )}
    </div>
  );
}
