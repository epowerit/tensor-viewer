import type { DetailLevel } from "./stages";

/**
 * The diagram's detail dial: how finely the model is cut, from the whole model
 * as one card to every recorded step. Each setting's glyph is the model cut
 * into more pieces; its name is what the cards show at that setting (the
 * model, its blocks, their attention and feed-forward calls, operations, every
 * step). ← and → move along it.
 */
export function DetailDial({
  levels,
  current,
  disabled,
  cards,
  onChoose,
}: {
  levels: DetailLevel[];
  /** The level the folded stages match, or -1 when folded by hand. */
  current: number;
  disabled: boolean;
  /** How many cards each level draws. */
  cards?: number[];
  onChoose: (level: DetailLevel) => void;
}) {
  if (levels.length < 2) return null;
  const pieces = (i: number) =>
    Math.round(1 + (i * 5) / Math.max(1, levels.length - 1));
  return (
    <div
      className="detail-dial"
      role="group"
      aria-label="Diagram detail"
      onKeyDown={(event) => {
        if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
        event.preventDefault();
        const from = current >= 0 ? current : levels.length - 1;
        const next = Math.max(
          0,
          Math.min(
            levels.length - 1,
            from + (event.key === "ArrowRight" ? 1 : -1),
          ),
        );
        if (next === current) return;
        onChoose(levels[next]);
        requestAnimationFrame(() =>
          event.currentTarget
            .querySelectorAll<HTMLButtonElement>("button")
            [next]?.focus(),
        );
      }}
    >
      <span className="detail-dial-name" aria-hidden="true">
        Detail
      </span>
      {levels.map((level, i) => {
        const count = pieces(i);
        const width = (14 - (count - 1)) / count;
        return (
          <button
            key={`${i}-${level.label}`}
            type="button"
            className="detail-dial-step"
            aria-pressed={i === current}
            aria-label={`Detail: ${level.label}`}
            title={`${level.label}${cards?.[i] !== undefined ? ` · ${cards[i]} ${cards[i] === 1 ? "card" : "cards"}` : ""}: ${level.detail}`}
            disabled={disabled}
            onClick={() => onChoose(level)}
          >
            <svg viewBox="0 0 14 12" width="14" height="12" aria-hidden="true">
              {Array.from({ length: count }, (_, piece) => (
                <rect
                  key={piece}
                  x={piece * (width + 1)}
                  y={1}
                  width={width}
                  height={10}
                  rx={Math.min(1.5, width / 2)}
                />
              ))}
            </svg>
          </button>
        );
      })}
      <span className="detail-dial-label" aria-live="polite">
        {current >= 0 ? levels[current].label : "as folded"}
      </span>
    </div>
  );
}
