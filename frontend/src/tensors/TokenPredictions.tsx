import type { Tensor } from "../api/client";
import { ravel } from "./coordinates";

export type Prediction = {
  /** The word at this position of the sentence. */
  word: string;
  /** The vocabulary entries most likely next, with their probability. */
  top: { id: number; word: string; probability: number }[];
  /** The flat index of the top entry's cell. */
  index: number;
};

/**
 * What a model predicts at each position of a sentence, read from a tensor
 * with the sentence's positions and a vocabulary axis: the most likely
 * entries along the vocabulary at each position. Scores that are not
 * already probabilities (logits) go through a softmax first. The other
 * axes stay where the cursor is.
 */
export function predictions(
  tensor: Tensor,
  at: number[],
  axes: {
    positions: number;
    vocabulary: number;
    words: string[];
    ids: string[];
  },
  top = 3,
): Prediction[] | null {
  if (tensor.values.length !== tensor.numel) return null;
  const value = (coordinates: number[]) =>
    Number(tensor.values[ravel(coordinates, tensor.shape)]);
  return axes.words.map((word, position) => {
    const scores = axes.ids.map((_, id) => {
      const coordinates = [...at];
      coordinates[axes.positions] = position;
      coordinates[axes.vocabulary] = id;
      return value(coordinates);
    });
    const probabilities =
      scores.every((s) => s >= 0 && s <= 1) &&
      Math.abs(scores.reduce((a, b) => a + b, 0) - 1) < 1e-3
        ? scores
        : softmax(scores);
    const ranked = probabilities
      .map((probability, id) => ({ id, word: axes.ids[id], probability }))
      .filter((item) => Number.isFinite(item.probability))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, top);
    const coordinates = [...at];
    coordinates[axes.positions] = position;
    coordinates[axes.vocabulary] = ranked[0]?.id ?? 0;
    return { word, top: ranked, index: ravel(coordinates, tensor.shape) };
  });
}

function softmax(scores: number[]) {
  const high = Math.max(...scores.filter(Number.isFinite));
  const raised = scores.map((s) =>
    Number.isFinite(s) ? Math.exp(s - high) : 0,
  );
  const total = raised.reduce((a, b) => a + b, 0) || 1;
  return raised.map((r) => r / total);
}

const percent = (p: number) =>
  p >= 0.995 ? "100%" : p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`;

/** Each word of the sentence with what the model predicts after it. */
export function TokenPredictions({
  items,
  current,
  onPick,
}: {
  items: Prediction[];
  /** The position the cursor is on. */
  current: number;
  onPick: (index: number) => void;
}) {
  return (
    <div className="token-predictions" aria-label="Predicted next words">
      <span className="token-predictions-title">Predicts</span>
      <ol>
        {items.map((item, position) => (
          <li key={position}>
            <button
              type="button"
              aria-current={position === current ? "true" : undefined}
              title={`After “${item.word}”: ${item.top
                .map((entry) => `${entry.word} ${percent(entry.probability)}`)
                .join(", ")}`}
              onClick={() => onPick(item.index)}
            >
              <span>{item.word}</span>
              <b>{item.top[0]?.word ?? "?"}</b>
              <small>
                {item.top[0] ? percent(item.top[0].probability) : ""}
              </small>
            </button>
          </li>
        ))}
      </ol>
    </div>
  );
}
