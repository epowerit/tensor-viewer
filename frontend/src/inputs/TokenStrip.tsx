import { tokenize } from "./samples";

/** Each token of a sentence with the integer id the model receives. */
export function TokenStrip({ text }: { text: string }) {
  const { tokens, ids, vocabulary } = tokenize(text);
  if (!tokens.length) return null;
  return (
    <span className="token-strip">
      {tokens.map((token, i) => (
        <span key={i} title={`Position ${i}: id ${ids[i]}`}>
          {token}
          <b>{ids[i]}</b>
        </span>
      ))}
      <small>
        {vocabulary.length} distinct{" "}
        {vocabulary.length === 1 ? "token" : "tokens"}: ids 0–
        {vocabulary.length - 1}
      </small>
    </span>
  );
}
