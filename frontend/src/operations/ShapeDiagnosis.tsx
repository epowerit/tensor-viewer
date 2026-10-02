import { Lightbulb } from "lucide-react";
import type { Diagnosis } from "./diagnosis";

/** Operand shapes of a failed operation, with the conflicting axes marked. */
export function ShapeDiagnosis({
  diagnosis,
  bare = false,
}: {
  diagnosis: Diagnosis;
  /** Omit the title when the surrounding list already shows it. */
  bare?: boolean;
}) {
  return (
    <div className="shape-diagnosis">
      {!bare && <h3>{diagnosis.title}</h3>}
      {diagnosis.operands.length > 0 && (
        <div className="diagnosis-operands">
          {diagnosis.operands.map((operand, i) => (
            <div key={i}>
              <span>
                {operand.label} <code>{operand.name}</code>
              </span>
              <div
                className="diagnosis-shape"
                aria-label={`${operand.label} shape ${operand.shape.join(" by ")}${operand.marks.length ? `, mismatch at axis ${operand.marks.join(" and ")}` : ""}`}
              >
                {operand.shape.length ? (
                  operand.shape.map((size, axis) => (
                    <b
                      key={axis}
                      className={operand.marks.includes(axis) ? "conflict" : ""}
                    >
                      {size}
                    </b>
                  ))
                ) : (
                  <i>scalar</i>
                )}
                {operand.dtype && <em>{operand.dtype}</em>}
              </div>
            </div>
          ))}
        </div>
      )}
      <p>{diagnosis.explanation}</p>
      {diagnosis.suggestion && (
        <p className="diagnosis-suggestion">
          <Lightbulb size={13} /> {diagnosis.suggestion}
        </p>
      )}
    </div>
  );
}
