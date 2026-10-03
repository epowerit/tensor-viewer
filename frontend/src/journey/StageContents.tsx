import type { Operation, Tensor } from "../api/client";
import { kindName } from "../operations/kindName";
import type { JourneyStage } from "./stages";

/** At most this many steps are listed; the rest are counted. */
const SHOWN = 14;

const shape = (tensor: Tensor | undefined) =>
  tensor ? `[${tensor.shape.join(", ")}]` : "";

/**
 * What a folded card holds, without unfolding it: its signature, what goes
 * in and what comes out, and each step inside with the tensor it makes.
 */
export function StageContents({
  stage,
  operations,
  tensors,
}: {
  stage: JourneyStage;
  operations: Operation[];
  tensors: Record<string, Tensor>;
}) {
  const byId = new Map(operations.map((op) => [op.id, op]));
  const steps = stage.operationIds
    .map((id) => byId.get(id))
    .filter((op): op is Operation => !!op);
  const input = tensors[stage.inputs?.[0] ?? ""];
  const output = tensors[stage.layout?.via ?? stage.outputs?.[0] ?? ""];
  return (
    <div className="stage-contents" role="tooltip">
      <header>
        <b>{stage.title}</b>
        <span>
          {steps.length} {steps.length === 1 ? "step" : "steps"}
        </span>
      </header>
      {input && output && (
        <p className="stage-contents-signature">
          {input.name} {shape(input)} → {shape(output)}
          {stage.layout && <em> · values unchanged</em>}
        </p>
      )}
      <ol>
        {steps.slice(0, SHOWN).map((op) => {
          const result = tensors[op.outputs[0]];
          return (
            <li key={op.id} className={op.status === "error" ? "failed" : ""}>
              <span className="stage-contents-line">
                {op.source?.line ?? ""}
              </span>
              <span className="stage-contents-step">
                {kindName(op.kind)}
                {result && result.name !== op.kind && <b> {result.name}</b>}
              </span>
              <code>{shape(result)}</code>
            </li>
          );
        })}
      </ol>
      {steps.length > SHOWN && (
        <p className="stage-contents-more">
          and {steps.length - SHOWN} more · double-click the card to unfold it
        </p>
      )}
    </div>
  );
}
