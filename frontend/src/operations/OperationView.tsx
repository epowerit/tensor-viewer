import { useState } from "react";
import {
  ArrowRight,
  ArrowUpRight,
  Lightbulb,
  MousePointer2,
  Sparkles,
  Route,
} from "lucide-react";
import type { Operation, Run } from "../api/client";
import { ValuesToggle } from "../tensors/ValuesToggle";
import { TensorCard } from "../tensors/TensorCard";
import { outputIndices } from "../tensors/relationships";
import { unravel } from "../tensors/coordinates";
import { presenters } from "./presenters";
import { findPatchJourney } from "./patches";
import { PatchEmbeddingView } from "./PatchEmbeddingView";
import { linearProjection } from "./linear";
import { LinearProjectionView } from "./LinearProjectionView";
import { layoutTransition } from "./layoutTransition";
import { LayoutTransitionView } from "./LayoutTransitionView";
import { MutationView } from "./MutationView";
import { tensorProducer } from "../tensors/provenance";
import { tensorAddition } from "./addition";
import { AdditionView } from "./AdditionView";

type Props = {
  operation: Operation;
  run: Run;
  onJump: (index: number) => void;
  showValues: boolean;
  onShowValues: (show: boolean) => void;
  compact?: boolean;
  expanded?: boolean;
};

export function OperationView(props: Props) {
  const [patchView, setPatchView] = useState(true);
  const [linearView, setLinearView] = useState(true);
  const [mutationView, setMutationView] = useState(true);
  const [additionView, setAdditionView] = useState(true);
  const journey = findPatchJourney(props.run, props.operation);
  const projection = linearProjection(props.run, props.operation);
  const addition = tensorAddition(props.run, props.operation);
  if (props.operation.mutations?.length && mutationView)
    return <MutationView {...props} onDetails={() => setMutationView(false)} />;
  return (
    <>
      {!!props.operation.mutations?.length && (
        <div className="patch-return">
          <button
            className="secondary-button"
            onClick={() => setMutationView(true)}
          >
            <ArrowRight size={14} /> In-place changes
          </button>
        </div>
      )}
      {journey && (
        <div hidden={!patchView}>
          <PatchEmbeddingView
            journey={journey}
            run={props.run}
            operationId={props.operation.id}
            showValues={props.showValues}
            onShowValues={props.onShowValues}
            onDetails={() => setPatchView(false)}
          />
        </div>
      )}
      {journey && !patchView && (
        <div className="patch-return">
          <button
            className="secondary-button"
            onClick={() => setPatchView(true)}
          >
            <ArrowRight size={14} /> Patch to token lesson
          </button>
        </div>
      )}
      {projection && (
        <div hidden={!linearView}>
          <LinearProjectionView
            key={props.operation.id}
            projection={projection}
            run={props.run}
            showValues={props.showValues}
            onShowValues={props.onShowValues}
            onDetails={() => setLinearView(false)}
          />
        </div>
      )}
      {projection && !linearView && (
        <div className="patch-return">
          <button
            className="secondary-button"
            onClick={() => setLinearView(true)}
          >
            <ArrowRight size={14} /> Linear projection lesson
          </button>
        </div>
      )}
      {addition && (
        <div hidden={!additionView}>
          <AdditionView
            key={props.operation.id}
            addition={addition}
            run={props.run}
            showValues={props.showValues}
            onShowValues={props.onShowValues}
            onDetails={() => setAdditionView(false)}
          />
        </div>
      )}
      {addition && !additionView && (
        <div className="patch-return">
          <button
            className="secondary-button"
            onClick={() => setAdditionView(true)}
          >
            <ArrowRight size={14} /> Addition lesson
          </button>
        </div>
      )}
      {(!journey || !patchView) &&
        (!projection || !linearView) &&
        (!addition || !additionView) && <TensorOperationView {...props} />}
    </>
  );
}

function TensorOperationView({
  operation: op,
  run,
  onJump,
  showValues,
  onShowValues,
  compact = false,
  expanded = false,
}: Props) {
  const [selected, setSelected] = useState(0);
  const [inputChoice, setInputChoice] = useState(0);
  const [outputChoice, setOutputChoice] = useState(0);
  const [inputSelected, setInputSelected] = useState<number | null>(null);
  const [predict, setPredict] = useState(false);
  const [followElement, setFollowElement] = useState(false);
  const [guess, setGuess] = useState("");
  const [feedback, setFeedback] = useState("");
  const inputs = op.inputs.map((id) => run.trace.tensors[id]);
  const output = run.trace.tensors[op.outputs[outputChoice]];
  const isDot = op.lesson.interaction === "dot_product";
  const canMap =
    outputChoice === 0 &&
    inputChoice === 0 &&
    !!(op.lesson.mapping || op.lesson.mapping_rule);
  const matches =
    canMap && inputSelected !== null
      ? outputIndices(op, inputs[0], output, inputSelected)
      : null;
  const presenter =
    presenters[
      outputChoice === 0 && inputChoice === 0
        ? op.lesson.interaction
        : "inspect"
    ] ?? presenters.inspect;
  const presentation =
    output && output.numel > 0 && (!matches || matches.length)
      ? presenter(op, inputs, output, selected)
      : null;
  const first = inputs[inputChoice];
  const transition =
    canMap && (!matches || matches.length)
      ? layoutTransition(op, first, output, selected)
      : null;
  const displayed = [first, ...(isDot ? [inputs[1]] : []), output].filter(
    Boolean,
  );
  // A shared frame keeps cell sizes comparable across the transformation.
  const gridFrame = {
    rows: Math.max(
      1,
      ...displayed.map((tensor) => Math.min(8, tensor.shape.at(-2) ?? 1)),
    ),
    columns: Math.max(
      1,
      ...displayed.map((tensor) => Math.min(8, tensor.shape.at(-1) ?? 1)),
    ),
  };
  const mappedInput = inputSelected ?? presentation?.leftHighlights[0];
  const sourceLines = run.project.code.split("\n");
  const sourceStart = Math.max(0, (op.source?.line ?? 1) - 3);
  const dependencies = op.inputs.map((id) => ({
    tensor: run.trace.tensors[id],
    producer: tensorProducer(
      run.trace.operations,
      id,
      run.trace.operations.indexOf(op),
    ),
  }));

  return (
    <div className={`operation-view ${expanded ? "operation-expanded" : ""}`}>
      {!compact && (
        <div className="scene-heading">
          <div>
            <span className={`category category-${op.lesson.category}`}>
              {op.lesson.category === "layout"
                ? "Shape & arrangement"
                : op.lesson.category === "compute"
                  ? "Computation"
                  : op.lesson.category === "memory"
                    ? "Memory layout"
                    : op.lesson.category === "normalize"
                      ? "Normalization"
                      : "Tensor operation"}
            </span>
            <h2>{op.lesson.title}</h2>
            <p>{op.lesson.summary}</p>
          </div>
        </div>
      )}
      {!compact && op.source && (
        <details className="source-panel">
          <summary>
            <span className="source-location">Line {op.source.line}</span>
            <code>{sourceLines[op.source.line - 1]?.trim() || op.kind}</code>
            <span className="source-expand">Code context</span>
          </summary>
          <pre>
            {sourceLines.slice(sourceStart, sourceStart + 5).map((line, i) => (
              <div
                key={i}
                className={
                  sourceStart + i + 1 === op.source?.line ? "source-active" : ""
                }
              >
                <span>{sourceStart + i + 1}</span>
                <code>{line || " "}</code>
              </div>
            ))}
          </pre>
        </details>
      )}
      <div className="scene" data-testid="operation-scene">
        <div className="scene-meta">
          <ValuesToggle
            checked={showValues}
            onChange={onShowValues}
            shapeOnly={run.project.capture_mode === "shapes"}
          />
          {transition && !predict && (
            <button
              className="text-button"
              aria-pressed={followElement}
              onClick={() => setFollowElement(!followElement)}
            >
              <Route size={14} />{" "}
              {followElement ? "Tensor details" : "Follow element"}
            </button>
          )}
          {output && (
            <button
              className="text-button"
              onClick={() => {
                setPredict(!predict);
                setFollowElement(false);
                setFeedback("");
              }}
            >
              <Sparkles size={13} />
              {predict ? "Show the result" : "Predict the shape"}
            </button>
          )}
        </div>
        {transition && followElement && !predict && (
          <LayoutTransitionView
            mapping={transition}
            runId={run.id}
            showValues={showValues}
            onInputSelect={(index) => {
              setInputSelected(index);
              const next = outputIndices(op, first, output, index);
              if (next.length) setSelected(next[0]);
            }}
            onOutputSelect={(index) => {
              setInputSelected(null);
              setSelected(index);
            }}
          />
        )}
        <div hidden={!!transition && followElement && !predict}>
          <div className={`tensor-flow ${isDot ? "tensor-flow-three" : ""}`}>
            {first ? (
              <div>
                <TensorCard
                  runId={run.id}
                  tensor={first}
                  gridFrame={gridFrame}
                  expandDetails={op.lesson.category === "memory"}
                  label={isDot ? "Left input" : "Before"}
                  showValues={showValues}
                  highlights={
                    inputSelected !== null
                      ? [inputSelected]
                      : presentation?.leftHighlights
                  }
                  focusIndex={mappedInput}
                  onSelect={
                    canMap
                      ? (index) => {
                          setInputSelected(index);
                          const next = outputIndices(
                            op,
                            inputs[0],
                            output,
                            index,
                          );
                          if (next.length) setSelected(next[0]);
                        }
                      : undefined
                  }
                />
                {!isDot && inputs.length > 1 && (
                  <label className="tensor-select">
                    Input
                    <select
                      value={inputChoice}
                      onChange={(e) => {
                        setInputChoice(Number(e.target.value));
                        setInputSelected(null);
                      }}
                    >
                      {inputs.map((t, i) => (
                        <option key={`${t.id}-${i}`} value={i}>
                          {t.name} · {t.role} [{t.shape.join(", ")}]
                        </option>
                      ))}
                    </select>
                  </label>
                )}
              </div>
            ) : (
              <div className="factory-input">
                <Sparkles size={26} />
                <p>Tensor creation</p>
                <code>{op.kind}</code>
              </div>
            )}
            {isDot && inputs[1] && (
              <>
                <div className="flow-symbol">@</div>
                <TensorCard
                  runId={run.id}
                  tensor={inputs[1]}
                  gridFrame={gridFrame}
                  expandDetails={op.lesson.category === "memory"}
                  label="Right input"
                  showValues={showValues}
                  highlights={presentation?.rightHighlights}
                  focusIndex={presentation?.rightHighlights[0]}
                />
              </>
            )}
            <div className="flow-symbol">
              {expanded && <span className="flow-operation">{op.kind}</span>}
              <ArrowRight size={23} />
            </div>
            {predict && output ? (
              <div className="prediction">
                <Lightbulb size={30} />
                <h3>What shape comes next?</h3>
                <p>
                  Follow the operation’s arguments, then predict the output
                  dimensions.
                </p>
                <input
                  aria-label="Predicted output shape"
                  placeholder="e.g. 1, 2, 3, 4"
                  value={guess}
                  onChange={(e) => setGuess(e.target.value)}
                />
                <button
                  className="primary-button"
                  onClick={() => {
                    const dims = guess
                      .replace(/[\[\]()]/g, "")
                      .split(/[,x×\s]+/)
                      .filter(Boolean)
                      .map(Number);
                    if (JSON.stringify(dims) === JSON.stringify(output.shape)) {
                      setFeedback(
                        "Correct. Now follow an element through the transformation.",
                      );
                      setPredict(false);
                    } else
                      setFeedback(
                        "Not quite. Check the dimension order and try again.",
                      );
                  }}
                >
                  Check prediction
                </button>
              </div>
            ) : output ? (
              <div>
                <TensorCard
                  runId={run.id}
                  tensor={output}
                  gridFrame={gridFrame}
                  expandDetails={op.lesson.category === "memory"}
                  label={isDot ? "Output" : "After"}
                  tone="output"
                  showValues={showValues}
                  highlights={matches ?? [selected]}
                  focusIndex={selected}
                  onSelect={(index) => {
                    setInputSelected(null);
                    setSelected(index);
                  }}
                />
                {op.outputs.length > 1 && (
                  <label className="tensor-select">
                    Output
                    <select
                      value={outputChoice}
                      onChange={(e) => {
                        setOutputChoice(Number(e.target.value));
                        setSelected(0);
                        setInputSelected(null);
                      }}
                    >
                      {op.outputs.map((id, i) => (
                        <option key={`${id}-${i}`} value={i}>
                          {run.trace.tensors[id].name}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
              </div>
            ) : (
              <div className="operation-error">
                <h3>
                  {op.error
                    ? "This operation stopped the run"
                    : "No tensor returned"}
                </h3>
                <p>
                  {op.error ||
                    "Inspect In-place changes for the recorded side effects."}
                </p>
                <span>The input tensors are preserved for inspection.</span>
              </div>
            )}
          </div>
          <div className="scene-caption">
            <MousePointer2 size={15} />
            <span>
              {predict
                ? "Enter the dimensions, then check your prediction."
                : canMap
                  ? "Select either tensor to follow the same value."
                  : "Select any cell to inspect it. Select a result to see supported relationships."}
            </span>
            <small>
              {run.project.capture_mode === "shapes"
                ? "Shape preview · values not computed"
                : "Exact values on selection · rounded cell labels"}{" "}
              · 0-based indices
            </small>
          </div>
        </div>
      </div>
      {feedback && (
        <div className="prediction-feedback" role="status">
          {feedback}
        </div>
      )}
      {presentation && !predict && !(transition && followElement) && (
        <div className="element-insight">
          <div className="insight-icon">
            <MousePointer2 size={17} />
          </div>
          <div>
            <h3>{presentation.title}</h3>
            <p>{presentation.text}</p>
            {matches && matches.length > 1 && (
              <p className="mapping-note">
                Highlighting {matches.length}
                {matches.length === 256 ? " or more" : ""} matching output
                positions; some may be on another slice or page.
              </p>
            )}
            {presentation.expression && (
              <code className="calculation">{presentation.expression}</code>
            )}
          </div>
        </div>
      )}
      {matches?.length === 0 && first && !predict && (
        <div className="element-insight">
          <div>
            <h3>No output uses this element</h3>
            <p>
              Input [{unravel(inputSelected!, first.shape).join(", ")}] is not
              included in this operation’s recorded mapping.
            </p>
          </div>
        </div>
      )}
      <details className="explanation-details">
        <summary>How this operation works</summary>
        <section className="explanation">
          <p>{op.lesson.detail}</p>
          <div className="dependencies">
            <span>Follow an input back</span>
            {dependencies.map(({ tensor, producer }, i) =>
              producer ? (
                <button key={i} onClick={() => onJump(producer.index)}>
                  {tensor.name}
                  <small>step {producer.index + 1}</small>
                  <ArrowUpRight size={12} />
                </button>
              ) : (
                <span className="dependency-origin" key={i}>
                  {tensor.name} · {tensor.role}
                </span>
              ),
            )}
          </div>
        </section>
      </details>
    </div>
  );
}
