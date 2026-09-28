import { useState } from "react";
import {
  ArrowRight,
  ArrowUpRight,
  Check,
  Code2,
  Lightbulb,
  MousePointer2,
  Sparkles,
} from "lucide-react";
import type { Operation, Run } from "../api/client";
import { TensorCard } from "../tensors/TensorCard";
import { presenters } from "./presenters";

type Props = {
  operation: Operation;
  run: Run;
  onJump: (index: number) => void;
};

export function OperationView({ operation: op, run, onJump }: Props) {
  const [selected, setSelected] = useState(0);
  const [values, setValues] = useState(false);
  const [inputChoice, setInputChoice] = useState(0);
  const [outputChoice, setOutputChoice] = useState(0);
  const [predict, setPredict] = useState(false);
  const [guess, setGuess] = useState("");
  const [feedback, setFeedback] = useState("");
  const inputs = op.inputs.map((id) => run.trace.tensors[id]);
  const output = run.trace.tensors[op.outputs[outputChoice]];
  const isDot = op.lesson.interaction === "dot_product";
  const presenter =
    presenters[
      outputChoice === 0 && inputChoice === 0
        ? op.lesson.interaction
        : "inspect"
    ] ?? presenters.inspect;
  const presentation =
    output && output.numel > 0 ? presenter(op, inputs, output, selected) : null;
  const first = inputs[inputChoice];
  const mappedInput = presentation?.leftHighlights[0];
  const sourceLines = run.project.code.split("\n");
  const sourceStart = Math.max(0, (op.source?.line ?? 1) - 3);
  const dependencies = op.inputs.map((id) => ({
    tensor: run.trace.tensors[id],
    producer: run.trace.operations.find((p) => p.outputs.includes(id)),
  }));

  return (
    <div className="operation-view">
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
        <div className="segmented" aria-label="Tensor display">
          <button
            className={!values ? "active" : ""}
            onClick={() => setValues(false)}
          >
            Structure
          </button>
          <button
            className={values ? "active" : ""}
            onClick={() => setValues(true)}
          >
            Values
          </button>
        </div>
      </div>
      <div className="scene" data-testid="operation-scene">
        <div className="scene-meta">
          <span>
            <span className="dot" /> Actual execution · CPU
          </span>
          <button
            className="text-button"
            onClick={() => {
              setPredict(!predict);
              setFeedback("");
            }}
          >
            <Sparkles size={13} />
            {predict ? "Show the result" : "Predict the shape"}
          </button>
        </div>
        <div className={`tensor-flow ${isDot ? "tensor-flow-three" : ""}`}>
          {first ? (
            <div>
              <TensorCard
                tensor={first}
                label={isDot ? "Left input" : "Before"}
                showValues={values}
                highlights={presentation?.leftHighlights}
                focusIndex={mappedInput}
                onSelect={
                  op.lesson.mapping && inputChoice === 0
                    ? (index) => {
                        const next = op.lesson.mapping!.indexOf(index);
                        if (next >= 0) setSelected(next);
                      }
                    : undefined
                }
              />
              {!isDot && inputs.length > 1 && (
                <label className="tensor-select">
                  Input
                  <select
                    value={inputChoice}
                    onChange={(e) => setInputChoice(Number(e.target.value))}
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
                tensor={inputs[1]}
                label="Right input"
                showValues={values}
                highlights={presentation?.rightHighlights}
                focusIndex={presentation?.rightHighlights[0]}
              />
            </>
          )}
          <div className="flow-symbol">
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
                tensor={output}
                label={isDot ? "Output" : "After"}
                tone="output"
                showValues={values}
                highlights={[selected]}
                focusIndex={selected}
                onSelect={setSelected}
              />
              {op.outputs.length > 1 && (
                <label className="tensor-select">
                  Output
                  <select
                    value={outputChoice}
                    onChange={(e) => {
                      setOutputChoice(Number(e.target.value));
                      setSelected(0);
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
              <h3>This operation stopped the run</h3>
              <p>{op.error}</p>
              <span>The input tensors are preserved for inspection.</span>
            </div>
          )}
        </div>
        <div className="scene-caption">
          <MousePointer2 size={13} /> Select an output cell to explore its
          relationship to the input.{" "}
          <span>Dimensions drawn schematically.</span>
        </div>
      </div>
      {feedback && (
        <div className="prediction-feedback" role="status">
          {feedback}
        </div>
      )}
      {presentation && !predict && (
        <div className="element-insight">
          <div className="insight-icon">
            <MousePointer2 size={17} />
          </div>
          <div>
            <h3>{presentation.title}</h3>
            <p>{presentation.text}</p>
            {presentation.expression && (
              <code className="calculation">{presentation.expression}</code>
            )}
          </div>
        </div>
      )}
      <div className="operation-bottom">
        <section className="explanation">
          <div className="section-label">
            <Lightbulb size={15} /> What happened
          </div>
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
        <section className="source-panel">
          <div className="section-label">
            <Code2 size={15} /> Source{" "}
            <span>saved run · line {op.source?.line ?? "—"}</span>
          </div>
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
        </section>
      </div>
      <div className="operation-footer">
        <Check size={13} /> Values and shapes recorded from PyTorch{" "}
        <span className="mono">{op.module}</span>
      </div>
    </div>
  );
}
