import { useCallback, useEffect, useId, useRef, useState } from "react";
import {
  Box,
  ChevronDown,
  CircleAlert,
  FileCode2,
  LoaderCircle,
  Plus,
  Trash2,
} from "lucide-react";
import { api, type Draft, type ForwardInput } from "../api/client";
import { entryPath, sourceCode } from "../sources/files";
import {
  forwardCall,
  forwardInputs,
  forwardIssue,
  validInputName,
  withForwardInputs,
} from "./forward";
import { TensorInputFields } from "./TensorInputFields";
import { revealField } from "../components/revealField";

export function ForwardInputs({
  draft,
  busy,
  active,
  onChange,
  onValidity,
  reviewRequest,
}: {
  draft: Draft;
  busy: boolean;
  active: boolean;
  onChange: (draft: Draft) => void;
  onValidity: (valid: boolean) => void;
  reviewRequest: number;
}) {
  const inputs = forwardInputs(draft);
  const keys = useRef(inputs.map(() => crypto.randomUUID()));
  const [selected, setSelected] = useState(0);
  const root = useRef<HTMLElement>(null);
  const handledReview = useRef(0);
  const [validity, setValidity] = useState<Record<string, boolean>>({});
  const recordValidity = useCallback((id: string, valid: boolean) => {
    setValidity((current) =>
      current[id] === valid ? current : { ...current, [id]: valid },
    );
  }, []);
  const issue = forwardIssue(inputs, draft.capture_mode);
  const valid = !issue && keys.current.every((key) => validity[key] !== false);
  useEffect(() => {
    if (!active || !reviewRequest || reviewRequest === handledReview.current)
      return;
    const invalidIndex = inputs.findIndex(
      (_, index) =>
        validity[keys.current[index]] === false ||
        !!forwardIssue(inputs.slice(0, index + 1), draft.capture_mode),
    );
    const target = invalidIndex < 0 ? 0 : invalidIndex;
    if (selected !== target) {
      setSelected(target);
      return;
    }
    handledReview.current = reviewRequest;
    const card = root.current?.querySelectorAll<HTMLElement>(
      ".forward-input-card",
    )[target];
    const field =
      card?.querySelector<HTMLElement>('[aria-invalid="true"]') ??
      card?.querySelector<HTMLElement>("input:not(:disabled)");
    revealField(field ?? null);
  }, [reviewRequest, active, selected, validity]);
  useEffect(() => onValidity(valid), [valid, onValidity]);
  // A module that declares its inputs in comments can apply them again after
  // its `# input` lines change.
  const entryCode =
    draft.script == null && !draft.blueprint
      ? sourceCode(draft, entryPath(draft))
      : "";
  const declares = (word: string) =>
    new RegExp(`^\\s*#\\s*${word}\\b`, "m").test(entryCode);
  const [reading, setReading] = useState(false);
  const [readNotes, setReadNotes] = useState<string[] | null>(null);
  async function readFromCode() {
    setReading(true);
    setReadNotes(null);
    try {
      const read = await api.readSource(
        entryCode,
        draft.name,
        draft.class_name || undefined,
      );
      keys.current = forwardInputs(read.draft).map(() => crypto.randomUUID());
      setSelected(0);
      // Constructor arguments and capture mode change only when declared, so
      // values set here are not replaced by defaults.
      onChange({
        ...draft,
        input: read.draft.input,
        input_name: read.draft.input_name,
        additional_inputs: read.draft.additional_inputs,
        ...(declares("constructor")
          ? { constructor: read.draft.constructor }
          : {}),
        ...(declares("capture")
          ? { capture_mode: read.draft.capture_mode }
          : {}),
      });
      setReadNotes(read.notes);
    } catch (error) {
      setReadNotes([(error as Error).message]);
    } finally {
      setReading(false);
    }
  }
  function update(
    index: number,
    next: ForwardInput,
    mode = draft.capture_mode,
  ) {
    onChange({
      ...withForwardInputs(
        draft,
        inputs.map((item, i) => (i === index ? next : item)),
      ),
      capture_mode: mode,
    });
  }
  return (
    <section
      ref={root}
      className="config-card forward-inputs"
      aria-label="Forward inputs"
    >
      <div className="section-label">
        <Box size={16} /> Input tensors{" "}
        {inputs.length > 1 && (
          <span className="input-count">{inputs.length} / 8</span>
        )}
        {declares("input") && (
          <button
            type="button"
            className="read-declarations"
            disabled={busy || reading}
            title="Apply the code's # input lines (and # constructor: or # capture: when present)"
            onClick={() => void readFromCode()}
          >
            {reading ? (
              <LoaderCircle size={12} className="spin" />
            ) : (
              <FileCode2 size={12} />
            )}
            Read from code
          </button>
        )}
      </div>
      {readNotes && (
        <p
          className={`read-declarations-note ${readNotes.length ? "has-notes" : ""}`}
          role="status"
        >
          {readNotes.length
            ? readNotes.join(" ")
            : "Inputs now match the code's declarations."}
        </p>
      )}
      {inputs.length > 1 && (
        <div className="forward-call">
          <span>CALL</span>
          <code>{forwardCall(inputs)}</code>
        </div>
      )}
      <div className="forward-input-list">
        {inputs.map((item, index) => (
          <InputCard
            key={keys.current[index]}
            identity={keys.current[index]}
            item={item}
            allowBinding={!draft.blueprint}
            index={index}
            expanded={selected === index}
            busy={busy}
            active={active && selected === index}
            captureMode={draft.capture_mode ?? "values"}
            invalid={
              validity[keys.current[index]] === false ||
              !!forwardIssue([item], draft.capture_mode) ||
              inputs.some((other, i) => i !== index && other.name === item.name)
            }
            nameInvalid={
              !validInputName(item.name) ||
              inputs.some((other, i) => i !== index && other.name === item.name)
            }
            bindingInvalid={
              item.binding === "positional" &&
              inputs
                .slice(0, index)
                .some((other) => other.binding === "keyword")
            }
            onValidity={recordValidity}
            onToggle={() => setSelected(selected === index ? -1 : index)}
            onChange={(next, mode) => update(index, next, mode)}
            onRemove={
              index === 0
                ? undefined
                : () => {
                    keys.current.splice(index, 1);
                    setSelected(Math.max(0, index - 1));
                    onChange(
                      withForwardInputs(
                        draft,
                        inputs.filter((_, i) => i !== index),
                      ),
                    );
                  }
            }
          />
        ))}
      </div>
      <div className="capture-mode" aria-label="Recording mode">
        {(["values", "shapes"] as const).map((mode) => (
          <button
            type="button"
            key={mode}
            disabled={busy}
            aria-pressed={(draft.capture_mode ?? "values") === mode}
            onClick={() => onChange({ ...draft, capture_mode: mode })}
          >
            {mode === "values" ? "Values & shapes" : "Shapes only"}
          </button>
        ))}
      </div>
      <p className="capture-description">
        {draft.capture_mode === "shapes"
          ? "Trace all input shapes without allocating values. Data-dependent code may require a value run."
          : "Record actual values for every input. Only visible cells are loaded into the diagram."}
      </p>
      {!draft.blueprint && (
        <button
          className="secondary-button add-forward-input"
          disabled={busy || inputs.length >= 8}
          onClick={() => {
            let number = inputs.length + 1;
            while (inputs.some((item) => item.name === `input${number}`))
              number++;
            keys.current.push(crypto.randomUUID());
            setSelected(inputs.length);
            onChange(
              withForwardInputs(draft, [
                ...inputs,
                {
                  name: `input${number}`,
                  binding: inputs.some((item) => item.binding === "keyword")
                    ? "keyword"
                    : "positional",
                  input: {
                    shape: [1, 3, 8],
                    axis_names: ["batch", "tokens", "features"],
                    generator: "random",
                    dtype: "float32",
                    seed: 7 + inputs.length,
                    random_stream: "input",
                    uploaded: null,
                  },
                },
              ]),
            );
          }}
        >
          <Plus size={14} /> Add tensor input
        </button>
      )}
      {issue && (
        <p className="field-error" role="alert">
          {issue}
        </p>
      )}
      {!valid && !issue && (
        <p className="field-error" role="alert">
          Resolve the invalid fields in your input cards before running.
        </p>
      )}
    </section>
  );
}

function InputCard({
  item,
  allowBinding,
  index,
  identity,
  expanded,
  busy,
  active,
  captureMode,
  invalid,
  nameInvalid,
  bindingInvalid,
  onValidity,
  onChange,
  onToggle,
  onRemove,
}: {
  item: ForwardInput;
  allowBinding: boolean;
  index: number;
  identity: string;
  expanded: boolean;
  busy: boolean;
  active: boolean;
  captureMode: "values" | "shapes";
  invalid: boolean;
  nameInvalid: boolean;
  bindingInvalid: boolean;
  onValidity: (id: string, valid: boolean) => void;
  onChange: (item: ForwardInput, mode?: "values" | "shapes") => void;
  onToggle: () => void;
  onRemove?: () => void;
}) {
  const id = useId();
  const report = useCallback(
    (valid: boolean) => onValidity(identity, valid),
    [identity, onValidity],
  );
  return (
    <section
      className={`forward-input-card ${expanded ? "expanded" : ""}`}
      aria-label={`Input ${index + 1}: ${item.name}`}
    >
      <button
        className="forward-input-heading"
        aria-expanded={expanded}
        aria-controls={id}
        onClick={onToggle}
      >
        <span className="input-number">
          {String(index + 1).padStart(2, "0")}
        </span>
        <span>
          <b>{item.name || "Unnamed input"}</b>
          <small>
            [{item.input.shape.join(" × ")}] · {item.input.dtype}
          </small>
        </span>
        {invalid && (
          <CircleAlert
            size={14}
            className="input-invalid-mark"
            aria-label="Input needs attention"
          />
        )}
        <ChevronDown size={15} />
      </button>
      <div id={id} className="forward-input-body" hidden={!expanded}>
        <TensorInputFields
          input={item.input}
          captureMode={captureMode}
          primary={index === 0}
          busy={busy}
          active={active}
          onChange={(input, mode) => onChange({ ...item, input }, mode)}
          onValidity={report}
        />
        {allowBinding && (
          <details className="disclosure-settings">
            <summary>
              Argument binding <span>optional</span>
            </summary>
            <p className="field-hint">
              Positional inputs follow their order. Keyword inputs use the exact
              argument name in forward.
            </p>
            <div className="field-pair">
              <label>
                Input name
                <input
                  value={item.name}
                  aria-invalid={nameInvalid}
                  maxLength={100}
                  disabled={busy}
                  onChange={(event) =>
                    onChange({ ...item, name: event.target.value })
                  }
                />
              </label>
              <label>
                Pass as
                <select
                  value={item.binding}
                  aria-invalid={bindingInvalid}
                  disabled={busy}
                  onChange={(event) =>
                    onChange({
                      ...item,
                      binding: event.target.value as ForwardInput["binding"],
                    })
                  }
                >
                  <option value="positional">Positional</option>
                  <option value="keyword">Keyword</option>
                </select>
              </label>
            </div>
          </details>
        )}
        {onRemove && (
          <button
            className="secondary-button remove-forward-input"
            disabled={busy}
            onClick={onRemove}
          >
            <Trash2 size={13} /> Remove input
          </button>
        )}
      </div>
    </section>
  );
}
