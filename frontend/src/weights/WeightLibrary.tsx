import { useEffect, useId, useRef, useState } from "react";
import {
  Check,
  ChevronDown,
  CircleAlert,
  Database,
  FileUp,
  LoaderCircle,
  X,
} from "lucide-react";
import {
  api,
  draftSignature,
  type Draft,
  type SavedWeights,
  type WeightCheck,
} from "../api/client";
import { forwardInputs, forwardIssue } from "../inputs/forward";
import { checkpointFileIssue } from "./model";

function checkpointSize(bytes: number) {
  return bytes < 1024 * 1024
    ? `${(bytes / 1024).toFixed(1)} KiB`
    : `${(bytes / 1024 / 1024).toFixed(2)} MiB`;
}

export function WeightLibrary({
  draft,
  onChange,
  busy,
  active = true,
  invalid = false,
}: {
  draft: Draft;
  onChange: (draft: Draft) => void;
  busy: boolean;
  active?: boolean;
  invalid?: boolean;
}) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<SavedWeights[]>([]);
  const [selected, setSelected] = useState("");
  const [loading, setLoading] = useState(false);
  const [retry, setRetry] = useState(0);
  const [error, setError] = useState("");
  const [importing, setImporting] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [pending, setPending] = useState(false);
  const [checking, setChecking] = useState(false);
  const [report, setReport] = useState<{
    signature: string;
    result: WeightCheck;
  } | null>(null);
  const picker = useRef<HTMLInputElement>(null);
  const uploadController = useRef<AbortController | null>(null);
  const checkController = useRef<AbortController | null>(null);
  const signature = draftSignature(draft);
  const currentReport = report?.signature === signature ? report.result : null;
  const locked = busy || pending || checking;
  const issue = file ? checkpointFileIssue(file) : "";
  const chosen =
    items.find((item) => item.id === selected) ??
    items.find((item) => item.id === draft.weights?.id) ??
    items[0];
  useEffect(
    () => () => {
      uploadController.current?.abort();
      checkController.current?.abort();
    },
    [],
  );
  useEffect(() => {
    if (!open || !active) return;
    const controller = new AbortController();
    setLoading(true);
    setError("");
    api
      .weights(controller.signal)
      .then((weights) => {
        if (!controller.signal.aborted) setItems(weights);
      })
      .catch((e: Error) => {
        if (!controller.signal.aborted) setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [open, active, retry]);
  async function upload() {
    if (!file || issue || !name.trim() || locked) return;
    const controller = new AbortController();
    uploadController.current = controller;
    setPending(true);
    setError("");
    try {
      const saved = await api.uploadWeights(
        file,
        name.trim(),
        controller.signal,
      );
      if (controller.signal.aborted) return;
      setItems((current) => [saved, ...current]);
      setSelected(saved.id);
      setFile(null);
      setName("");
      setImporting(false);
    } catch (e) {
      if (!controller.signal.aborted) setError((e as Error).message);
    } finally {
      if (!controller.signal.aborted) setPending(false);
    }
  }
  async function check() {
    const controller = new AbortController();
    checkController.current = controller;
    setChecking(true);
    setError("");
    try {
      const result = await api.checkWeights(draft, controller.signal);
      if (!controller.signal.aborted) setReport({ signature, result });
    } catch (e) {
      if (!controller.signal.aborted) setError((e as Error).message);
    } finally {
      if (!controller.signal.aborted) setChecking(false);
    }
  }
  return (
    <section className="weight-library" aria-label="Model weights">
      <div className="weight-library-heading">
        <Database size={16} />
        <h3>Model weights</h3>
        <span>{draft.weights ? "Checkpoint" : "Initialized"}</span>
      </div>
      {draft.weights ? (
        <div className="active-checkpoint">
          <b>{draft.weights.name}</b>
          <small>
            {draft.weights.tensors.length.toLocaleString()} tensors ·{" "}
            {checkpointSize(draft.weights.byte_count)}
          </small>
          <code title={draft.weights.sha256}>
            SHA-256 {draft.weights.sha256.slice(0, 16)}…
          </code>
          <div className="checkpoint-actions">
            <button
              className="secondary-button small"
              disabled={
                locked ||
                invalid ||
                !!forwardIssue(forwardInputs(draft), draft.capture_mode)
              }
              onClick={() => void check()}
            >
              {checking ? (
                <LoaderCircle size={13} className="spin" />
              ) : (
                <Check size={13} />
              )}{" "}
              {checking ? "Checking…" : "Check compatibility"}
            </button>
            <button
              className="icon-button"
              title="Use initialized weights"
              aria-label="Use initialized weights"
              disabled={locked}
              onClick={() => {
                onChange({ ...draft, weights: null });
                setReport(null);
                setError("");
              }}
            >
              <X size={14} />
            </button>
          </div>
          <p className="settings-note">
            Checks construct the module on the meta device and load matching
            state. Run checks again before calling forward.
          </p>
        </div>
      ) : (
        <p className="settings-note">
          The module initializes its own parameters using the model seed. Choose
          a checkpoint to inspect saved weights instead.
        </p>
      )}
      {currentReport && (
        <div
          className={`checkpoint-report ${currentReport.compatible ? "compatible" : "incompatible"}`}
          role="status"
        >
          <div>
            {currentReport.compatible ? (
              <Check size={14} />
            ) : (
              <CircleAlert size={14} />
            )}
            <b>
              {currentReport.compatible
                ? `${currentReport.tensor_count} tensors match this model`
                : "Checkpoint does not match"}
            </b>
          </div>
          {!!currentReport.issues.length && (
            <ul>
              {currentReport.issues.map((message, index) => (
                <li key={index}>{message}</li>
              ))}
            </ul>
          )}
        </div>
      )}
      <button
        className="weight-library-toggle"
        aria-expanded={open}
        aria-controls={id}
        disabled={pending}
        onClick={() => setOpen(!open)}
      >
        <Database size={14} />
        <span>
          {draft.weights ? "Change checkpoint" : "Choose saved weights"}
        </span>
        <ChevronDown size={14} />
      </button>
      {open && (
        <div id={id} className="weight-library-body">
          {loading ? (
            <p className="settings-note" role="status">
              Loading checkpoints…
            </p>
          ) : (
            <>
              {items.length > 0 && (
                <>
                  <label>
                    Saved checkpoint
                    <select
                      value={chosen?.id ?? ""}
                      disabled={locked}
                      onChange={(e) => setSelected(e.target.value)}
                    >
                      {items.map((item) => (
                        <option key={item.id} value={item.id}>
                          {item.name} · {item.id.slice(0, 6)}
                        </option>
                      ))}
                    </select>
                  </label>
                  {chosen && (
                    <>
                      <WeightManifest weights={chosen} />
                      <button
                        className="primary-button small"
                        disabled={locked || chosen.id === draft.weights?.id}
                        onClick={() => {
                          onChange({ ...draft, weights: chosen });
                          setReport(null);
                          setError("");
                          setOpen(false);
                        }}
                      >
                        {chosen.id === draft.weights?.id
                          ? "Current checkpoint"
                          : "Use checkpoint"}
                      </button>
                    </>
                  )}
                </>
              )}
              {!items.length && (
                <p className="settings-note">
                  No saved weights yet. Import a state dictionary to start.
                </p>
              )}
            </>
          )}
          {!importing ? (
            <button
              className="text-button checkpoint-import-trigger"
              disabled={locked}
              onClick={() => {
                setImporting(true);
                setError("");
              }}
            >
              <FileUp size={14} /> Import checkpoint
            </button>
          ) : (
            <div className="checkpoint-import">
              <p className="settings-note">
                .pt or .pth · tensor state dictionary · up to 64 MiB. Full model
                objects are not supported.
              </p>
              <code className="checkpoint-export-hint">
                torch.save(model.state_dict(), "weights.pt")
              </code>
              <input
                hidden
                ref={picker}
                type="file"
                accept=".pt,.pth"
                aria-label="Checkpoint file"
                disabled={locked}
                onChange={(e) => {
                  const next = e.target.files?.[0];
                  e.target.value = "";
                  if (!next) return;
                  setFile(next);
                  setName(next.name.replace(/\.(pt|pth)$/i, "").slice(0, 80));
                  setError("");
                }}
              />
              <button
                className="secondary-button small"
                disabled={locked}
                onClick={() => picker.current?.click()}
              >
                <FileUp size={13} />
                {file ? "Choose another checkpoint" : "Choose .pt or .pth file"}
              </button>
              {file && (
                <p className="checkpoint-file">
                  {file.name} · {checkpointSize(file.size)}
                </p>
              )}
              <label>
                Checkpoint name
                <input
                  value={name}
                  maxLength={80}
                  disabled={locked}
                  onChange={(e) => setName(e.target.value)}
                />
              </label>
              {issue && (
                <p className="field-error" role="alert">
                  {issue}
                </p>
              )}
              <div className="checkpoint-actions">
                <button
                  className="primary-button small"
                  disabled={locked || !file || !!issue || !name.trim()}
                  onClick={() => void upload()}
                >
                  {pending ? (
                    <LoaderCircle size={13} className="spin" />
                  ) : (
                    <FileUp size={13} />
                  )}{" "}
                  {pending ? "Importing…" : "Save to library"}
                </button>
                <button
                  className="secondary-button small"
                  disabled={pending}
                  onClick={() => {
                    setImporting(false);
                    setFile(null);
                    setError("");
                  }}
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      )}
      {error && (
        <div role="alert" className="field-error">
          {error}{" "}
          <button
            className="text-button"
            disabled={locked}
            onClick={() => setRetry((n) => n + 1)}
          >
            Refresh library
          </button>
        </div>
      )}
    </section>
  );
}

function WeightManifest({ weights }: { weights: SavedWeights }) {
  const [query, setQuery] = useState("");
  const matches = weights.tensors.filter((tensor) =>
    tensor.name.toLowerCase().includes(query.trim().toLowerCase()),
  );
  return (
    <details className="checkpoint-manifest">
      <summary>
        {weights.tensors.length.toLocaleString()} saved tensors · inspect names
        and shapes
      </summary>
      <input
        aria-label="Find weight tensor"
        placeholder="Find a parameter or buffer…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <div className="checkpoint-tensor-list">
        {matches.slice(0, 20).map((tensor) => (
          <div key={tensor.name}>
            <code>{tensor.name}</code>
            <span>
              [{tensor.shape.join(", ")}] · {tensor.dtype}
            </span>
          </div>
        ))}
      </div>
      <p className="settings-note">
        {Math.min(20, matches.length)} of {matches.length} matches shown
      </p>
    </details>
  );
}
