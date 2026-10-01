import { useEffect, useRef, useState } from "react";
import { Box, LoaderCircle } from "lucide-react";
import { api, type Draft, type RuntimeEnvironment } from "../api/client";

export function EnvironmentSetup({
  draft,
  onChange,
  busy,
  onBusy,
}: {
  draft: Draft;
  onChange: (draft: Draft) => void;
  busy: boolean;
  onBusy: (busy: boolean) => void;
}) {
  const [items, setItems] = useState<RuntimeEnvironment[]>([]);
  const [requirements, setRequirements] = useState("");
  const [installing, setInstalling] = useState(false);
  const [error, setError] = useState("");
  const latest = useRef({ draft, onChange, onBusy });
  latest.current = { draft, onChange, onBusy };
  const setupVersion = useRef(0);
  const setupPending = useRef(false);
  useEffect(
    () => () => {
      setupVersion.current++;
    },
    [],
  );
  useEffect(() => {
    const controller = new AbortController();
    api
      .environments(controller.signal)
      .then((environments) => {
        if (!controller.signal.aborted) setItems(environments);
      })
      .catch((e) => {
        if (!controller.signal.aborted) setError(e.message);
      });
    return () => controller.abort();
  }, []);
  const selected = items.find((item) => item.id === draft.environment);
  return (
    <section className="config-card environment-setup">
      <div className="section-label">
        <Box size={16} /> Python environment
      </div>
      <label>
        Run with
        <select
          value={draft.environment ?? ""}
          disabled={busy || installing}
          onChange={(e) =>
            onChange({ ...draft, environment: e.target.value || null })
          }
        >
          <option value="">TensorViewer environment</option>
          {draft.environment && !selected && (
            <option value={draft.environment}>
              Saved environment · unavailable
            </option>
          )}
          {items.map((item) => (
            <option key={item.id} value={item.id}>
              {item.requirements.join(", ") || "Base dependencies"} ·{" "}
              {item.id.slice(0, 6)}
            </option>
          ))}
        </select>
      </label>
      {selected && (
        <small>
          Python {selected.python} · {Object.keys(selected.packages).length}{" "}
          packages available
        </small>
      )}
      <details>
        <summary>Add a dependency environment</summary>
        <p className="source-help">
          Create a separate environment using the backend’s base packages. Add
          pinned dependencies below; requirements files are never installed
          automatically.
        </p>
        <label>
          Pinned packages
          <textarea
            rows={3}
            className="json-input"
            value={requirements}
            placeholder="einops==0.8.1"
            disabled={busy || installing}
            onChange={(e) => setRequirements(e.target.value)}
          />
        </label>
        <small className="source-help">
          One name==version per line. Published wheels only. Leave empty to use
          base dependencies in a separate environment.
        </small>
        <button
          className="secondary-button small"
          disabled={busy || installing}
          onClick={async () => {
            if (busy || setupPending.current) return;
            const version = ++setupVersion.current;
            setupPending.current = true;
            setInstalling(true);
            onBusy(true);
            setError("");
            try {
              const item = await api.setupEnvironment(
                requirements
                  .split(/\n/)
                  .map((p) => p.trim())
                  .filter(Boolean),
              );
              if (version !== setupVersion.current) return;
              setItems((old) => [...old.filter((p) => p.id !== item.id), item]);
              // Inputs can change while setup runs in the background.
              latest.current.onChange({
                ...latest.current.draft,
                environment: item.id,
              });
            } catch (e) {
              if (version === setupVersion.current)
                setError((e as Error).message);
            } finally {
              if (version === setupVersion.current) {
                setupPending.current = false;
                setInstalling(false);
                latest.current.onBusy(false);
              }
            }
          }}
        >
          {installing && <LoaderCircle size={14} className="spin" />}
          {installing
            ? "Setting up environment…"
            : "Create and select environment"}
        </button>
      </details>
      {error && (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}
