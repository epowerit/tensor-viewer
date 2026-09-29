import { useEffect, useId, useRef, useState } from "react";
import {
  ArrowDownToLine,
  BookmarkPlus,
  ChevronDown,
  Library,
  LoaderCircle,
} from "lucide-react";
import { api, type Draft, type InputFixture } from "../api/client";
import { fixtureSettings, generatorLabels, inputIssue } from "./fixtures";
import { UploadInput } from "./UploadInput";

export function InputLibrary({
  input,
  captureMode = "values",
  busy,
  invalid,
  onApply,
  active = true,
}: {
  input: Draft["input"];
  captureMode?: "values" | "shapes";
  busy: boolean;
  active?: boolean;
  invalid?: boolean;
  onApply: (settings: Pick<Draft, "input" | "capture_mode">) => void;
}) {
  const id = useId();
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<InputFixture[]>([]);
  const [selected, setSelected] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [retry, setRetry] = useState(0);
  const saveController = useRef<AbortController | null>(null);
  const saveButton = useRef<HTMLButtonElement>(null);
  useEffect(() => () => saveController.current?.abort(), []);
  useEffect(() => {
    if (!open || !active) return;
    const controller = new AbortController();
    setLoading(true);
    setError("");
    api
      .inputFixtures(controller.signal)
      .then((result) => {
        if (controller.signal.aborted) return;
        setItems(result);
        setLoading(false);
      })
      .catch((e: Error) => {
        if (!controller.signal.aborted) {
          setError(e.message);
          setLoading(false);
        }
      });
    return () => controller.abort();
  }, [open, active, retry]);
  const matches = items.filter((item) =>
    item.name.toLowerCase().includes(query.toLowerCase()),
  );
  const chosen = matches.find((item) => item.id === selected) ?? matches[0];
  const issue = invalid
    ? "Apply valid input settings before saving."
    : inputIssue(input, captureMode);

  async function save() {
    if (busy || saving || !name.trim() || issue) return;
    const controller = new AbortController();
    saveController.current = controller;
    setSaving(true);
    setError("");
    setNotice("");
    try {
      const fixture = await api.saveInputFixture(
        {
          name: name.trim(),
          ...fixtureSettings({ input, capture_mode: captureMode }),
        },
        controller.signal,
      );
      if (controller.signal.aborted) return;
      setItems((current) => [fixture, ...current]);
      setSelected(fixture.id);
      setQuery("");
      setName("");
      setNaming(false);
      setNotice(`Saved “${fixture.name}”. Available in every project.`);
      requestAnimationFrame(() => saveButton.current?.focus());
    } catch (e) {
      if (!controller.signal.aborted) setError((e as Error).message);
    } finally {
      if (!controller.signal.aborted) setSaving(false);
    }
  }

  return (
    <section className="input-library" aria-label="Saved input library">
      <button
        className="input-library-toggle"
        aria-expanded={open}
        aria-controls={id}
        disabled={saving}
        onClick={() => {
          setOpen(!open);
          setNotice("");
        }}
      >
        <Library size={15} />
        <span>Saved inputs</span>
        <ChevronDown size={14} />
      </button>
      {open && (
        <div className="input-library-body" id={id}>
          <p className="settings-note">
            Reuse a named input configuration across projects.
          </p>
          {loading ? (
            <p className="input-library-message" role="status">
              <LoaderCircle size={13} className="spin" /> Loading inputs…
            </p>
          ) : (
            <>
              {items.length > 0 ? (
                <>
                  {items.length > 5 && (
                    <input
                      aria-label="Search saved inputs"
                      placeholder="Find an input…"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                    />
                  )}
                  {chosen ? (
                    <>
                      <label>
                        Saved configuration
                        <select
                          value={chosen.id}
                          onChange={(e) => {
                            setSelected(e.target.value);
                            setNotice("");
                          }}
                        >
                          {matches.map((item) => (
                            <option key={item.id} value={item.id}>
                              {item.name} · [{item.input.shape.join(" × ")}]
                            </option>
                          ))}
                        </select>
                      </label>
                      <div className="input-fixture-preview">
                        <code>[{chosen.input.shape.join(", ")}]</code>
                        <span>
                          {chosen.input.axis_names.join(" · ") ||
                            "Unnamed axes"}
                        </span>
                        <span>
                          {generatorLabels[chosen.input.generator]} ·{" "}
                          {chosen.input.dtype} ·{" "}
                          {chosen.input.uploaded ? "model seed" : "seed"}{" "}
                          {chosen.input.seed}
                        </span>
                        {chosen.input.uploaded && (
                          <span className="input-upload-file">
                            {chosen.input.uploaded.file_name}
                          </span>
                        )}
                        <span>
                          {chosen.capture_mode === "shapes"
                            ? "Shapes only"
                            : "Values & shapes"}
                          {chosen.input.generator === "random" &&
                            (chosen.input.random_stream === "input"
                              ? " · independent input seed"
                              : " · shared model seed")}
                        </span>
                      </div>
                      <button
                        className="secondary-button small"
                        disabled={busy || saving}
                        onClick={() => {
                          onApply(fixtureSettings(chosen));
                          setNotice(
                            `Applied “${chosen.name}”. Further edits affect only this project.`,
                          );
                        }}
                      >
                        <ArrowDownToLine size={13} /> Use this input
                      </button>
                    </>
                  ) : (
                    <p className="input-library-message">
                      No matching saved inputs.
                    </p>
                  )}
                </>
              ) : (
                !error && (
                  <p className="input-library-message">
                    No saved inputs yet. Save this configuration to reuse it.
                  </p>
                )
              )}
            </>
          )}
          {error && (
            <div className="field-error" role="alert">
              {error}{" "}
              <button
                className="text-button"
                onClick={() => setRetry(retry + 1)}
                disabled={saving}
              >
                Retry loading
              </button>
            </div>
          )}
          <div className="input-library-save">
            {naming ? (
              <>
                <label>
                  Input name
                  <input
                    autoFocus
                    maxLength={80}
                    placeholder="e.g. Small token sequence"
                    value={name}
                    disabled={busy || saving}
                    onChange={(e) => setName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        void save();
                      }
                      if (e.key === "Escape") {
                        e.preventDefault();
                        e.stopPropagation();
                        setNaming(false);
                        requestAnimationFrame(() =>
                          saveButton.current?.focus(),
                        );
                      }
                    }}
                  />
                </label>
                <p className="settings-note">
                  [{input.shape.join(", ")}] ·{" "}
                  {generatorLabels[input.generator]} ·{" "}
                  {captureMode === "shapes" ? "Shapes only" : "Values & shapes"}
                </p>
                <div className="input-library-actions">
                  <button
                    className="primary-button small"
                    disabled={busy || saving || !name.trim() || !!issue}
                    onClick={() => void save()}
                  >
                    {saving ? (
                      <LoaderCircle size={13} className="spin" />
                    ) : (
                      <BookmarkPlus size={13} />
                    )}{" "}
                    Save input
                  </button>
                  <button
                    className="secondary-button small"
                    disabled={saving}
                    onClick={() => {
                      setNaming(false);
                      requestAnimationFrame(() => saveButton.current?.focus());
                    }}
                  >
                    Cancel
                  </button>
                </div>
              </>
            ) : (
              <button
                ref={saveButton}
                className="text-button"
                disabled={busy || loading || !!issue}
                onClick={() => {
                  setNaming(true);
                  setNotice("");
                }}
              >
                <BookmarkPlus size={14} /> Save current input
              </button>
            )}
            {issue && <p className="settings-note">{issue}</p>}
          </div>
          <UploadInput
            busy={busy || loading || saving}
            onImported={(fixture) => {
              setItems((current) => [fixture, ...current]);
              setSelected(fixture.id);
              setQuery("");
              setNotice(
                `Imported “${fixture.name}”. Choose Use this input to apply it.`,
              );
            }}
          />
          {notice && (
            <p className="input-library-notice" role="status">
              {notice}
            </p>
          )}
        </div>
      )}
    </section>
  );
}
