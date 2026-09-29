import { useEffect, useRef, useState } from "react";
import { Code2, LoaderCircle, Puzzle, Upload, X } from "lucide-react";
import {
  api,
  type CustomComponent,
  type CustomComponentDraft,
} from "../api/client";
import { parseArguments } from "./custom";

const EXAMPLE: CustomComponentDraft = {
  name: "Feature scaling",
  description: "Multiply each tensor element by a configurable scale.",
  class_name: "FeatureScale",
  constructor: { scale: 2 },
  code: `from torch import nn

class FeatureScale(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        scaled = x * self.scale
        return scaled
`,
};

export function CustomComponentDialog({
  initial,
  onClose,
  onSave,
}: {
  initial?: CustomComponentDraft;
  onClose: () => void;
  onSave: (component: CustomComponent) => void;
}) {
  const [draft, setDraft] = useState<CustomComponentDraft>(initial ?? EXAMPLE);
  const [argumentsText, setArgumentsText] = useState(
    JSON.stringify((initial ?? EXAMPLE).constructor, null, 2),
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  const nameInput = useRef<HTMLInputElement>(null);
  const file = useRef<HTMLInputElement>(null);
  useEffect(() => {
    dialog.current?.showModal();
    nameInput.current?.select();
  }, []);
  return (
    <dialog
      ref={dialog}
      className="custom-component-dialog"
      aria-labelledby="custom-component-title"
      onCancel={(e) => {
        if (saving) e.preventDefault();
        else onClose();
      }}
    >
      <form
        onSubmit={async (e) => {
          e.preventDefault();
          if (saving) return;
          setError("");
          try {
            const constructor = parseArguments(argumentsText);
            setSaving(true);
            const component = await api.saveComponent({
              ...draft,
              name: draft.name.trim(),
              constructor,
            });
            onSave(component);
          } catch (e) {
            setError((e as Error).message);
            setSaving(false);
          }
        }}
      >
        <header className="custom-dialog-header">
          <span className="custom-dialog-icon">
            <Puzzle size={22} />
          </span>
          <div>
            <span className="eyebrow">YOUR COMPONENT LIBRARY</span>
            <h2 id="custom-component-title">
              {initial ? "Create a new version" : "Make it your own"}
            </h2>
            <p>A PyTorch module you can reuse in any sequence.</p>
          </div>
          <button
            type="button"
            className="icon-button"
            aria-label="Close custom component"
            disabled={saving}
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </header>
        <div className="custom-dialog-body" inert={saving}>
          <div className="custom-source-panel">
            <div className="custom-source-heading">
              <span>
                <Code2 size={15} /> Python module
              </span>
              <button
                type="button"
                className="secondary-button"
                onClick={() => file.current?.click()}
              >
                <Upload size={14} /> Import .py
              </button>
            </div>
            <input
              ref={file}
              type="file"
              accept=".py,text/x-python"
              hidden
              onChange={async (e) => {
                const upload = e.target.files?.[0];
                e.target.value = "";
                if (!upload) return;
                if (upload.size > 48000) {
                  setError("Use a Python file with at most 12,000 characters.");
                  return;
                }
                try {
                  const code = await upload.text();
                  if (code.length > 12000)
                    throw new Error(
                      "Use a Python file with at most 12,000 characters.",
                    );
                  const className = code.match(/^class\s+(\w+)\s*\(/m)?.[1];
                  setDraft((d) => ({
                    ...d,
                    code,
                    class_name: className ?? d.class_name,
                  }));
                  setError("");
                } catch (e) {
                  setError((e as Error).message);
                }
              }}
            />
            <textarea
              className="custom-source-editor"
              aria-label="Custom component Python code"
              spellCheck={false}
              required
              maxLength={12000}
              value={draft.code}
              onChange={(e) => setDraft({ ...draft, code: e.target.value })}
            />
            <div className="custom-source-footer">
              <span>forward(x) → tensor</span>
              <span>{draft.code.length.toLocaleString()} / 12,000</span>
            </div>
          </div>
          <div className="custom-details-panel">
            <label>
              Component name
              <input
                ref={nameInput}
                autoFocus
                required
                maxLength={80}
                value={draft.name}
                onChange={(e) => setDraft({ ...draft, name: e.target.value })}
              />
            </label>
            <label>
              Description
              <input
                maxLength={240}
                placeholder="What does this component do?"
                value={draft.description}
                onChange={(e) =>
                  setDraft({ ...draft, description: e.target.value })
                }
              />
            </label>
            <label>
              Module class
              <input
                required
                pattern="[A-Za-z_][A-Za-z_0-9]*"
                value={draft.class_name}
                onChange={(e) =>
                  setDraft({ ...draft, class_name: e.target.value })
                }
              />
            </label>
            <label>
              Default arguments <span className="field-format">JSON</span>
              <textarea
                aria-label="Default constructor arguments"
                className="custom-arguments"
                spellCheck={false}
                value={argumentsText}
                onChange={(e) => setArgumentsText(e.target.value)}
              />
            </label>
            <div className="custom-contract">
              <b>One tensor in. One tensor out.</b>
              <p>
                Use installed Python packages and preserve the input dtype.
                Shapes are checked when you choose{" "}
                <strong>Check custom shapes</strong> in the builder.
              </p>
              <p>
                {initial
                  ? "This saves a new library entry. Other projects keep their existing source."
                  : "Saving adds this module to your local library without running it."}
              </p>
            </div>
          </div>
        </div>
        <footer className="custom-dialog-footer">
          <div>
            {error ? (
              <p className="field-error" role="alert">
                {error}
              </p>
            ) : (
              <p>Source and arguments travel with each project.</p>
            )}
          </div>
          <button
            type="button"
            className="secondary-button"
            disabled={saving}
            onClick={onClose}
          >
            Cancel
          </button>
          <button
            className="primary-button"
            disabled={saving || !draft.name.trim()}
          >
            {saving ? (
              <LoaderCircle size={15} className="spin" />
            ) : (
              <Puzzle size={15} />
            )}
            {saving
              ? "Saving…"
              : initial
                ? "Save & use new version"
                : "Save & add component"}
          </button>
        </footer>
      </form>
    </dialog>
  );
}

export function ConstructorArguments({
  value,
  onApply,
  onValidity,
}: {
  value: Record<string, unknown>;
  onApply: (value: Record<string, unknown>) => void;
  onValidity: (valid: boolean) => void;
}) {
  const serialized = JSON.stringify(value, null, 2);
  const [text, setText] = useState(serialized);
  const [error, setError] = useState("");
  const dirty = serialized !== text;
  useEffect(() => {
    setText(serialized);
    setError("");
  }, [serialized]);
  useEffect(() => {
    onValidity(!dirty);
    return () => onValidity(true);
  }, [dirty]);
  return (
    <div className="constructor-settings">
      <label>
        Constructor arguments <span className="field-format">JSON</span>
        <textarea
          className="custom-arguments"
          aria-label="Component constructor arguments"
          spellCheck={false}
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            setError("");
          }}
        />
      </label>
      {error && (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
      {dirty && (
        <button
          className="secondary-button"
          onClick={() => {
            try {
              onApply(parseArguments(text));
              setError("");
            } catch (e) {
              setError((e as Error).message);
            }
          }}
        >
          Apply arguments
        </button>
      )}
      <p className="settings-note">
        Settings apply to this node. Recheck shapes after a change.
      </p>
    </div>
  );
}
