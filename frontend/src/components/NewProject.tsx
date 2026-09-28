import { useEffect, useRef, useState } from "react";
import { ArrowRight, Box, Layers3, X } from "lucide-react";
import type { Draft, Template } from "../api/client";

export function NewProject({
  templates,
  onClose,
  onCreate,
}: {
  templates: Template[];
  onClose: () => void;
  onCreate: (draft: Draft) => Promise<void>;
}) {
  const [chosen, setChosen] = useState(templates[0]);
  const [name, setName] = useState("My tensor experiment");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    dialog.current?.showModal();
  }, []);
  return (
    <dialog
      ref={dialog}
      className="new-project-dialog"
      onCancel={(e) => {
        if (creating) e.preventDefault();
        else onClose();
      }}
    >
      <div className="dialog-top">
        <span className="eyebrow">A new experiment</span>
        <button
          className="icon-button"
          aria-label="Close new project"
          onClick={onClose}
          disabled={creating}
        >
          <X size={19} />
        </button>
      </div>
      <h2>Start with a little curiosity.</h2>
      <p>Choose a starting point. Every line is yours to change.</p>
      <label>
        Project name
        <input
          autoFocus
          value={name}
          maxLength={100}
          onChange={(e) => setName(e.target.value)}
        />
      </label>
      <div className="template-options">
        {templates.map((t) => (
          <button
            className={`template-option ${chosen.id === t.id ? "chosen" : ""}`}
            key={t.id}
            onClick={() => setChosen(t)}
          >
            {t.id === "attention" ? <Layers3 size={22} /> : <Box size={22} />}
            <b>{t.id === "attention" ? "Inside attention" : "Tensor basics"}</b>
            <span>{t.description}</span>
          </button>
        ))}
      </div>
      {error && (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
      <button
        className="primary-button create-project-button"
        disabled={!name.trim() || creating}
        onClick={async () => {
          setCreating(true);
          try {
            await onCreate({ ...chosen.project, name: name.trim() });
          } catch (e) {
            setError((e as Error).message);
            setCreating(false);
          }
        }}
      >
        {creating ? "Creating…" : "Create project"}
        <ArrowRight size={15} />
      </button>
    </dialog>
  );
}
