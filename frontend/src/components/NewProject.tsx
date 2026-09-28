import { useEffect, useRef, useState } from "react";
import { ArrowRight, LoaderCircle, X } from "lucide-react";
import type { Draft } from "../api/client";
import { blankProject } from "../builder/model";

export function NewProject({
  onClose,
  onCreate,
}: {
  onClose: () => void;
  onCreate: (draft: Draft) => Promise<void>;
}) {
  const [name, setName] = useState("Untitled experiment");
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState("");
  const dialog = useRef<HTMLDialogElement>(null);
  const nameInput = useRef<HTMLInputElement>(null);
  useEffect(() => {
    dialog.current?.showModal();
    nameInput.current?.select();
  }, []);
  return (
    <dialog
      ref={dialog}
      className="new-project-dialog blank-project-dialog"
      aria-labelledby="new-project-title"
      onCancel={(e) => {
        if (creating) e.preventDefault();
        else onClose();
      }}
    >
      <form
        onSubmit={async (e) => {
          e.preventDefault();
          if (!name.trim() || creating) return;
          setCreating(true);
          try {
            await onCreate(blankProject(name.trim()));
          } catch (e) {
            setError((e as Error).message);
            setCreating(false);
          }
        }}
      >
        <div className="dialog-top">
          <span className="eyebrow">A blank canvas</span>
          <button
            type="button"
            className="icon-button"
            aria-label="Close new project"
            onClick={onClose}
            disabled={creating}
          >
            <X size={18} />
          </button>
        </div>
        <h2 id="new-project-title">New project</h2>
        <p>
          Start with open space. Add an input, a model, or a layer from the
          toolbox.
        </p>
        <label>
          Project name
          <input
            ref={nameInput}
            autoFocus
            required
            maxLength={100}
            disabled={creating}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        {error && (
          <p role="alert" className="field-error">
            {error}
          </p>
        )}
        <button
          type="submit"
          className="primary-button create-project-button"
          disabled={!name.trim() || creating}
        >
          {creating ? (
            <LoaderCircle size={15} className="spin" />
          ) : (
            <ArrowRight size={15} />
          )}{" "}
          {creating ? "Creating…" : "Open canvas"}
        </button>
      </form>
    </dialog>
  );
}
