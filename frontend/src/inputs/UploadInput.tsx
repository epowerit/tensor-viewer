import { useEffect, useRef, useState } from "react";
import { FileUp, LoaderCircle, Upload } from "lucide-react";
import { api, type InputFixture } from "../api/client";
import { uploadIssue } from "./fixtures";

export function UploadInput({
  busy,
  onImported,
}: {
  busy: boolean;
  onImported: (fixture: InputFixture) => void;
}) {
  const [open, setOpen] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const controller = useRef<AbortController | null>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const picker = useRef<HTMLInputElement>(null);
  useEffect(() => () => controller.current?.abort(), []);
  async function upload() {
    if (!file || !name.trim() || busy || pending || uploadIssue(file)) return;
    const request = new AbortController();
    controller.current = request;
    setPending(true);
    setError("");
    try {
      const imported = await api.uploadInput(file, name.trim(), request.signal);
      if (request.signal.aborted) return;
      onImported(imported);
      setOpen(false);
      setFile(null);
      setName("");
      requestAnimationFrame(() => trigger.current?.focus());
    } catch (e) {
      if (!request.signal.aborted) setError((e as Error).message);
    } finally {
      if (!request.signal.aborted) setPending(false);
    }
  }
  return (
    <div className="input-upload">
      {!open ? (
        <button
          ref={trigger}
          className="text-button"
          disabled={busy}
          onClick={() => setOpen(true)}
        >
          <FileUp size={14} /> Import NumPy file
        </button>
      ) : (
        <>
          <p className="settings-note">
            One .npy tensor · float32, float64, or int64 · up to 8,388,608
            values. Imported inputs are saved in your local library.
          </p>
          <input
            hidden
            ref={picker}
            type="file"
            accept=".npy"
            aria-label="NumPy tensor file"
            disabled={busy || pending}
            onChange={(e) => {
              const next = e.target.files?.[0];
              e.target.value = "";
              if (!next) return;
              setError(uploadIssue(next));
              setFile(next);
              setName(next.name.replace(/\.npy$/i, "").slice(0, 80));
            }}
          />
          <button
            className="secondary-button small"
            disabled={busy || pending}
            onClick={() => picker.current?.click()}
          >
            <Upload size={13} />{" "}
            {file ? "Choose another file" : "Choose .npy file"}
          </button>
          {file && (
            <p className="input-upload-file">
              {file.name}
              <span>
                {(file.size / 1024).toLocaleString(undefined, {
                  maximumFractionDigits: 1,
                })}{" "}
                KiB
              </span>
            </p>
          )}
          <label>
            Imported input name
            <input
              value={name}
              maxLength={80}
              placeholder="Give this tensor a name"
              disabled={busy || pending}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  void upload();
                }
              }}
            />
          </label>
          {error && (
            <p className="field-error" role="alert">
              {error}
            </p>
          )}
          <div className="input-library-actions">
            <button
              className="primary-button small"
              disabled={
                busy ||
                pending ||
                !file ||
                !!(file && uploadIssue(file)) ||
                !name.trim()
              }
              onClick={() => void upload()}
            >
              {pending ? (
                <LoaderCircle size={13} className="spin" />
              ) : (
                <FileUp size={13} />
              )}{" "}
              {pending ? "Importing…" : "Import input"}
            </button>
            <button
              className="secondary-button small"
              disabled={pending}
              onClick={() => {
                setOpen(false);
                setError("");
                requestAnimationFrame(() => trigger.current?.focus());
              }}
            >
              Cancel
            </button>
          </div>
        </>
      )}
    </div>
  );
}
