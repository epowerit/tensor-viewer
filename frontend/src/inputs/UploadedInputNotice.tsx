import { FileCheck2 } from "lucide-react";
import type { Draft } from "../api/client";

export function UploadedInputNotice({
  input,
  disabled,
  onChange,
}: {
  input: Draft["input"];
  disabled: boolean;
  onChange: (input: Draft["input"]) => void;
}) {
  if (!input.uploaded) return null;
  return (
    <div className="uploaded-input-notice">
      <strong>
        <FileCheck2 size={15} /> Uploaded tensor
      </strong>
      <span>{input.uploaded.file_name}</span>
      <p>
        Shape and dtype come from the file. Add operations to transform them.
        Each run uses a fresh copy.
      </p>
      <button
        className="text-button"
        disabled={disabled}
        onClick={() =>
          onChange({ ...input, generator: "arange", uploaded: null })
        }
      >
        Use generated values
      </button>
    </div>
  );
}
