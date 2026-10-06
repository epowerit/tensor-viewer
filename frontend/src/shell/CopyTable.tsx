import { useState } from "react";

type Cell = string | number | null | undefined;

/** Rows as CSV, quoted the way Flow's copy quotes them. */
export function toCsv(rows: Cell[][]): string {
  const cell = (value: Cell) => {
    const text = value === null || value === undefined ? "" : String(value);
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };
  return rows.map((row) => row.map(cell).join(",")).join("\n");
}

/**
 * Copies a tab's table as CSV, header first, for a spreadsheet or a
 * notebook; it says so for a moment once copied. `rows` is built only on a
 * click, so a large table costs nothing until then.
 */
export function CopyTable({
  rows,
  what,
}: {
  rows: () => Cell[][];
  /** What is copied, for the button's title. */
  what: string;
}) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      className="text-button copy-table"
      title={`Copy ${what} as CSV`}
      onClick={() => {
        void navigator.clipboard
          .writeText(toCsv(rows()))
          .then(() => {
            setCopied(true);
            window.setTimeout(() => setCopied(false), 1400);
          })
          .catch(() => {});
      }}
    >
      {copied ? "Copied" : "Copy as CSV"}
    </button>
  );
}
