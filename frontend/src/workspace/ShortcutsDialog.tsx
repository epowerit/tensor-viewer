import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import "./shortcuts.css";

const mac =
  typeof navigator !== "undefined" &&
  /Mac|iPhone|iPad/.test(navigator.platform);
const mod = mac ? "⌘" : "Ctrl";

/** Every keyboard shortcut, grouped by where it works. */
export const SHORTCUTS: { group: string; items: [string[], string][] }[] = [
  {
    group: "Anywhere",
    items: [
      [[mod, "K"], "Project search: steps, calls, loops, tensors, actions"],
      [[mod, "↵"], "Run"],
      [[mod, "S"], "Save, and update the diagram from the saved code"],
      [["⇧", mod, "↵"], "Check shapes without running"],
      [[mod, "B"], "Show or hide the side bar"],
      [[mod, "J"], "Show or hide the Tensors shelf"],
      [[mod, "E"], "Show or hide the code"],
      [["?"], "These shortcuts"],
    ],
  },
  {
    group: "Stepping, anywhere",
    items: [
      [["F11"], "Step into: the next step"],
      [["F10"], "Step over the calls the current step makes"],
      [["⇧", "F11"], "Step out of the call around the current step"],
      [["⇧", "F10"], "Step back"],
      [
        [mod, "Alt", "["],
        "Fold the call around the current step into one card",
      ],
      [[mod, "Alt", "]"], "Unfold the card on screen, one level"],
    ],
  },
  {
    group: "Canvas",
    items: [
      [["Space"], "Play or pause"],
      [["↵"], "Inspect the current step"],
      [["Esc"], "Back to the overview, or clear a traced cell"],
      [["[", "]"], "Show the previous or next pass of a folded loop"],
      [["+", "−"], "Zoom"],
      [["Home"], "Fit the whole model"],
      [
        ["←", "→"],
        "On the Detail dial: coarser or finer, from the whole model to every step",
      ],
      [["Alt", "→"], "Follow the flow to the step that reads this tensor"],
      [["Alt", "←"], "Back to the step that made its first operand"],
      [["Alt", "↑", "↓"], "Other steps that read the same tensor"],
      [["←", "→", "↑", "↓"], "Pan"],
    ],
  },
  {
    group: "Step lesson",
    items: [
      [["[", "]"], "Same step in the previous or next pass"],
      [["Esc"], "Back to the canvas"],
    ],
  },
  {
    group: "Tensor grid",
    items: [
      [["←", "→", "↑", "↓"], "Move between cells"],
      [["Home", "End"], "First or last cell of the row"],
      [[mod, "Home", "End"], "First or last cell of the plane"],
      [["PgUp", "PgDn"], "Previous or next slice, e.g. channel"],
      [["⇧", "←", "→", "↑", "↓"], "Select a rectangle of cells; ⇧-click too"],
      [[mod, "A"], "Select the whole plane"],
      [["Esc"], "Clear the selection"],
      [[mod, "C"], "Copy the cell, or the selection as rows"],
      [["T"], "Swap the rows and columns on screen"],
      [["W"], "An 8 × 8 or 16 × 16 window of a large plane, in every grid"],
      [["[", "]"], "Previous or next state of the same name"],
      [[mod, "F"], "Find values in this tensor; / too"],
      [["G"], "Go to cell"],
      [["↵", "⇧", "↵"], "Next or previous match in Find values"],
    ],
  },
  {
    group: "Tensors shelf",
    items: [
      [["←", "→", "↑", "↓"], "Move between tensors"],
      [["P"], "Pin or unpin the focused tensor"],
      [["Esc"], "Clear the filter"],
    ],
  },
  {
    group: "Project in the side bar, focused",
    items: [
      [["F2"], "Rename it in place; Enter keeps, Escape cancels"],
      [["Del"], "Delete it, with Undo for a few seconds"],
    ],
  },
  {
    group: "Side bar edge, focused",
    items: [
      [["←", "→"], "Narrow or widen the side bar (⇧ for bigger steps)"],
      [["Home", "End"], "Narrowest or widest"],
      [["↵"], "The usual width; a double-click does the same"],
    ],
  },
  {
    group: "Editor",
    items: [
      [["F9"], "Toggle a breakpoint on the caret's line"],
      [[mod, "F10"], "Run to cursor: the caret line's next step"],
      [[mod, "I"], "Preview the tensor named at the caret"],
    ],
  },
];

export function ShortcutsDialog({ onClose }: { onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    dialog.current?.showModal();
  }, []);
  return (
    <dialog
      ref={dialog}
      className="run-compare shortcuts-dialog"
      aria-labelledby="shortcuts-title"
      onCancel={onClose}
      onClick={(event) => {
        if (event.target === dialog.current) onClose();
      }}
    >
      <header>
        <div>
          <span className="eyebrow">KEYBOARD</span>
          <h2 id="shortcuts-title">Keyboard shortcuts</h2>
        </div>
        <button
          className="icon-button"
          aria-label="Close keyboard shortcuts"
          onClick={onClose}
        >
          <X size={18} />
        </button>
      </header>
      <div className="shortcuts-groups">
        {SHORTCUTS.map(({ group, items }) => (
          <section key={group} aria-label={group}>
            <h3>{group}</h3>
            <dl>
              {items.map(([keys, action]) => (
                <div key={`${keys.join("+")}-${action}`}>
                  <dt>
                    {keys.map((key) => (
                      <kbd key={key}>{key}</kbd>
                    ))}
                  </dt>
                  <dd>{action}</dd>
                </div>
              ))}
            </dl>
          </section>
        ))}
      </div>
    </dialog>
  );
}
