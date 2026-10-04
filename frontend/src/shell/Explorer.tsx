import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  CircleAlert,
  Boxes,
  ChevronDown,
  ChevronRight,
  FileCode2,
  FoldHorizontal,
  UnfoldHorizontal,
  GitBranch,
  MoreHorizontal,
  Package,
  Plus,
  Repeat,
  Search,
  SquareTerminal,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import {
  api,
  type Draft,
  type LibraryEntry,
  type Project,
  type Run,
  type Tensor,
} from "../api/client";
import { libraryProjectName } from "../components/LibraryPicker";
import {
  hiddenOperations,
  loopFolds,
  playbackSteps,
  representative,
} from "../journey/loops";
import { ShapeGlyph } from "../editor/ShapeGlyph";
import { TensorPeek } from "../editor/TensorPeek";
import {
  entryPath,
  pathIssue,
  projectFiles,
  updateFile,
} from "../sources/files";
import "./collections.css";
import { ProjectMenu, type ProjectAction } from "./ProjectMenu";
import { TensorShape } from "../tensors/InkShape";
import { kindName } from "../operations/kindName";
import {
  OTHER_CODE,
  expression,
  foldedOutline,
  stepOutline,
  type OutlineRow,
} from "./outline";

type Props = {
  projects: Project[];
  /** A project's newest run, when it has one. */
  lastRunOf?: (
    id: string,
  ) => { failed: boolean; operation_count: number } | undefined;
  project: Project | null;
  draft: Draft | null;
  run: Run | null;
  busy: boolean;
  activeFile: string;
  currentNode: string | null;
  /** Steps a folded card on the canvas runs together, as playback's step. */
  currentRange?: string[];
  /** Steps the last save added, edited, or reshaped, marked for a moment. */
  changed?: ReadonlySet<string> | null;
  /** Where a recorded line is in the code now, edited since the run. */
  placeLine?: (file: string | null, line: number) => number | null;
  /** The canvas's folding: a module header folds or unfolds its call. */
  canvasFolds?: {
    call: (
      path: string,
      first: string,
    ) => { id: string; folded: boolean } | null;
    toggle: (id: string) => void;
  } | null;
  onProject: (project: Project) => void;
  /** Rename, duplicate or delete a project, from its row's menu or keys. */
  onProjectAction?: (
    project: Project,
    action: ProjectAction,
    name?: string,
  ) => void;
  /** A project to rename in place, asked for elsewhere (the palette). */
  renameRequest?: { id: string; key: number } | null;
  onNewProject: () => void;
  onOpenFile: (path: string) => void;
  onChange: (draft: Draft) => void;
  onSelect: (node: string) => void;
  /** How far playback has activated the run; later steps are dimmed. */
  activatedThrough?: number;
  /** The folded loop whose repeats are playing. */
  currentLoop?: string | null;
};

function Section({
  title,
  action,
  children,
}: {
  title: string;
  action?: React.ReactNode;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(true);
  return (
    <section className="explorer-section">
      <header>
        <button aria-expanded={open} onClick={() => setOpen(!open)}>
          {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          {title}
        </button>
        {action}
      </header>
      {open && children}
    </section>
  );
}

export const projectKind = (
  item: Pick<Draft, "script" | "blueprint" | "repository">,
) =>
  item.script != null
    ? "console"
    : item.blueprint
      ? "canvas"
      : item.repository
        ? "repository"
        : "module";
const KIND_ICONS = {
  console: SquareTerminal,
  canvas: Boxes,
  repository: GitBranch,
  module: FileCode2,
};

/**
 * Projects, and the open project's files, each with an outline of the steps
 * its lines recorded in the displayed run.
 */
export function Explorer({
  projects,
  lastRunOf,
  project,
  draft,
  run,
  busy,
  activeFile,
  currentNode,
  currentRange,
  changed,
  placeLine = (_, line) => line,
  canvasFolds,
  onProject,
  onProjectAction,
  renameRequest,
  onNewProject,
  onOpenFile,
  onChange,
  onSelect,
  activatedThrough,
  currentLoop,
}: Props) {
  const [adding, setAdding] = useState(false);
  const [path, setPath] = useState("");
  const [projectQuery, setProjectQuery] = useState("");
  const [treeOpen, setTreeOpen] = useState(true);
  // A project's actions, open at a point; and the project renamed in place.
  const [menu, setMenu] = useState<{
    project: Project;
    x: number;
    y: number;
    from: HTMLElement | null;
  } | null>(null);
  const [renaming, setRenaming] = useState<string | null>(null);
  useEffect(() => {
    if (renameRequest) setRenaming(renameRequest.id);
  }, [renameRequest?.key]);
  // Projects pinned to the top of the list, kept in this browser.
  const [pinned, setPinned] = useState<string[]>(() => {
    try {
      const saved = JSON.parse(
        localStorage.getItem("tensorviewer.pinnedProjects") ?? "[]",
      );
      return Array.isArray(saved)
        ? saved.filter((id) => typeof id === "string")
        : [];
    } catch {
      return [];
    }
  });
  function togglePin(id: string) {
    setPinned((current) => {
      const next = current.includes(id)
        ? current.filter((other) => other !== id)
        : [...current, id];
      try {
        localStorage.setItem(
          "tensorviewer.pinnedProjects",
          JSON.stringify(next),
        );
      } catch {
        // Without storage, pins last until the page reloads.
      }
      return next;
    });
  }
  function projectAction(item: Project, action: ProjectAction) {
    if (action === "rename") setRenaming(item.id);
    else if (action === "pin") togglePin(item.id);
    else onProjectAction?.(item, action);
  }
  // Module calls folded in each file's outline, as "file\nkey".
  const [folded, setFolded] = useState<ReadonlySet<string>>(new Set());
  const foldedIn = (file: string) =>
    new Set(
      [...folded]
        .filter((entry) => entry.startsWith(`${file}\n`))
        .map((entry) => entry.slice(file.length + 1)),
    );
  function toggleFold(file: string, key: string) {
    const entry = `${file}\n${key}`;
    const next = new Set(folded);
    if (next.has(entry)) next.delete(entry);
    else next.add(entry);
    setFolded(next);
  }
  // A folded card the canvas plays as one step: its steps share a band.
  const inRange = useMemo(() => new Set(currentRange ?? []), [currentRange]);
  // A step's tensor peeks out beside the side bar while its row is hovered.
  const [peek, setPeek] = useState<{
    tensor: Tensor;
    source: string;
    left: number;
    top: number;
  } | null>(null);
  const peekTimer = useRef<number | undefined>(undefined);
  useEffect(() => () => window.clearTimeout(peekTimer.current), []);
  function peekAt(element: HTMLElement, tensor: Tensor, source: string) {
    window.clearTimeout(peekTimer.current);
    peekTimer.current = window.setTimeout(() => {
      const row = element.getBoundingClientRect();
      const side = element.closest(".side-bar")?.getBoundingClientRect() ?? row;
      setPeek({
        tensor,
        source,
        left: side.right + 10,
        top: Math.max(8, Math.min(row.top - 8, window.innerHeight - 300)),
      });
    }, 350);
  }
  function endPeek() {
    window.clearTimeout(peekTimer.current);
    setPeek(null);
  }
  const upload = useRef<HTMLInputElement>(null);
  const kind = draft ? projectKind(draft) : null;
  const editable = kind === "module" || kind === "repository";
  const files = draft ? Object.keys(projectFiles(draft)).sort() : [];
  const entry = draft ? entryPath(draft) : "";
  const issue =
    draft && path
      ? (pathIssue(path) ??
        (projectFiles(draft)[path] !== undefined
          ? "This file already exists."
          : null))
      : null;
  const operations = run?.trace.operations ?? [];
  // The step list follows playback: a folded loop's later passes sit behind
  // one loop row, as on the canvas.
  const folds = useMemo(() => (run ? loopFolds(run.trace) : []), [run]);
  const steps = useMemo(
    () => playbackSteps(operations, folds),
    [operations, folds],
  );
  const stepList = useRef<HTMLDivElement>(null);
  // While a loop's repeats play, its row is the current one.
  const shownStep =
    currentNode && !currentLoop
      ? representative(hiddenOperations(folds), currentNode)
      : null;
  // Steps live under the file whose lines recorded them, like an outline.
  const outline = useMemo(
    () => stepOutline(steps, operations, files, entry),
    // `files` is rebuilt each render; its contents follow the draft.
    [steps, operations, files.join("\n"), entry],
  );
  const currentStepId = currentLoop ? `loop:${currentLoop}` : shownStep;
  const currentFile = [...outline].find(([, rows]) =>
    rows.some((row) => row.kind === "step" && row.step.id === currentStepId),
  )?.[0];
  // A file's outline opens where playback is, and wherever it is opened.
  const [opened, setOpened] = useState<Record<string, boolean>>({});
  const isOpen = (file: string) =>
    opened[file] ??
    (file === currentFile ||
      (currentFile === undefined &&
        file === (outline.has(entry) ? entry : file)));
  // Library projects keep their curriculum order, grouped by track; your own
  // projects stay most recent first. A renamed copy counts as your own.
  const [library, setLibrary] = useState<LibraryEntry[]>([]);
  useEffect(() => {
    let current = true;
    api
      .library()
      .then((found) => current && setLibrary(found))
      .catch(() => {});
    return () => {
      current = false;
    };
  }, []);
  // The current step's row stays in view as playback moves, and the open
  // project's row when no step is current; the library arriving can move it.
  useEffect(() => {
    const list = stepList.current;
    (
      list?.querySelector('[aria-current="step"]') ??
      list?.querySelector(".explorer-in-range") ??
      list?.querySelector(".explorer-project-open")
    )?.scrollIntoView({ block: "nearest" });
  }, [currentNode, currentLoop, currentRange, project?.id, library.length]);
  const entryByName = new Map(
    library.map((item) => [libraryProjectName(item), item]),
  );
  const matchingProjects = projects.filter((item) =>
    (item.id === project?.id ? draft?.name || item.name : item.name)
      .toLowerCase()
      .includes(projectQuery.trim().toLowerCase()),
  );
  // Pinned projects lead the list, in the order they were pinned, whether
  // your own or from the library.
  const pinnedProjects = pinned.flatMap(
    (id) => matchingProjects.find((item) => item.id === id) ?? [],
  );
  const unpinned = matchingProjects.filter((item) => !pinned.includes(item.id));
  const ownProjects = unpinned.filter((item) => !entryByName.has(item.name));
  const libraryProjects = unpinned
    .filter((item) => entryByName.has(item.name))
    .sort(
      (a, b) =>
        entryByName.get(a.name)!.number - entryByName.get(b.name)!.number,
    );
  const tracks = [
    ...new Set(
      libraryProjects.map((item) => entryByName.get(item.name)!.track),
    ),
  ];
  const projectItem = (item: Project) => {
    const Icon = KIND_ICONS[projectKind(item)];
    const current = item.id === project?.id;
    const last = lastRunOf?.(item.id);
    const shownName = current ? draft?.name || item.name : item.name;
    const row = (
      <li
        key={item.id}
        className={`${current && draft ? "explorer-project-open" : ""}${menu?.project.id === item.id ? " explorer-project-menu-open" : ""}`}
      >
        {renaming === item.id ? (
          <label className="explorer-project-rename">
            <Icon size={14} />
            <input
              aria-label={`New name for ${shownName}`}
              defaultValue={shownName}
              autoFocus
              onFocus={(event) => event.currentTarget.select()}
              onKeyDown={(event) => {
                if (event.key === "Enter") event.currentTarget.blur();
                if (event.key === "Escape") {
                  event.currentTarget.value = shownName;
                  event.currentTarget.blur();
                }
              }}
              onBlur={(event) => {
                const name = event.currentTarget.value.trim();
                setRenaming(null);
                if (name && name !== shownName)
                  onProjectAction?.(item, "rename", name);
              }}
            />
          </label>
        ) : (
          <button
            className={current ? "current" : ""}
            aria-current={current ? "true" : undefined}
            disabled={busy && !current}
            title={`${item.name} · ${projectKind(item)} · ${
              last
                ? last.failed
                  ? `last run stopped with an error after ${last.operation_count} steps`
                  : `last run: ${last.operation_count} steps`
                : "not run yet"
            }`}
            onClick={() => onProject(item)}
            onContextMenu={(event) => {
              if (!onProjectAction) return;
              event.preventDefault();
              setMenu({
                project: item,
                x: event.clientX,
                y: event.clientY,
                from: event.currentTarget,
              });
            }}
            onKeyDown={(event) => {
              if (!onProjectAction || busy) return;
              if (event.key === "F2") {
                event.preventDefault();
                setRenaming(item.id);
              } else if (
                event.key === "Delete" ||
                (event.key === "Backspace" && (event.metaKey || event.ctrlKey))
              ) {
                event.preventDefault();
                if (projects.length > 1) onProjectAction(item, "delete");
              }
            }}
          >
            <Icon size={14} />
            <span>{shownName}</span>
            {last && (
              <small className="explorer-wide-only" aria-hidden="true">
                {last.operation_count}{" "}
                {last.operation_count === 1 ? "step" : "steps"}
              </small>
            )}
            {last?.failed && (
              <CircleAlert
                size={12}
                className="project-stopped"
                aria-label="Last run stopped with an error"
              />
            )}
          </button>
        )}
        {onProjectAction && renaming !== item.id && (
          <button
            className="icon-button explorer-project-action explorer-project-more"
            aria-label={`Actions for ${shownName}`}
            aria-haspopup="menu"
            aria-expanded={menu?.project.id === item.id}
            title="Rename, duplicate or delete"
            disabled={busy}
            onClick={(event) => {
              // Opens below the row, its right edge at the row's.
              const box = (
                event.currentTarget.closest("li") ?? event.currentTarget
              ).getBoundingClientRect();
              setMenu(
                menu?.project.id === item.id
                  ? null
                  : {
                      project: item,
                      x: box.right - 180,
                      y: box.bottom + 4,
                      from: event.currentTarget,
                    },
              );
            }}
          >
            <MoreHorizontal size={14} />
          </button>
        )}
        {current && draft && editable && (
          <>
            <button
              className="icon-button explorer-project-action explorer-file-action"
              aria-label="Replace the open file from a .py file"
              title="Replace the open file from a .py file"
              disabled={busy}
              onClick={() => upload.current?.click()}
            >
              <Upload size={13} />
            </button>
            <button
              className="icon-button explorer-project-action explorer-file-action"
              aria-label="Add source file"
              aria-expanded={adding}
              title="Add source file"
              disabled={busy || files.length >= 128}
              onClick={() => {
                setTreeOpen(true);
                setAdding(!adding);
              }}
            >
              <Plus size={14} />
            </button>
          </>
        )}
        {current && draft && (
          <button
            className="icon-button explorer-project-action"
            aria-expanded={treeOpen}
            aria-label={`${treeOpen ? "Hide" : "Show"} the files of ${draft.name || item.name}`}
            title={treeOpen ? "Hide its files" : "Show its files"}
            onClick={() => setTreeOpen(!treeOpen)}
          >
            {treeOpen ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
          </button>
        )}
      </li>
    );
    // The selected project opens onto its files, and each file onto its
    // steps, like a folder in an editor's explorer.
    return current && draft && treeOpen
      ? [
          row,
          <li key={`${item.id}-tree`} className="explorer-project-tree">
            {projectTree}
          </li>,
        ]
      : row;
  };
  /** One row of a file's outline: a module header, a step, or a loop. */
  const stepRow = (
    row: OutlineRow,
    index = 0,
    fold?: {
      file: string;
      key: string;
      hidden?: number;
      holds?: string[];
      first?: string;
    },
  ) => {
    if (row.kind === "module") {
      const open = fold?.hidden === undefined;
      // The call this header names, as the canvas draws it.
      const call =
        fold?.first && canvasFolds
          ? canvasFolds.call(row.path.split(" / ").at(-1)!, fold.first)
          : null;
      return (
        <li
          key={`module-${index}-${row.path}`}
          className={`explorer-step-module${open ? "" : " folded"}${!open && fold?.holds?.some((id) => inRange.has(id)) ? " explorer-in-range" : ""}${!open && fold?.holds?.some((id) => changed?.has(id)) ? " explorer-changed" : ""}`}
          style={{ paddingLeft: 10 + Math.min(row.depth, 4) * 8 }}
          title={row.path}
        >
          {fold ? (
            <button
              type="button"
              aria-expanded={open}
              aria-label={`${open ? "Fold" : "Unfold"} ${row.path}`}
              onClick={() => toggleFold(fold.file, fold.key)}
            >
              {open ? <ChevronDown size={10} /> : <ChevronRight size={10} />}
              {row.name}
              {!open && (
                <small>
                  {fold.hidden} {fold.hidden === 1 ? "step" : "steps"}
                </small>
              )}
            </button>
          ) : (
            row.name
          )}
          {call && (
            <button
              type="button"
              className={`explorer-canvas-fold${call.folded ? " folded" : ""}`}
              aria-pressed={call.folded}
              aria-label={`${call.folded ? "Unfold" : "Fold"} ${row.name} on the canvas`}
              title={
                call.folded
                  ? "Folded on the canvas into one card, played as one step. Unfold it there."
                  : "Fold this call on the canvas into one card, played as one step"
              }
              onClick={() => canvasFolds?.toggle(call.id)}
            >
              {call.folded ? (
                <UnfoldHorizontal size={11} />
              ) : (
                <FoldHorizontal size={11} />
              )}
            </button>
          )}
        </li>
      );
    }
    const { step, depth } = row;
    // In a file's outline the line number leads and rows indent under their
    // module; the flat list keeps a glyph of each shape instead.
    const inOutline = row.line !== null || depth > 0;
    // The number is where the line is now, in code edited since the run.
    const line =
      row.line !== null
        ? placeLine(
            (step.fold ? step.fold.file : step.operation?.source?.file) ?? null,
            row.line,
          )
        : null;
    const indent = inOutline
      ? { marginLeft: Math.min(depth, 4) * 8 }
      : undefined;
    const gutter = (
      <span className="explorer-step-line" aria-hidden="true">
        {line ?? ""}
      </span>
    );
    if (step.fold) {
      const current = currentLoop === step.fold.id;
      return (
        <li key={step.id} style={indent}>
          <button
            className={`explorer-loop ${current ? "current" : ""}`}
            aria-current={current ? "step" : undefined}
            title={`${step.fold.text} · line ${step.fold.line} · ${step.fold.iterations.length} identical passes, played once`}
            onClick={() => onSelect(step.id)}
          >
            {gutter}
            <Repeat size={13} aria-hidden="true" />
            <span>{step.fold.text}</span>
            <small>×{step.fold.iterations.length}</small>
          </button>
        </li>
      );
    }
    const op = step.operation;
    const output = run!.trace.tensors[op.outputs[0]];
    // A result holding NaN or infinity is marked, as the shelf and Flow do.
    const broken = output?.histogram?.non_finite ?? 0;
    const unlit =
      activatedThrough !== undefined &&
      activatedThrough >= 0 &&
      op.index > activatedThrough;
    return (
      <li key={op.id} style={indent}>
        <button
          className={`${op.id === shownStep ? "current" : ""} ${op.status === "error" ? "failed" : ""} ${unlit ? "unlit" : ""} ${inRange.has(op.id) ? "explorer-in-range" : ""} ${changed?.has(op.id) ? "explorer-changed" : ""}`}
          aria-current={op.id === shownStep ? "step" : undefined}
          onClick={() => {
            endPeek();
            onSelect(op.id);
          }}
          title={
            output
              ? undefined
              : `${line ? `Line ${line} · ` : ""}${op.source?.text ?? op.kind}`
          }
          onMouseEnter={(event) =>
            output &&
            peekAt(
              event.currentTarget,
              output,
              `${line ? `Line ${line} · ` : ""}${op.source?.text ?? kindName(op.kind)}`,
            )
          }
          onMouseLeave={endPeek}
        >
          {inOutline ? (
            gutter
          ) : (
            <ShapeGlyph
              shape={output?.shape ?? []}
              failed={op.status === "error"}
            />
          )}
          <span>
            {kindName(op.kind)}
            {output && output.name !== op.kind && <b> {output.name}</b>}
          </span>
          {inOutline && op.source?.text && (
            <code className="explorer-step-code explorer-wide-only">
              {expression(op.source.text)}
            </code>
          )}
          {!!broken && (
            <span
              className="explorer-step-broken"
              role="img"
              aria-label={`${broken.toLocaleString()} NaN or infinite ${broken === 1 ? "value" : "values"}`}
              title={`${broken.toLocaleString()} NaN or infinite ${broken === 1 ? "value" : "values"}`}
            />
          )}
          <small>
            {op.status === "error"
              ? "failed"
              : output && <TensorShape tensor={output} />}
          </small>
        </button>
      </li>
    );
  };
  /** A file's step count, and how many playback has reached. */
  const progress = (rows: OutlineRow[]) => {
    const own = rows.filter((row) => row.kind === "step");
    const total = own.length;
    const reached =
      activatedThrough !== undefined && activatedThrough >= 0
        ? own.filter(
            (row) =>
              row.kind === "step" &&
              (row.step.operation
                ? row.step.operation.index <= activatedThrough
                : true),
          ).length
        : total;
    return { total, reached };
  };
  /** The outline toggle on a file row, with its step count. */
  const outlineToggle = (file: string, label: string) => {
    const rows = outline.get(file);
    if (!rows) return null;
    const { total, reached } = progress(rows);
    const open = isOpen(file);
    return (
      <button
        type="button"
        className="explorer-outline-toggle"
        aria-expanded={open}
        aria-label={`${open ? "Hide" : "Show"} the ${total} steps of ${label}`}
        title={`${total} recorded ${total === 1 ? "step" : "steps"}${reached < total ? `, ${reached} reached by playback` : ""}`}
        onClick={() => setOpened({ ...opened, [file]: !open })}
      >
        {reached < total ? `${reached}/${total}` : total}
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
      </button>
    );
  };
  const outlineOf = (file: string) =>
    outline.has(file) && isOpen(file) ? (
      <li className="explorer-outline" key={`outline-${file}`}>
        <ol className="explorer-list explorer-steps">
          {foldedOutline(
            outline.get(file)!,
            foldedIn(file),
            (row) =>
              row.kind === "step" &&
              (row.step.id === shownStep ||
                (!!row.step.fold && row.step.fold.id === currentLoop)),
          ).map(({ row, key, hidden, holds, first }, index) =>
            stepRow(
              row,
              index,
              key ? { file, key, hidden, holds, first } : undefined,
            ),
          )}
        </ol>
      </li>
    ) : null;
  /** The open project's files, each with its outline of steps. */
  const projectTree = draft ? (
    <>
      <ul className="explorer-list">
        {files.flatMap((file) => [
          <li
            key={file}
            className={
              isOpen(file) && outline.has(file)
                ? "explorer-file-open"
                : undefined
            }
          >
            <button
              className={file === activeFile ? "current" : ""}
              aria-current={file === activeFile ? "true" : undefined}
              onClick={() => onOpenFile(file)}
              title={file}
            >
              <FileCode2 size={14} />
              <span>{kind === "console" ? "console.py" : file}</span>
              {file === entry && kind !== "console" && (
                <small>{kind === "canvas" ? "generated" : "entry"}</small>
              )}
            </button>
            {editable && file !== entry && (
              <button
                className="icon-button explorer-remove"
                aria-label={`Remove ${file}`}
                title={`Remove ${file}`}
                disabled={busy}
                onClick={() => {
                  const next = { ...draft.files };
                  delete next[file];
                  onChange({ ...draft, files: next });
                  if (file === activeFile) onOpenFile(entry);
                }}
              >
                <Trash2 size={12} />
              </button>
            )}
            {outlineToggle(file, kind === "console" ? "console.py" : file)}
          </li>,
          outlineOf(file),
        ])}
        {outline.has(OTHER_CODE) && [
          <li key="other-code">
            <button
              type="button"
              className="explorer-other-code"
              title="Steps recorded in code outside this project's files, such as an installed package"
              onClick={() =>
                setOpened({
                  ...opened,
                  [OTHER_CODE]: !isOpen(OTHER_CODE),
                })
              }
            >
              <Package size={14} />
              <span>Other code</span>
            </button>
            {outlineToggle(OTHER_CODE, "other code")}
          </li>,
          outlineOf(OTHER_CODE),
        ]}
      </ul>
      {!operations.length && (
        <p className="explorer-empty explorer-outline-hint">
          Run to outline each file's steps here.
        </p>
      )}
      {adding && editable && (
        <form
          className="explorer-add"
          onSubmit={(event) => {
            event.preventDefault();
            if (!path || issue) return;
            onChange(updateFile(draft, path, ""));
            onOpenFile(path);
            setPath("");
            setAdding(false);
          }}
        >
          <input
            autoFocus
            aria-label="New file path"
            placeholder="layers/attention.py"
            value={path}
            disabled={busy}
            onChange={(event) => setPath(event.target.value)}
            onKeyDown={(event) => {
              // Escape leaves without adding a file.
              if (event.key === "Escape") {
                event.preventDefault();
                event.stopPropagation();
                setPath("");
                setAdding(false);
              }
            }}
          />
          {issue && <p className="field-error">{issue}</p>}
        </form>
      )}
    </>
  ) : null;
  const currentListed = matchingProjects.some(
    (item) => item.id === project?.id,
  );
  return (
    <div
      className="explorer collection-explorer"
      ref={stepList}
      onScroll={endPeek}
    >
      {peek &&
        createPortal(
          <div
            className="tensor-peek-layer explorer-peek"
            style={{ left: peek.left, top: peek.top }}
          >
            <TensorPeek tensor={peek.tensor} />
            <code className="explorer-peek-source">{peek.source}</code>
          </div>,
          document.body,
        )}
      {menu && (
        <ProjectMenu
          name={
            menu.project.id === project?.id
              ? draft?.name || menu.project.name
              : menu.project.name
          }
          at={{ x: menu.x, y: menu.y }}
          canDelete={projects.length > 1}
          pinned={pinned.includes(menu.project.id)}
          library={entryByName.has(menu.project.name)}
          onAction={(action) => projectAction(menu.project, action)}
          onClose={() => {
            // Focus goes back to where the menu was opened from.
            menu.from?.focus({ preventScroll: true });
            setMenu(null);
          }}
        />
      )}
      <input
        hidden
        ref={upload}
        type="file"
        accept=".py,text/x-python"
        onChange={async (event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (!draft || !file || file.size > 500000) return;
          const code = await file.text();
          const found = code.match(
            /class\s+(\w+)\s*\([^)]*(?:nn\.Module|Module)/,
          );
          onChange({
            ...updateFile(draft, activeFile, code),
            class_name:
              activeFile === entry
                ? (found?.[1] ?? draft.class_name)
                : draft.class_name,
          });
        }}
      />

      <Section
        title="Projects"
        action={
          <button
            className="icon-button"
            aria-label="New project"
            title="New project"
            disabled={busy}
            onClick={onNewProject}
          >
            <Plus size={14} />
          </button>
        }
      >
        {(projects.length > 5 || projectQuery) && (
          <div className="collection-search">
            <Search size={13} aria-hidden="true" />
            <input
              aria-label="Find a project"
              placeholder="Find a project…"
              value={projectQuery}
              onChange={(event) => setProjectQuery(event.target.value)}
            />
            {projectQuery && (
              <button
                className="icon-button"
                aria-label="Clear project search"
                onClick={() => setProjectQuery("")}
              >
                <X size={12} />
              </button>
            )}
          </div>
        )}
        {!!pinnedProjects.length && (
          <>
            <h3 className="explorer-pinned-title">Pinned</h3>
            <ul className="explorer-list">{pinnedProjects.map(projectItem)}</ul>
            {!!ownProjects.length && (
              <h3 className="explorer-pinned-title">Recent</h3>
            )}
          </>
        )}
        <ul className="explorer-list">{ownProjects.map(projectItem)}</ul>
        {!matchingProjects.length && (
          <p className="explorer-empty" role="status">
            {projectQuery
              ? "No projects match this name."
              : "Create a project to start exploring tensors."}
          </p>
        )}
        {!!matchingProjects.length &&
          !ownProjects.length &&
          !pinnedProjects.length &&
          !projectQuery && (
            <p className="explorer-empty">
              Your own projects appear here. The library is below.
            </p>
          )}
      </Section>
      {draft && !currentListed && !projectQuery && (
        // An open project the list does not show still has its files.
        <Section title={draft.name || "Open project"}>{projectTree}</Section>
      )}
      {!draft && (
        // A run shown without its project still lists its steps.
        <Section title={`Steps${steps.length ? ` · ${steps.length}` : ""}`}>
          {operations.length ? (
            <ol className="explorer-list explorer-steps">
              {steps.map((step) =>
                stepRow({ kind: "step", step, line: null, depth: 0 }),
              )}
            </ol>
          ) : (
            <p className="explorer-empty">Run to list every recorded step.</p>
          )}
        </Section>
      )}
      {!!libraryProjects.length && (
        <Section title="Library">
          {tracks.map((track) => (
            <div key={track} className="explorer-track">
              <h3>{track}</h3>
              <ul className="explorer-list">
                {libraryProjects
                  .filter((item) => entryByName.get(item.name)!.track === track)
                  .map(projectItem)}
              </ul>
            </div>
          ))}
        </Section>
      )}
    </div>
  );
}
