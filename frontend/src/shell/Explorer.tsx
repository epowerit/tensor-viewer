import { useEffect, useMemo, useRef, useState } from "react";
import {
  Boxes,
  ChevronDown,
  ChevronRight,
  FileCode2,
  GitBranch,
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
} from "../api/client";
import { libraryProjectName } from "../components/LibraryPicker";
import {
  hiddenOperations,
  loopFolds,
  playbackSteps,
  representative,
} from "../journey/loops";
import { ShapeGlyph } from "../editor/ShapeGlyph";
import {
  entryPath,
  pathIssue,
  projectFiles,
  updateFile,
} from "../sources/files";
import "./collections.css";
import { TensorShape } from "../tensors/InkShape";
import { kindName } from "../operations/kindName";

type Props = {
  projects: Project[];
  project: Project | null;
  draft: Draft | null;
  run: Run | null;
  busy: boolean;
  activeFile: string;
  currentNode: string | null;
  onProject: (project: Project) => void;
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

/** Projects, the open project's files, and the steps of the displayed run. */
export function Explorer({
  projects,
  project,
  draft,
  run,
  busy,
  activeFile,
  currentNode,
  onProject,
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
  // The current step's row stays in view as playback moves.
  const stepList = useRef<HTMLOListElement>(null);
  useEffect(() => {
    stepList.current
      ?.querySelector('[aria-current="step"]')
      ?.scrollIntoView({ block: "nearest" });
  }, [currentNode, currentLoop]);
  // While a loop's repeats play, its row is the current one.
  const shownStep =
    currentNode && !currentLoop
      ? representative(hiddenOperations(folds), currentNode)
      : null;
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
  const entryByName = new Map(
    library.map((item) => [libraryProjectName(item), item]),
  );
  const matchingProjects = projects.filter((item) =>
    (item.id === project?.id ? draft?.name || item.name : item.name)
      .toLowerCase()
      .includes(projectQuery.trim().toLowerCase()),
  );
  const ownProjects = matchingProjects.filter(
    (item) => !entryByName.has(item.name),
  );
  const libraryProjects = matchingProjects
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
    return (
      <li key={item.id}>
        <button
          className={current ? "current" : ""}
          aria-current={current ? "true" : undefined}
          disabled={busy && !current}
          title={`${item.name} · ${projectKind(item)}`}
          onClick={() => onProject(item)}
        >
          <Icon size={14} />
          <span>{current ? draft?.name || item.name : item.name}</span>
        </button>
      </li>
    );
  };
  return (
    <div className="explorer collection-explorer">
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
        <ul className="explorer-list">{ownProjects.map(projectItem)}</ul>
        {!matchingProjects.length && (
          <p className="explorer-empty" role="status">
            {projectQuery
              ? "No projects match this name."
              : "Create a project to start exploring tensors."}
          </p>
        )}
        {!!matchingProjects.length && !ownProjects.length && !projectQuery && (
          <p className="explorer-empty">
            Your own projects appear here. The library is below.
          </p>
        )}
      </Section>
      {draft && (
        <Section
          title="Files"
          action={
            editable && (
              <>
                <button
                  className="icon-button"
                  aria-label="Replace the open file from a .py file"
                  title="Replace the open file from a .py file"
                  disabled={busy}
                  onClick={() => upload.current?.click()}
                >
                  <Upload size={13} />
                </button>
                <button
                  className="icon-button"
                  aria-label="Add source file"
                  title="Add source file"
                  disabled={busy || files.length >= 128}
                  onClick={() => setAdding(!adding)}
                >
                  <Plus size={14} />
                </button>
              </>
            )
          }
        >
          <ul className="explorer-list">
            {files.map((file) => (
              <li key={file}>
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
              </li>
            ))}
          </ul>
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
              />
              {issue && <p className="field-error">{issue}</p>}
            </form>
          )}
          <input
            hidden
            ref={upload}
            type="file"
            accept=".py,text/x-python"
            onChange={async (event) => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (!file || file.size > 500000) return;
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
        </Section>
      )}
      <Section
        title={`Steps${steps.length ? ` · ${steps.length}` : ""}${steps.length !== operations.length ? ` of ${operations.length}` : ""}`}
      >
        {operations.length ? (
          <ol className="explorer-list explorer-steps" ref={stepList}>
            {steps.map((step) => {
              if (step.fold) {
                const current = currentLoop === step.fold.id;
                return (
                  <li key={step.id}>
                    <button
                      className={`explorer-loop ${current ? "current" : ""}`}
                      aria-current={current ? "step" : undefined}
                      title={`${step.fold.text} · line ${step.fold.line} · ${step.fold.iterations.length} identical passes, played once`}
                      onClick={() => onSelect(step.id)}
                    >
                      <Repeat size={13} aria-hidden="true" />
                      <span>{step.fold.text}</span>
                      <small>×{step.fold.iterations.length}</small>
                    </button>
                  </li>
                );
              }
              const op = step.operation;
              const output = run!.trace.tensors[op.outputs[0]];
              const unlit =
                activatedThrough !== undefined &&
                activatedThrough >= 0 &&
                op.index > activatedThrough;
              return (
                <li key={op.id}>
                  <button
                    className={`${op.id === shownStep ? "current" : ""} ${op.status === "error" ? "failed" : ""} ${unlit ? "unlit" : ""}`}
                    aria-current={op.id === shownStep ? "step" : undefined}
                    onClick={() => onSelect(op.id)}
                    title={op.source?.text ?? op.kind}
                  >
                    <ShapeGlyph
                      shape={output?.shape ?? []}
                      failed={op.status === "error"}
                    />
                    <span>
                      {kindName(op.kind)}
                      {output && output.name !== op.kind && (
                        <b> {output.name}</b>
                      )}
                    </span>
                    <small>
                      {op.status === "error"
                        ? "failed"
                        : output && <TensorShape tensor={output} />}
                    </small>
                  </button>
                </li>
              );
            })}
          </ol>
        ) : (
          <p className="explorer-empty">Run to list every recorded step.</p>
        )}
      </Section>
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
