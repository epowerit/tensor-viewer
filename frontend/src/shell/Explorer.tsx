import { useRef, useState } from "react";
import {
  Boxes,
  ChevronDown,
  ChevronRight,
  FileCode2,
  GitBranch,
  Plus,
  Search,
  SquareTerminal,
  Trash2,
  Upload,
  X,
} from "lucide-react";
import type { Draft, Project, Run } from "../api/client";
import { ShapeGlyph } from "../editor/ShapeGlyph";
import {
  entryPath,
  pathIssue,
  projectFiles,
  updateFile,
} from "../sources/files";
import "./collections.css";

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
  const matchingProjects = projects.filter((item) =>
    (item.id === project?.id ? draft?.name || item.name : item.name)
      .toLowerCase()
      .includes(projectQuery.trim().toLowerCase()),
  );
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
        <ul className="explorer-list">
          {matchingProjects.map((item) => {
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
          })}
        </ul>
        {!matchingProjects.length && (
          <p className="explorer-empty" role="status">
            {projectQuery
              ? "No projects match this name."
              : "Create a project to start exploring tensors."}
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
        title={`Steps${operations.length ? ` · ${operations.length}` : ""}`}
      >
        {operations.length ? (
          <ol className="explorer-list explorer-steps">
            {operations.map((op) => {
              const output = run!.trace.tensors[op.outputs[0]];
              return (
                <li key={op.id}>
                  <button
                    className={`${op.id === currentNode ? "current" : ""} ${op.status === "error" ? "failed" : ""}`}
                    aria-current={op.id === currentNode ? "step" : undefined}
                    onClick={() => onSelect(op.id)}
                    title={op.source?.text ?? op.kind}
                  >
                    <ShapeGlyph
                      shape={output?.shape ?? []}
                      failed={op.status === "error"}
                    />
                    <span>
                      {op.kind}
                      {output && output.name !== op.kind && (
                        <b> {output.name}</b>
                      )}
                    </span>
                    <small>
                      {op.status === "error"
                        ? "failed"
                        : output
                          ? `[${output.shape.join(", ")}]`
                          : ""}
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
    </div>
  );
}
