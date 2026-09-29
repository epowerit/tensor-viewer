import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  Boxes,
  Check,
  CircleAlert,
  Code2,
  History,
  LoaderCircle,
  Play,
  Plus,
  Save,
  Workflow,
  X,
} from "lucide-react";
import { api, toDraft, draftSignature } from "./api/client";
import type { Draft, Project, Run, RunSummary } from "./api/client";
import { Walkthrough } from "./components/Walkthrough";
import { ProjectEditor } from "./components/ProjectEditor";
import { NewProject } from "./components/NewProject";
import { TensorMark } from "./components/TensorMark";
import { BuilderCanvas } from "./builder/BuilderCanvas";
import { blankProject } from "./builder/model";
import type { ToolGroup } from "./builder/Toolbox";
import { forwardInputs, forwardIssue } from "./inputs/forward";

type Tab = "walkthrough" | "code" | "history";

export default function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [history, setHistory] = useState<RunSummary[]>([]);
  const [tab, setTab] = useState<Tab>("walkthrough");
  const [busy, setBusy] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [showNew, setShowNew] = useState(false);
  const [editorValid, setValid] = useState(true);
  const valid =
    editorValid &&
    (!draft || !forwardIssue(forwardInputs(draft), draft.capture_mode));
  const [builderValid, setBuilderValid] = useState(false);
  const [surface, setSurface] = useState<"build" | "trace">("trace");
  const [toolboxGroup, setToolboxGroup] = useState<ToolGroup | null>(null);
  const selection = useRef(0);
  const dirty =
    draft && project
      ? draftSignature(draft) !== draftSignature(project)
      : false;
  const stale =
    draft && run
      ? draftSignature(draft) !== draftSignature(run.project)
      : false;

  async function openProject(next: Project) {
    const request = ++selection.current;
    setToolboxGroup(null);
    setProject(next);
    setDraft(toDraft(next));
    setSurface(next.blueprint ? "build" : "trace");
    setBuilderValid(false);
    setRun(null);
    setHistory([]);
    setValid(true);
    setError("");
    const runs = await api.runs(next.id);
    if (request !== selection.current) return;
    setHistory(runs);
    if (runs[0]) {
      const saved = await api.getRun(runs[0].id);
      if (request === selection.current) setRun(saved);
    }
  }

  useEffect(() => {
    let cancelled = false;
    async function initialize() {
      try {
        const existing = await api.projects();
        if (cancelled) return;
        if (existing.length) {
          setProjects(existing);
          await openProject(existing[0]);
        } else {
          const initial = await api.create(blankProject("My first experiment"));
          setProjects([initial]);
          setProject(initial);
          setDraft(toDraft(initial));
          setSurface("build");
        }
      } catch (e) {
        if (!cancelled)
          setError(
            `Could not connect to the local backend. ${(e as Error).message}`,
          );
      } finally {
        if (!cancelled) {
          setLoading(false);
          setBusy(false);
        }
      }
    }
    void initialize();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(() => setNotice(""), 3500);
    return () => clearTimeout(timer);
  }, [notice]);

  async function save() {
    if (!draft || !project) return null;
    const saved = await api.save(project.id, draft);
    setProject(saved);
    setDraft(toDraft(saved));
    setProjects((items) => items.map((p) => (p.id === saved.id ? saved : p)));
    return saved;
  }
  async function execute() {
    if (!valid || (draft?.blueprint && !builderValid)) {
      setTab("code");
      return;
    }
    setError("");
    setToolboxGroup(null);
    setBusy(true);
    setExecuting(true);
    try {
      const saved = await save();
      if (!saved) return;
      setTab("walkthrough");
      const next = await api.run(saved.id);
      setRun(next);
      setSurface("trace");
      setHistory(await api.runs(saved.id));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
      setExecuting(false);
    }
  }
  async function switchProject(next: Project) {
    if (next.id === project?.id || busy) return;
    if (!valid) {
      setError("Fix the input settings before switching projects.");
      setTab("code");
      return;
    }
    setBusy(true);
    try {
      if (dirty) await save();
      await openProject(next);
      setTab("walkthrough");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function create(d: Draft) {
    if (dirty && valid) await save();
    const created = await api.create(d);
    setProjects((items) => [created, ...items]);
    await openProject(created);
    setTab("walkthrough");
    setShowNew(false);
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="project-navigation">
          <div className="brand" aria-label="TensorViewer">
            <TensorMark />
            <span>
              Tensor<span className="brand-light">Viewer</span>
            </span>
          </div>
          <select
            className="project-switcher"
            aria-label="Current project"
            value={project?.id ?? ""}
            disabled={busy || loading}
            onChange={(event) => {
              const next = projects.find(
                (item) => item.id === event.target.value,
              );
              if (next) void switchProject(next);
            }}
          >
            {!project && <option value="">Opening workspace…</option>}
            {projects.map((item) => (
              <option key={item.id} value={item.id}>
                {item.id === project?.id ? draft?.name || item.name : item.name}
              </option>
            ))}
          </select>
          <button
            className="secondary-button new-project-button"
            onClick={() => setShowNew(true)}
            disabled={loading || busy || !valid}
          >
            <Plus size={16} /> New project
          </button>
        </div>
        <div className="project-actions">
          {project && (
            <button
              className={`secondary-button save-project-button ${dirty ? "" : "saved-button"}`}
              aria-label={dirty ? "Save project" : "All changes saved"}
              title={dirty ? "Save project changes" : "All changes saved"}
              disabled={!dirty || busy || !valid || !draft?.name.trim()}
              onClick={async () => {
                setBusy(true);
                try {
                  await save();
                  setNotice("Project saved");
                } catch (e) {
                  setError((e as Error).message);
                } finally {
                  setBusy(false);
                }
              }}
            >
              {dirty ? <Save size={15} /> : <Check size={15} />}{" "}
              <span className="save-button-label">
                {dirty ? "Save" : "Saved"}
              </span>
            </button>
          )}
          <button
            className="primary-button"
            disabled={
              !draft ||
              busy ||
              !valid ||
              !draft.name.trim() ||
              (!!draft.blueprint && !builderValid)
            }
            title={
              draft?.blueprint && !builderValid
                ? "Configure your input and resolve connection errors to run"
                : "Save and run this project"
            }
            onClick={() => void execute()}
          >
            {executing ? (
              <LoaderCircle size={16} className="spin" />
            ) : (
              <Play size={15} fill="currentColor" />
            )}
            {executing ? "Running…" : "Run"}
          </button>
        </div>
      </header>
      <div className="workspace-body">
        <nav className="workspace-rail" aria-label="Workspace tools">
          {draft?.blueprint && (
            <button
              className={
                surface === "build" && tab === "walkthrough" ? "active" : ""
              }
              aria-label="Model builder"
              aria-pressed={surface === "build" && tab === "walkthrough"}
              onClick={() => {
                setToolboxGroup(null);
                setSurface("build");
                setTab("walkthrough");
              }}
              title="Model builder"
            >
              <Boxes size={19} />
              <span className="rail-label">Build</span>
              <span className="rail-tooltip">Model builder</span>
            </button>
          )}
          {(
            [
              {
                id: "walkthrough",
                icon: Workflow,
                label: "Tensor canvas",
                short: "Explore",
              },
              {
                id: "code",
                icon: Code2,
                label: "Code & inputs",
                short: "Code",
              },
              {
                id: "history",
                icon: History,
                label: "Run history",
                short: "Runs",
              },
            ] as const
          ).map((item) => (
            <button
              key={item.id}
              className={
                tab === item.id &&
                (item.id !== "walkthrough" || surface === "trace")
                  ? "active"
                  : ""
              }
              aria-label={item.label}
              aria-pressed={
                tab === item.id &&
                (item.id !== "walkthrough" || surface === "trace")
              }
              title={item.label}
              onClick={() => {
                setToolboxGroup(null);
                if (item.id === "walkthrough") setSurface("trace");
                setTab(
                  tab === item.id && item.id !== "walkthrough"
                    ? "walkthrough"
                    : item.id,
                );
              }}
            >
              <item.icon size={19} />
              <span className="rail-label" aria-hidden="true">
                {item.short}
              </span>
              <span className="rail-tooltip">{item.label}</span>
            </button>
          ))}
          <span className="rail-local" title="Running locally">
            <span className="status-dot" />
          </span>
        </nav>
        <main className="workspace-content">
          {(error ||
            (stale && tab === "walkthrough" && surface === "trace")) && (
            <div className="workspace-messages">
              {error && (
                <div className="app-error" role="alert">
                  <CircleAlert size={17} />
                  <span>{error}</span>
                  <button
                    className="icon-button"
                    aria-label="Dismiss error"
                    onClick={() => setError("")}
                  >
                    <X size={16} />
                  </button>
                </div>
              )}
              {stale && tab === "walkthrough" && surface === "trace" && (
                <div className="stale-banner">
                  <History size={14} />
                  Showing a saved execution. Run again to visualize your current
                  code and inputs.
                </div>
              )}
            </div>
          )}
          {loading ? (
            <div className="loading-state">
              <LoaderCircle className="spin" size={25} />
              <p>Opening your workspace…</p>
            </div>
          ) : surface === "build" && draft?.blueprint ? (
            <BuilderCanvas
              key={project?.id}
              draft={draft}
              toolboxGroup={toolboxGroup}
              onToolboxGroup={setToolboxGroup}
              onChange={setDraft}
              onValidity={setBuilderValid}
              busy={busy}
              hasRun={!!run}
              onShowRun={() => setSurface("trace")}
            />
          ) : (
            <Walkthrough
              run={run}
              busy={busy}
              active={tab === "walkthrough" && !showNew}
              onInspect={() => setTab("walkthrough")}
            />
          )}
          {draft && (
            <aside
              className="workspace-drawer editor-drawer"
              hidden={tab !== "code"}
              aria-label="Code and input editor"
            >
              <header className="drawer-heading">
                <div>
                  <span className="eyebrow">PROJECT SETTINGS</span>
                  <h2>Code & inputs</h2>
                </div>
                <button
                  className="icon-button"
                  aria-label="Close code editor"
                  onClick={() => setTab("walkthrough")}
                >
                  <X size={18} />
                </button>
              </header>
              <div className="drawer-scroll">
                {draft.blueprint && (
                  <div className="generated-code-note">
                    <p>
                      This code is generated from your canvas. Configure
                      components in Build, or switch to a custom Python project
                      to edit it directly.
                    </p>
                    <button
                      className="secondary-button"
                      onClick={() => {
                        setDraft({ ...draft, blueprint: null });
                        setSurface("trace");
                      }}
                    >
                      Use as custom code
                    </button>
                  </div>
                )}
                <ProjectEditor
                  key={project?.id}
                  active={tab === "code"}
                  draft={draft}
                  onChange={setDraft}
                  onValidity={setValid}
                  busy={busy || !!draft.blueprint}
                />
              </div>
            </aside>
          )}
          {tab === "history" && (
            <aside
              className="workspace-drawer history-drawer"
              aria-label="Saved runs"
            >
              <header className="drawer-heading">
                <div>
                  <span className="eyebrow">SAVED EXECUTIONS</span>
                  <h2>Run history</h2>
                </div>
                <button
                  className="icon-button"
                  aria-label="Close run history"
                  onClick={() => setTab("walkthrough")}
                >
                  <X size={18} />
                </button>
              </header>
              <div className="drawer-scroll">
                <section className="history-panel">
                  <div className="history-heading">
                    <h2>Previous runs</h2>
                    <p>
                      Revisit a saved execution with its original code and
                      inputs.
                    </p>
                  </div>
                  {history.length ? (
                    <div className="history-list">
                      {history.map((item, index) => (
                        <button
                          key={item.id}
                          disabled={busy}
                          onClick={async () => {
                            setBusy(true);
                            try {
                              setRun(await api.getRun(item.id));
                              setSurface("trace");
                              setTab("walkthrough");
                            } catch (e) {
                              setError((e as Error).message);
                            } finally {
                              setBusy(false);
                            }
                          }}
                        >
                          <span
                            className={`history-icon ${item.failed ? "failed" : ""}`}
                          >
                            {item.failed ? (
                              <CircleAlert size={18} />
                            ) : (
                              <Check size={18} />
                            )}
                          </span>
                          <div>
                            <b>
                              {item.failed
                                ? "Stopped execution"
                                : "Completed execution"}
                              {index === 0 && (
                                <span className="tiny-badge">Latest</span>
                              )}
                            </b>
                            <small>
                              {new Date(item.created_at).toLocaleString()}
                            </small>
                          </div>
                          <span>{item.operation_count} operations</span>
                          <ArrowRight size={16} />
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="history-empty">
                      <History size={28} />
                      <p>Your first run will appear here.</p>
                    </div>
                  )}
                  {run && (
                    <details className="run-configuration">
                      <summary>Configuration of the displayed run</summary>
                      <pre>
                        {JSON.stringify(
                          {
                            class_name: run.project.class_name,
                            constructor: run.project.constructor,
                            input: run.project.input,
                            input_name: run.project.input_name ?? "x",
                            input_binding:
                              run.project.input_binding ?? "positional",
                            additional_inputs:
                              run.project.additional_inputs ?? [],
                            weights: run.project.weights ?? null,
                          },
                          null,
                          2,
                        )}
                      </pre>
                    </details>
                  )}
                </section>
              </div>
            </aside>
          )}
        </main>
      </div>
      {showNew && (
        <NewProject onClose={() => setShowNew(false)} onCreate={create} />
      )}
      {notice && (
        <div className="toast" role="status">
          <Check size={15} />
          {notice}
        </div>
      )}
    </div>
  );
}
