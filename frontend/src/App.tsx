import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  BookOpen,
  Box,
  Check,
  ChevronDown,
  CircleAlert,
  Code2,
  FlaskConical,
  History,
  Layers3,
  LoaderCircle,
  PanelLeftClose,
  Play,
  Plus,
  Save,
  Workflow,
  X,
} from "lucide-react";
import { api, toDraft } from "./api/client";
import type { Draft, Project, Run, RunSummary, Template } from "./api/client";
import { Walkthrough } from "./components/Walkthrough";
import { ProjectEditor } from "./components/ProjectEditor";
import { NewProject } from "./components/NewProject";

type Tab = "walkthrough" | "code" | "history";

export default function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [history, setHistory] = useState<RunSummary[]>([]);
  const [tab, setTab] = useState<Tab>("walkthrough");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [showNew, setShowNew] = useState(false);
  const [valid, setValid] = useState(true);
  const [collapsed, setCollapsed] = useState(() => window.innerWidth < 900);
  const selection = useRef(0);
  const dirty =
    draft && project
      ? JSON.stringify(draft) !== JSON.stringify(toDraft(project))
      : false;
  const stale =
    draft && run
      ? JSON.stringify(draft) !== JSON.stringify(run.project)
      : false;

  async function openProject(next: Project) {
    const request = ++selection.current;
    setProject(next);
    setDraft(toDraft(next));
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
        const [existing, starters] = await Promise.all([
          api.projects(),
          api.templates(),
        ]);
        if (cancelled) return;
        setTemplates(starters);
        if (existing.length) {
          setProjects(existing);
          await openProject(existing[0]);
        } else {
          const initial = await api.create(starters[0].project);
          setProjects([initial]);
          setProject(initial);
          setDraft(toDraft(initial));
          setBusy(true);
          setLoading(false);
          // Only the bundled, newly created example runs automatically.
          const initialRun = await api.run(initial.id);
          if (!cancelled) {
            setRun(initialRun);
            setHistory(await api.runs(initial.id));
          }
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
    setProjects((items) => items.map((p) => (p.id === saved.id ? saved : p)));
    return saved;
  }
  async function execute() {
    if (!valid) {
      setTab("code");
      return;
    }
    setError("");
    setBusy(true);
    try {
      const saved = await save();
      if (!saved) return;
      setTab("walkthrough");
      const next = await api.run(saved.id);
      setRun(next);
      setHistory(await api.runs(saved.id));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
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
    setTab("code");
    setShowNew(false);
  }

  return (
    <div className={`app-shell ${collapsed ? "sidebar-collapsed" : ""}`}>
      <aside className="sidebar">
        <button
          className="icon-button sidebar-close"
          aria-label="Close sidebar"
          onClick={() => setCollapsed(true)}
        >
          <X size={18} />
        </button>
        <a className="brand" href="/" aria-label="TensorViewer home">
          <span className="brand-mark">
            <Box size={23} strokeWidth={1.65} />
          </span>
          <span>
            Tensor<span className="brand-light">Viewer</span>
            <small>MAKE IT MAKE SENSE</small>
          </span>
        </a>
        <div className="workspace-label">
          <span className="workspace-avatar">L</span>
          <div>
            Local workspace<small>Your learning lab</small>
          </div>
          <ChevronDown size={13} />
        </div>
        <button
          className="new-project-button"
          onClick={() => setShowNew(true)}
          disabled={!templates.length || busy || !valid}
        >
          <Plus size={16} />
          New project
        </button>
        <div className="sidebar-section-heading">
          PROJECTS<span>{projects.length}</span>
        </div>
        <nav className="project-list" aria-label="Projects">
          {projects.map((p) => (
            <button
              key={p.id}
              onClick={() => void switchProject(p)}
              disabled={busy}
              className={p.id === project?.id ? "active" : ""}
            >
              <Workflow size={16} />
              <span>{p.name}</span>
              {p.id === project?.id && <span className="project-active-dot" />}
            </button>
          ))}
        </nav>
        <div className="sidebar-learn">
          <div className="sidebar-section-heading">EXPLORE</div>
          <button
            onClick={() => {
              if (!busy && valid) setShowNew(true);
            }}
            disabled={busy || !valid}
          >
            <BookOpen size={16} />
            Example library<span className="tiny-badge">2</span>
          </button>
          <div className="learning-note">
            <span className="mini-stack">
              <Layers3 size={25} />
            </span>
            <b>
              Small tensors.
              <br />
              Big understanding.
            </b>
            <p>
              Follow a single value.
              <br />
              See the whole idea.
            </p>
          </div>
        </div>
        <div className="sidebar-bottom">
          <span className="dot" />
          <span>Running on your computer</span>
          <span className="version">v0.1</span>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumbs">
            <button
              className="icon-button"
              aria-label="Toggle sidebar"
              onClick={() => setCollapsed(!collapsed)}
            >
              <PanelLeftClose size={17} />
            </button>
            <span>Workspace</span>
            <span className="breadcrumb-slash">/</span>
            <b>{project?.name ?? "Tensor explorer"}</b>
          </div>
          <div className="topbar-right">
            <span className="local-pill">
              <span className="dot" />
              LOCAL
            </span>
            <span className="user-avatar">
              <FlaskConical size={17} />
            </span>
          </div>
        </header>
        <main>
          <div className="project-heading">
            <div>
              <div className="eyebrow">THE TENSOR LEARNING LAB</div>
              {draft ? (
                <input
                  className="project-title-input"
                  aria-label="Project name"
                  value={draft.name}
                  disabled={busy}
                  maxLength={100}
                  onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                />
              ) : (
                <h1>Inside the transformation</h1>
              )}
              <p>
                Follow the values. See the structure. Understand the operation.
              </p>
            </div>
            <div className="project-actions">
              <button
                className="secondary-button"
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
                <Save size={14} />
                Save{dirty && <span className="unsaved-dot" />}
              </button>
              <button
                className="primary-button"
                disabled={!draft || busy || !valid || !draft.name.trim()}
                onClick={() => void execute()}
              >
                {busy ? (
                  <LoaderCircle size={15} className="spin" />
                ) : (
                  <Play size={14} fill="currentColor" />
                )}
                {busy ? "Running…" : "Run forward pass"}
              </button>
            </div>
          </div>
          <div
            className="workspace-tabs"
            role="tablist"
            aria-label="Project views"
          >
            {(
              [
                { id: "walkthrough", icon: Workflow, label: "Walkthrough" },
                { id: "code", icon: Code2, label: "Code & inputs" },
                { id: "history", icon: History, label: "Run history" },
              ] as const
            ).map((t) => (
              <button
                key={t.id}
                role="tab"
                aria-selected={tab === t.id}
                onClick={() => setTab(t.id)}
              >
                <t.icon size={15} />
                {t.label}
                {t.id === "history" && history.length > 0 && (
                  <span>{history.length}</span>
                )}
              </button>
            ))}
            <div className="tab-meta">
              {draft?.class_name}
              <span>·</span>PyTorch
            </div>
          </div>
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
          {stale && tab === "walkthrough" && (
            <div className="stale-banner">
              <History size={14} />
              Showing a saved execution. Run again to visualize your current
              code and inputs.
            </div>
          )}
          {loading ? (
            <div className="loading-state">
              <LoaderCircle className="spin" size={25} />
              <p>Opening your workspace…</p>
            </div>
          ) : (
            <>
              <div hidden={tab !== "walkthrough"}>
                <Walkthrough run={run} busy={busy} />
              </div>
              {draft && (
                <div hidden={tab !== "code"}>
                  <ProjectEditor
                    key={project?.id}
                    draft={draft}
                    onChange={setDraft}
                    onValidity={setValid}
                    busy={busy}
                  />
                </div>
              )}
              {tab === "history" && (
                <section className="history-panel">
                  <div className="history-heading">
                    <h2>Every run, a preserved experiment.</h2>
                    <p>
                      Reopen the code, input settings, and tensor states
                      captured at execution time.
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
                          },
                          null,
                          2,
                        )}
                      </pre>
                    </details>
                  )}
                </section>
              )}
            </>
          )}
        </main>
        <footer className="main-footer">
          <span>
            <Box size={12} />
            TensorViewer
          </span>
          <span>One transformation at a time.</span>
          <span>Local execution · Forward pass only</span>
        </footer>
      </div>
      {showNew && templates.length > 0 && (
        <NewProject
          templates={templates}
          onClose={() => setShowNew(false)}
          onCreate={create}
        />
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
