import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  ArrowLeft,
  SlidersHorizontal,
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
import type {
  CompositionPlan,
  Draft,
  Project,
  Run,
  RunSummary,
} from "./api/client";
import { Walkthrough } from "./components/Walkthrough";
import { ProjectEditor } from "./components/ProjectEditor";
import { NewProject } from "./components/NewProject";
import { TensorMark } from "./components/TensorMark";
import { WorkspaceDrawer } from "./components/WorkspaceDrawer";
import { BuilderCanvas } from "./builder/BuilderCanvas";
import { compositionSignature, type BuildReadiness } from "./builder/readiness";
import { projectAction } from "./workflow/projectAction";
import { prepareComposition } from "./workflow/prepareComposition";
import { ForwardInputs } from "./inputs/ForwardInputs";
import { WeightLibrary } from "./weights/WeightLibrary";
import type { ToolGroup } from "./builder/Toolbox";
import { forwardInputs, forwardIssue } from "./inputs/forward";
import {
  runErrorTarget,
  type EditorNavigation,
} from "./sources/editorNavigation";

type Tab = "walkthrough" | "code" | "history" | "inputs";

export default function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [history, setHistory] = useState<RunSummary[]>([]);
  const [tab, setTab] = useState<Tab>("walkthrough");
  const [busy, setBusy] = useState(false);
  const [executing, setExecuting] = useState(false);
  const [checkingBeforeRun, setCheckingBeforeRun] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [showNew, setShowNew] = useState(false);
  const [editorValid, setValid] = useState(true);
  const [inputsValid, setInputsValid] = useState(true);
  const [builderEditingValid, setBuilderEditingValid] = useState(true);
  const [editorNavigation, setEditorNavigation] =
    useState<EditorNavigation | null>(null);
  const editorRequest = useRef(0);
  const [editorReviewRequest, setEditorReviewRequest] = useState(0);
  const [inputReviewRequest, setInputReviewRequest] = useState(0);
  const [modelReviewRequest, setModelReviewRequest] = useState(0);
  const valid =
    editorValid &&
    inputsValid &&
    (!draft?.blueprint || builderEditingValid) &&
    (!draft || !forwardIssue(forwardInputs(draft), draft.capture_mode));
  const [build, setBuild] = useState<BuildReadiness | null>(null);
  const [checkedPlan, setCheckedPlan] = useState<{
    signature: string;
    plan: CompositionPlan;
  } | null>(null);
  const currentBuild =
    draft?.blueprint && build?.signature === compositionSignature(draft)
      ? build
      : null;
  const builderValid = currentBuild?.state === "ready";
  const nextAction = projectAction({
    draft,
    hasRun: !!run,
    editorValid,
    inputsValid,
    builderEditingValid,
    build,
  });
  const unfinishedEdits =
    !!draft &&
    (!valid ||
      !draft.name.trim() ||
      (!draft.blueprint && nextAction.kind === "code"));
  const [surface, setSurface] = useState<"build" | "trace">("trace");
  const [toolboxGroup, setToolboxGroup] = useState<ToolGroup | null>(null);
  const diagramButton = useRef<HTMLButtonElement>(null);
  const workspace = useRef<HTMLElement>(null);
  const returnLabel =
    draft?.blueprint && surface === "build"
      ? "Back to model"
      : "Back to diagram";
  const selection = useRef(0);
  const dirty =
    draft && project
      ? draftSignature(draft) !== draftSignature(project)
      : false;
  const stale =
    draft && run
      ? draftSignature(draft) !== draftSignature(run.project)
      : false;
  const showSavedRunNotice =
    stale && tab === "walkthrough" && surface === "trace" && !run?.trace.error;

  async function openProject(next: Project) {
    const request = ++selection.current;
    setToolboxGroup(null);
    setProject(next);
    setDraft(toDraft(next));
    setSurface(next.blueprint ? "build" : "trace");
    setBuild(null);
    setCheckedPlan(null);
    setBuilderEditingValid(true);
    setEditorNavigation(null);
    setRun(null);
    setHistory([]);
    setValid(true);
    setInputsValid(true);
    setError("");
    const runs = await api.runs(next.id);
    if (request !== selection.current) return;
    setHistory(runs);
    if (runs[0]) {
      const saved = await api.getRun(runs[0].id);
      if (request === selection.current) {
        setRun(saved);
        setSurface("trace");
      }
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
          setShowNew(true);
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

  async function save(settings = draft) {
    if (!settings || !project) return null;
    const saved = await api.save(project.id, settings);
    setProject(saved);
    setDraft(toDraft(saved));
    setProjects((items) => items.map((p) => (p.id === saved.id ? saved : p)));
    return saved;
  }
  async function execute() {
    if (!draft || !project || busy || nextAction.kind !== "run") return;
    setError("");
    setToolboxGroup(null);
    setBusy(true);
    try {
      let settings = draft;
      if (draft.blueprint?.components.some((component) => component.custom)) {
        setCheckingBeforeRun(true);
        const plan = await prepareComposition(draft);
        setCheckedPlan({ signature: compositionSignature(draft), plan });
        if (!plan.valid) {
          reviewModel();
          return;
        }
        settings = {
          ...draft,
          code: plan.code,
          class_name: "ComposedModel",
          constructor: {},
        };
      }
      setCheckingBeforeRun(false);
      setExecuting(true);
      const saved = await save(settings);
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
      setCheckingBeforeRun(false);
    }
  }
  async function switchProject(next: Project) {
    if (next.id === project?.id || busy) return;
    if (!valid) {
      setError(
        !builderEditingValid && draft?.blueprint
          ? "Fix or revert the component arguments before switching projects."
          : !editorValid
            ? "Finish the code settings before switching projects."
            : "Fix the input settings before switching projects.",
      );
      if (draft?.blueprint && !builderEditingValid) editModel();
      else setTab(!editorValid ? "code" : "inputs");
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
    setTab(d.blueprint ? "walkthrough" : "inputs");
    setShowNew(false);
  }

  function openInputs() {
    setToolboxGroup(null);
    setDraft((current) =>
      current?.blueprint && !current.blueprint.has_input
        ? { ...current, blueprint: { ...current.blueprint, has_input: true } }
        : current,
    );
    setTab("inputs");
  }
  function returnToWorkspace() {
    setTab("walkthrough");
    requestAnimationFrame(() => {
      const target = diagramButton.current?.disabled
        ? workspace.current
        : diagramButton.current;
      target?.focus({ preventScroll: true });
    });
  }
  function editModel() {
    setToolboxGroup(null);
    if (draft?.blueprint) {
      setSurface("build");
      setTab("walkthrough");
    } else setTab("code");
  }

  function reviewModel() {
    editModel();
    setModelReviewRequest((request) => request + 1);
  }

  function primaryAction() {
    if (busy || loading) return;
    switch (nextAction.kind) {
      case "run":
        void execute();
        break;
      case "inputs":
        openInputs();
        setInputReviewRequest((request) => request + 1);
        break;
      case "code":
        setTab("code");
        setEditorReviewRequest((request) => request + 1);
        break;
      case "model":
        reviewModel();
        break;
      case "tools":
        editModel();
        setToolboxGroup("All");
        break;
    }
  }

  function fixRun() {
    if (!draft || !run) return;
    if (!draft.blueprint) {
      const target = runErrorTarget(draft, run);
      setEditorNavigation(
        target ? { ...target, request: ++editorRequest.current } : null,
      );
    }
    editModel();
  }

  function primaryButton() {
    const waiting =
      checkingBeforeRun || executing || (draft && nextAction.kind === "wait");
    return (
      <button
        className="primary-button"
        disabled={busy || loading || nextAction.kind === "wait"}
        title={nextAction.detail}
        onClick={primaryAction}
      >
        {waiting ? (
          <LoaderCircle size={16} className="spin" />
        ) : nextAction.kind === "run" ? (
          <Play size={15} fill="currentColor" />
        ) : (
          <ArrowRight size={16} />
        )}
        {checkingBeforeRun
          ? "Checking model…"
          : executing
            ? "Generating…"
            : nextAction.label}
      </button>
    );
  }

  function drawerActions() {
    return (
      <footer className="drawer-footer drawer-actions">
        <p>{nextAction.detail}</p>
        <div>
          <button className="secondary-button" onClick={returnToWorkspace}>
            <ArrowLeft size={14} /> {returnLabel}
          </button>
          {primaryButton()}
        </div>
      </footer>
    );
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
            {!project && (
              <option value="">
                {loading ? "Opening workspace…" : "Choose a project"}
              </option>
            )}
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
          {draft && (
            <>
              {run &&
                (draft.blueprint
                  ? surface !== "build" || tab !== "walkthrough"
                  : tab !== "code") && (
                  <button
                    className="secondary-button edit-model-button"
                    disabled={busy}
                    onClick={editModel}
                  >
                    <Code2 size={15} />
                    <span>Edit model</span>
                  </button>
                )}
              {(nextAction.kind !== "inputs" || tab === "inputs") && (
                <button
                  className="secondary-button inputs-button"
                  aria-pressed={tab === "inputs"}
                  disabled={busy}
                  onClick={() =>
                    tab === "inputs" ? setTab("walkthrough") : openInputs()
                  }
                >
                  <SlidersHorizontal size={15} />
                  <span>Inputs</span>
                </button>
              )}
            </>
          )}
          {project && (
            <button
              className={`secondary-button save-project-button ${unfinishedEdits ? "unfinished-button" : dirty ? "" : "saved-button"}`}
              aria-label={
                unfinishedEdits
                  ? "Finish unsaved edits"
                  : dirty
                    ? "Save project"
                    : "All changes saved"
              }
              title={
                unfinishedEdits
                  ? "Finish the incomplete settings before saving"
                  : dirty
                    ? "Save project changes"
                    : "All changes saved"
              }
              disabled={busy || (!unfinishedEdits && !dirty)}
              onClick={async () => {
                if (unfinishedEdits) {
                  primaryAction();
                  return;
                }
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
              {unfinishedEdits ? (
                <CircleAlert size={15} />
              ) : dirty ? (
                <Save size={15} />
              ) : (
                <Check size={15} />
              )}{" "}
              <span className="save-button-label">
                {unfinishedEdits ? "Finish edits" : dirty ? "Save" : "Saved"}
              </span>
            </button>
          )}
          {primaryButton()}
        </div>
      </header>
      <div className="workspace-body">
        <nav className="workspace-rail" aria-label="Workspace tools">
          {(
            [
              {
                id: "walkthrough",
                icon: Workflow,
                label: "Diagram",
                short: "Diagram",
              },
              {
                id: "code",
                icon: Code2,
                label: "Code",
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
              ref={item.id === "walkthrough" ? diagramButton : undefined}
              className={tab === item.id ? "active" : ""}
              aria-label={item.label}
              aria-pressed={tab === item.id}
              title={item.label}
              disabled={!draft || busy}
              onClick={() => {
                setToolboxGroup(null);
                if (item.id === "walkthrough" || tab === item.id)
                  returnToWorkspace();
                else setTab(item.id);
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
        <main className="workspace-content" ref={workspace} tabIndex={-1}>
          {(error || showSavedRunNotice) && (
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
              {showSavedRunNotice && (
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
          ) : (
            <>
              {draft?.blueprint && (
                <div
                  className="builder-surface"
                  hidden={surface !== "build"}
                  inert={tab !== "walkthrough" || showNew}
                >
                  <BuilderCanvas
                    key={project?.id}
                    draft={draft}
                    toolboxGroup={toolboxGroup}
                    onToolboxGroup={setToolboxGroup}
                    onChange={setDraft}
                    onReadiness={setBuild}
                    onEditingValidity={setBuilderEditingValid}
                    reviewRequest={modelReviewRequest}
                    checkedPlan={checkedPlan}
                    busy={busy}
                    hasRun={!!run}
                    onShowRun={() => {
                      setSurface("trace");
                      setTab("walkthrough");
                    }}
                    onInputs={openInputs}
                    active={
                      tab === "walkthrough" && surface === "build" && !showNew
                    }
                  />
                </div>
              )}
              {run || executing ? (
                <div
                  className="trace-surface"
                  hidden={surface === "build" && !!draft?.blueprint}
                >
                  <Walkthrough
                    run={run}
                    busy={executing}
                    stale={stale}
                    active={
                      tab === "walkthrough" && surface === "trace" && !showNew
                    }
                    onInspect={() => setTab("walkthrough")}
                    onEditModel={fixRun}
                    onEditInputs={openInputs}
                  />
                </div>
              ) : surface !== "build" || !draft?.blueprint ? (
                <section
                  className="project-ready"
                  aria-label={draft ? "Diagram setup" : "Welcome"}
                >
                  <div className="ready-symbol">
                    <Workflow size={32} strokeWidth={1.3} />
                  </div>
                  <span className="eyebrow">
                    {draft ? "MODEL → INPUT → DIAGRAM" : "TENSORVIEWER"}
                  </span>
                  <h1>
                    {draft
                      ? "Your model is ready to explore"
                      : "See what happens to every tensor"}
                  </h1>
                  <p>
                    {draft
                      ? "Choose the input your model expects, then generate a diagram to explore the transformations."
                      : "Build a model visually, paste code, or open a Python file."}
                  </p>
                  {draft ? (
                    <>
                      <div className="ready-model">
                        <Code2 size={16} />
                        <b>{draft.class_name}</b>
                        <span>Input</span>
                        <code>[{draft.input.shape.join(" × ")}]</code>
                      </div>
                      <button className="secondary-button" onClick={openInputs}>
                        <SlidersHorizontal size={15} /> Set up inputs{" "}
                        <ArrowRight size={14} />
                      </button>
                      <small>
                        Generate diagram runs your model with these inputs.
                      </small>
                    </>
                  ) : (
                    <button
                      className="primary-button"
                      onClick={() => setShowNew(true)}
                    >
                      <Plus size={16} /> New project
                    </button>
                  )}
                </section>
              ) : null}
            </>
          )}
          {draft && (
            <WorkspaceDrawer
              className="inputs-drawer"
              active={tab === "inputs"}
              label="Input settings"
              title="Inputs"
              eyebrow="STARTING TENSORS"
              closeLabel="Close inputs"
              returnLabel={returnLabel}
              onClose={returnToWorkspace}
              footer={drawerActions()}
            >
              <p className="drawer-intro">
                Set the shape and values that enter your model.
              </p>
              {stale && (
                <p className="draft-context">
                  <History size={14} />
                  Editing current inputs. The diagram shows a saved run; run
                  again to update it.
                </p>
              )}
              <ForwardInputs
                key={project?.id}
                draft={draft}
                onChange={setDraft}
                onValidity={setInputsValid}
                reviewRequest={inputReviewRequest}
                busy={busy}
                active={tab === "inputs"}
              />
              <details className="disclosure-settings weights-disclosure">
                <summary>
                  Model weights <span>optional</span>
                </summary>
                <WeightLibrary
                  draft={draft}
                  onChange={setDraft}
                  busy={busy}
                  active={tab === "inputs"}
                  invalid={!valid || (!!draft.blueprint && !builderValid)}
                />
              </details>
            </WorkspaceDrawer>
          )}
          {draft && (
            <WorkspaceDrawer
              className="editor-drawer"
              active={tab === "code"}
              label="Code editor"
              title={draft.blueprint ? "Generated code" : "Code"}
              eyebrow={draft.blueprint ? "FROM YOUR MODEL" : "PROJECT SETTINGS"}
              closeLabel="Close code editor"
              returnLabel={returnLabel}
              onClose={returnToWorkspace}
              footer={drawerActions()}
            >
              {stale && (
                <p className="draft-context">
                  <History size={14} />
                  Editing current code. The diagram shows a saved run; run again
                  to update it.
                </p>
              )}
              {draft.blueprint && (
                <div className="generated-code-note">
                  <p>
                    Read or copy the Python generated from your model. Configure
                    components on the model canvas, or use this as custom code
                    to edit it directly.
                  </p>
                  <button
                    className="secondary-button"
                    disabled={busy || !builderEditingValid}
                    onClick={() => {
                      setDraft({ ...draft, blueprint: null });
                      setSurface("trace");
                    }}
                  >
                    Use as custom code
                  </button>
                  {!builderEditingValid && (
                    <p className="field-error">
                      Finish or revert the component arguments first.{" "}
                      <button className="text-button" onClick={editModel}>
                        Review model
                      </button>
                    </p>
                  )}
                </div>
              )}
              <ProjectEditor
                key={project?.id}
                draft={draft}
                active={tab === "code"}
                navigation={editorNavigation}
                reviewRequest={editorReviewRequest}
                onChange={setDraft}
                onValidity={setValid}
                busy={busy}
                readOnly={!!draft.blueprint}
              />
            </WorkspaceDrawer>
          )}
          {tab === "history" && (
            <WorkspaceDrawer
              className="history-drawer"
              active
              label="Saved runs"
              title="Run history"
              eyebrow="SAVED EXECUTIONS"
              closeLabel="Close run history"
              returnLabel={returnLabel}
              onClose={returnToWorkspace}
            >
              <section className="history-panel">
                <div className="history-heading">
                  <h2>Previous runs</h2>
                  <p>
                    Revisit a saved execution with its original code and inputs.
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
                          entry_path: run.project.entry_path ?? "model.py",
                          import_root: run.project.import_root ?? ".",
                          source_files: Object.keys(run.project.files ?? {}),
                          repository: run.project.repository ?? null,
                          environment:
                            run.project.environment ?? "TensorViewer",
                          runtime: run.trace.runtime ?? {},
                        },
                        null,
                        2,
                      )}
                    </pre>
                  </details>
                )}
              </section>
            </WorkspaceDrawer>
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
