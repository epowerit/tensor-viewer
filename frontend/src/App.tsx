import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRight,
  CircleAlert,
  ChevronDown,
  Code2,
  Boxes,
  Plus,
  Workflow,
  Files,
  History,
  Link2,
  LoaderCircle,
  Play,
  Search,
  Settings,
  X,
} from "lucide-react";
import { api, toDraft, draftSignature } from "./api/client";
import type {
  CompositionPlan,
  Draft,
  Project,
  Run,
  RunSummary,
  Tensor,
} from "./api/client";
import { Walkthrough } from "./components/Walkthrough";
import { ProjectEditor } from "./components/ProjectEditor";
import { NewProject } from "./components/NewProject";
import { TensorMark } from "./components/TensorMark";
import { BuilderCanvas } from "./builder/BuilderCanvas";
import type { ToolGroup } from "./builder/Toolbox";
import { compositionSignature, type BuildReadiness } from "./builder/readiness";
import { projectAction } from "./workflow/projectAction";
import { prepareComposition } from "./workflow/prepareComposition";
import {
  runErrorTarget,
  type EditorNavigation,
} from "./sources/editorNavigation";
import { forwardInputs, forwardIssue } from "./inputs/forward";
import {
  appendSnippet,
  consoleProject,
  EXAMPLES,
  exampleInput,
  knownShapes,
  lastVariable,
  lineResults,
  needsValues,
  SNIPPETS,
  withShapeCheck,
} from "./console/script";
import { variables } from "./console/variables";
import { journeyStages } from "./journey/stages";
import { ShortcutsDialog } from "./workspace/ShortcutsDialog";
import { inTrace, lineageOf } from "./tensors/axisLineage";
import { inkOfAll } from "./tensors/axisInk";
import { AxisInkContext, CellPaintContext, InkShape } from "./tensors/InkShape";
import { cellPaintOfAll } from "./tensors/cellPaint";
import { CodeEditor } from "./editor/CodeEditor";
import { checkContracts } from "./editor/contracts";
import { ShapeGlyph } from "./editor/ShapeGlyph";
import { BottomPanel, type PanelTab } from "./shell/BottomPanel";
import { Explorer, projectKind } from "./shell/Explorer";
import { InputBar } from "./shell/InputBar";
import { collectProblems, problemCounts } from "./shell/problems";
import { RunsView } from "./shell/RunsView";
import { StatusBar } from "./shell/StatusBar";
import { entryPath, sourceCode, updateFile } from "./sources/files";
import { loopFolds, loopLines, type LoopLine } from "./journey/loops";
import { CommandPalette } from "./workspace/CommandPalette";
import { RunCompare } from "./workspace/RunCompare";
import { executionContextSignature } from "./workspace/executionContext";
import "./shell/tensorStudio.css";
import type { Command } from "./workspace/commands";
import {
  formatLocation,
  parseLocation,
  type WorkspaceLocation,
} from "./workspace/links";
import { kindName, ownName } from "./operations/kindName";

type WorkspacePanel = "settings" | "editor" | "side" | "shelf";

export default function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [run, setRun] = useState<Run | null>(null);
  const [history, setHistory] = useState<RunSummary[]>([]);
  const [side, setSide] = useState<"explorer" | "runs" | null>(null);
  const [settings, setSettings] = useState(false);
  const [editorOpen, setEditorOpen] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);
  const [panelTab, setPanelTab] = useState<PanelTab>("variables");
  const panelOpeners = useRef<Partial<Record<WorkspacePanel, HTMLElement>>>({});
  function rememberOpener(panel: WorkspacePanel) {
    const focused = document.activeElement;
    // A command dialog disappears after its action, so use the panel tool instead.
    if (
      focused instanceof HTMLElement &&
      focused !== document.body &&
      !focused.closest("dialog")
    )
      panelOpeners.current[panel] = focused;
    else delete panelOpeners.current[panel];
  }
  function focusPanel(panel: WorkspacePanel) {
    const selector = {
      settings: ".editor-drawer",
      editor: ".editor-column",
      side: ".side-bar",
      shelf: ".bottom-panel",
    }[panel];
    requestAnimationFrame(() => {
      if (!document.querySelector("dialog[open]"))
        document
          .querySelector<HTMLElement>(selector)
          ?.focus({ preventScroll: true });
    });
  }
  function closePanel(panel: WorkspacePanel) {
    const fallback = {
      settings: '.activity-bar [aria-label="Inputs and settings"]',
      editor: '[aria-label="Toggle code"]',
      shelf: '[aria-label="Toggle tensor shelf"]',
      side: `.activity-bar [aria-label="${side === "runs" ? "Run history" : "Projects"}"]`,
    }[panel];
    if (panel === "settings") setSettings(false);
    else if (panel === "editor") setEditorOpen(false);
    else if (panel === "side") setSide(null);
    else setPanelOpen(false);
    requestAnimationFrame(() => {
      const opener = panelOpeners.current[panel];
      const target =
        opener?.isConnected &&
        opener.getClientRects().length &&
        !opener.matches(":disabled")
          ? opener
          : document.querySelector<HTMLElement>(fallback);
      target?.focus({ preventScroll: true });
    });
  }
  function panelKeyDown(
    event: React.KeyboardEvent<HTMLElement>,
    panel: WorkspacePanel,
  ) {
    if (
      event.key !== "Escape" ||
      event.defaultPrevented ||
      document.querySelector("dialog[open]")
    )
      return;
    event.preventDefault();
    event.stopPropagation();
    closePanel(panel);
  }
  /** A review passes false: the reviewed field takes focus, not the panel. */
  function openSettings(focus = true) {
    if (!settings) rememberOpener("settings");
    setSettings(true);
    if (focus) focusPanel("settings");
  }
  function openCode(focus = true) {
    if (!editorOpen) rememberOpener("editor");
    setEditorOpen(true);
    if (focus) focusPanel("editor");
  }
  function openShelf(tab: PanelTab) {
    if (!panelOpen) rememberOpener("shelf");
    setPanelTab(tab);
    setPanelOpen(true);
    focusPanel("shelf");
  }
  const [activeFile, setActiveFile] = useState("model.py");
  const [openFiles, setOpenFiles] = useState<string[]>(["model.py"]);
  const [cursor, setCursor] = useState<{ line: number; column: number } | null>(
    null,
  );
  const [inputBarValid, setInputBarValid] = useState(true);
  const [setupOpen, setSetupOpen] = useState(false);
  useEffect(() => setSetupOpen(!run), [project?.id, run?.id]);
  // A shapes-only dry run of the current code, and the draft it was made for.
  const [check, setCheck] = useState<{ run: Run; signature: string } | null>(
    null,
  );
  const [checking, setChecking] = useState(false);
  const [liveCheck, setLiveCheck] = useState(() => {
    try {
      return localStorage.getItem("tensorviewer.liveCheck") === "on";
    } catch {
      return false;
    }
  });
  const checkRequest = useRef<AbortController | null>(null);
  const [editorWidth, setEditorWidth] = useState<number | null>(() => {
    try {
      const saved = Number(localStorage.getItem("tensorviewer.editorWidth"));
      return saved >= 240 ? saved : null;
    } catch {
      return null;
    }
  });
  // Breakpoints by project, then by file path, as sorted 1-based lines.
  const [breakpoints, setBreakpoints] = useState<
    Record<string, Record<string, number[]>>
  >(() => {
    try {
      return JSON.parse(
        localStorage.getItem("tensorviewer.breakpoints") ?? "{}",
      ) as Record<string, Record<string, number[]>>;
    } catch {
      return {};
    }
  });
  const [pauseOnWarnings, setPauseOnWarnings] = useState(() => {
    try {
      return localStorage.getItem("tensorviewer.pauseOnWarnings") === "on";
    } catch {
      return false;
    }
  });
  function changePauseOnWarnings(pause: boolean) {
    setPauseOnWarnings(pause);
    try {
      localStorage.setItem(
        "tensorviewer.pauseOnWarnings",
        pause ? "on" : "off",
      );
    } catch {
      // Private browsing: the choice still applies for this session.
    }
  }
  function saveBreakpoints(next: Record<string, Record<string, number[]>>) {
    setBreakpoints(next);
    try {
      localStorage.setItem("tensorviewer.breakpoints", JSON.stringify(next));
    } catch {
      // Private browsing: breakpoints still apply for this session.
    }
  }
  const [dragging, setDragging] = useState(false);
  function resizeEditor(width: number) {
    const next = Math.round(
      Math.max(240, Math.min(width, window.innerWidth - 420)),
    );
    setEditorWidth(next);
    try {
      localStorage.setItem("tensorviewer.editorWidth", String(next));
    } catch {
      // Private browsing: the width still applies for this session.
    }
  }
  const [busy, setBusy] = useState(false);
  // Keyboard shortcuts can repeat before React renders the disabled controls.
  const pendingAction = useRef(false);
  const [executing, setExecuting] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [showNew, setShowNew] = useState(false);
  const [editorValid, setValid] = useState(true);
  const [builderEditingValid, setBuilderEditingValid] = useState(true);
  const [checkingBeforeRun, setCheckingBeforeRun] = useState(false);
  // Requests that reveal the setting the next action needs; a new number repeats one.
  const [editorReviewRequest, setEditorReviewRequest] = useState(0);
  const [inputReviewRequest, setInputReviewRequest] = useState(0);
  const [modelReviewRequest, setModelReviewRequest] = useState(0);
  const [editorNavigation, setEditorNavigation] =
    useState<EditorNavigation | null>(null);
  const editorRequest = useRef(0);
  const valid =
    editorValid &&
    inputBarValid &&
    (!draft?.blueprint || builderEditingValid) &&
    (!draft || !forwardIssue(forwardInputs(draft), draft.capture_mode));
  const [build, setBuild] = useState<BuildReadiness | null>(null);
  const [checkedPlan, setCheckedPlan] = useState<{
    signature: string;
    plan: CompositionPlan;
  } | null>(null);
  const [surface, setSurface] = useState<"build" | "trace">("trace");
  const [toolboxGroup, setToolboxGroup] = useState<ToolGroup | null>(null);
  const [focusOperation, setFocusOperation] = useState<{
    id: string;
    key: number;
    cell?: number;
  } | null>(null);
  const [cell, setCell] = useState<{ node: string; index: number } | null>(
    null,
  );
  const [palette, setPalette] = useState(false);
  const [shortcuts, setShortcuts] = useState(false);
  const [compared, setCompared] = useState<Run | null>(null);
  const [currentOperation, setCurrentOperation] = useState<string | null>(null);
  const [currentLoop, setCurrentLoop] = useState<string | null>(null);
  const [activated, setActivated] = useState(-1);
  // A tensor name followed from the shelf: the nodes that wrote it.
  const [thread, setThread] = useState<string[] | null>(null);
  const [viewRevision, setViewRevision] = useState(0);
  const selection = useRef(0);
  const isConsole = draft?.script != null;
  const select = (id: string, cell?: number) =>
    setFocusOperation((previous) => ({
      id,
      cell,
      key: (previous?.key ?? 0) + 1,
    }));
  const dirty =
    draft && project
      ? draftSignature(draft) !== draftSignature(project)
      : false;
  const stale =
    draft && run
      ? draftSignature(draft) !== draftSignature(run.project)
      : false;

  const currentBuild =
    draft?.blueprint && build?.signature === compositionSignature(draft)
      ? build
      : null;
  const builderValid = currentBuild?.state === "ready";
  // The one next step: run, or the setting that has to be finished first.
  const nextAction = projectAction({
    draft,
    hasRun: !!run,
    editorValid,
    inputsValid: inputBarValid,
    builderEditingValid,
    build,
  });
  const canRun =
    !!draft && !!project && !loading && !busy && nextAction.kind === "run";
  const needsRunReview =
    !!draft &&
    !!project &&
    !loading &&
    !busy &&
    nextAction.kind !== "run" &&
    nextAction.kind !== "wait";

  /** Open the starting tensors, revealing the first one that needs attention. */
  function openInputs() {
    setToolboxGroup(null);
    setDraft((current) =>
      current?.blueprint && !current.blueprint.has_input
        ? { ...current, blueprint: { ...current.blueprint, has_input: true } }
        : current,
    );
    setInputReviewRequest((request) => request + 1);
    reviewInputs();
  }
  function editModel() {
    setToolboxGroup(null);
    setSettings(false);
    if (draft?.blueprint) setSurface("build");
    else openCode();
  }
  function reviewModel() {
    editModel();
    setModelReviewRequest((request) => request + 1);
  }
  function primaryAction() {
    if (busy || loading || pendingAction.current) return;
    switch (nextAction.kind) {
      case "run":
        void execute();
        break;
      case "inputs":
        openInputs();
        break;
      case "code":
        if (isConsole) openCode();
        else {
          if (draft && !draft.code.trim()) openCode();
          else {
            openSettings(false);
            setEditorReviewRequest((request) => request + 1);
          }
        }
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
  /** Fix code from a stopped run: open its source at the line that failed. */
  function fixRun() {
    if (!draft || !run) return;
    if (draft.blueprint) {
      reviewModel();
      return;
    }
    const target = isConsole
      ? run.trace.error?.line
        ? { file: entryPath(draft), line: run.trace.error.line }
        : null
      : runErrorTarget(draft, run);
    if (target && !isConsole) {
      setActiveFile(target.file);
      setOpenFiles((files) =>
        files.includes(target.file) ? files : [...files, target.file],
      );
    }
    setEditorNavigation(
      target ? { ...target, request: ++editorRequest.current } : null,
    );
    setToolboxGroup(null);
    setSettings(false);
    // The editor places the caret on the failed line itself.
    openCode(!target);
  }
  function reviewInputs() {
    if (!draft) return;
    if (!inputBarValid) {
      setSettings(false);
      setSetupOpen(true);
      requestAnimationFrame(() =>
        document
          .querySelector<HTMLElement>(
            '.experiment-disclosure [aria-invalid="true"]',
          )
          ?.focus(),
      );
    } else if (draft.blueprint) {
      setSettings(false);
      setSurface("build");
      setSetupOpen(true);
      requestAnimationFrame(() =>
        document
          .querySelector<HTMLElement>(
            !draft.name.trim()
              ? '[aria-label="Project setup"]'
              : '.builder-node.invalid .builder-node-heading, [aria-label="Configure input tensor"], .builder-empty .primary-button',
          )
          ?.focus(),
      );
    } else {
      openSettings();
      requestAnimationFrame(() => {
        const drawer = document.querySelector(".editor-drawer");
        const invalid = drawer?.querySelector<HTMLElement>(
          '[aria-invalid="true"]',
        );
        (
          invalid ??
          drawer?.querySelector<HTMLElement>(
            ".configuration input:not(:disabled)",
          )
        )?.focus();
      });
    }
  }

  function beginAction() {
    if (loading || busy || pendingAction.current) return false;
    pendingAction.current = true;
    setBusy(true);
    return true;
  }
  function finishAction() {
    pendingAction.current = false;
    setBusy(false);
  }

  async function openProject(next: Project, link: WorkspaceLocation = {}) {
    const request = ++selection.current;
    setToolboxGroup(null);
    setProject(next);
    setDraft(toDraft(next));
    setSurface(next.blueprint ? "build" : "trace");
    setEditorOpen(false);
    setPanelOpen(false);
    setActiveFile(entryPath(next));
    setOpenFiles([entryPath(next)]);
    setInputBarValid(true);
    setCursor(null);
    setCurrentOperation(null);
    setFocusOperation(null);
    setCell(null);
    setCompared(null);
    setBuild(null);
    setCheckedPlan(null);
    setBuilderEditingValid(true);
    setEditorNavigation(null);
    setRun(null);
    setHistory([]);
    setValid(true);
    setError("");
    await openProjectRun(next.id, link, request);
  }

  async function openProjectRun(
    projectId: string,
    link: WorkspaceLocation,
    request: number,
  ) {
    const runs = await api.runs(projectId);
    if (request !== selection.current) return;
    setHistory(runs);
    // A link names a saved run of this project; otherwise show the latest.
    const linked = link.run
      ? await api.getRun(link.run).catch(() => null)
      : null;
    const saved =
      linked?.project_id === projectId
        ? linked
        : runs[0]
          ? await api.getRun(runs[0].id)
          : null;
    if (request !== selection.current || !saved) return;
    setRun(saved);
    // A link can change only the cell, or return to the overview of the same run.
    // Reset the recorded view while retaining the project's unsaved draft.
    setFocusOperation(null);
    setCurrentOperation(null);
    setCell(null);
    setViewRevision((revision) => revision + 1);
    if (saved === linked) {
      setSurface("trace");
      if (link.node) select(link.node, link.cell);
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
          const link = parseLocation(window.location.hash);
          const target = existing.find((item) => item.id === link.project);
          await openProject(target ?? existing[0], target ? link : {});
        } else {
          // An empty workspace starts with the library, first lesson on top.
          const library = await api.installLibrary();
          if (cancelled) return;
          const initial = library.length
            ? library
            : [await api.create(consoleProject("My first experiment"))];
          setProjects(initial);
          await openProject(initial[0]);
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

  // Keep the address in step with the view, so it can be copied at any time.
  const address = formatLocation({
    project: project?.id,
    run: run?.project_id === project?.id ? run?.id : undefined,
    node: currentOperation ?? undefined,
    cell: cell?.node === currentOperation ? cell?.index : undefined,
  });
  useEffect(() => {
    if (loading || busy || window.location.hash === address) return;
    window.history.replaceState(
      null,
      "",
      address || window.location.pathname + window.location.search,
    );
  }, [address, loading, busy]);
  const openLink = useRef<() => Promise<void>>(async () => {});
  openLink.current = async () => {
    const link = parseLocation(window.location.hash);
    if (!link.project || formatLocation(link) === address || !beginAction())
      return;
    try {
      // The project may have been created in another tab since this one loaded.
      let known = projects;
      if (!known.some((item) => item.id === link.project)) {
        known = await api.projects();
        setProjects(known);
      }
      const target = known.find((item) => item.id === link.project);
      if (target && target.id === project?.id) {
        // A link to another cell/run must not replace the user's draft.
        await openProjectRun(target.id, link, ++selection.current);
      } else if (target) {
        if (dirty) {
          if (!valid || !draft?.name.trim()) {
            setError("Fix the project settings before switching projects.");
            openSettings();
            return;
          }
          await save();
        }
        await openProject(target, link);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      finishAction();
    }
  };
  useEffect(() => {
    const listener = () => void openLink.current();
    window.addEventListener("hashchange", listener);
    return () => window.removeEventListener("hashchange", listener);
  }, []);
  async function copyLink() {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setNotice("Link copied");
    } catch {
      setError(`Copy this address to share the view: ${window.location.href}`);
    }
  }

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
    if (loading || busy || pendingAction.current || !draft || !project) return;
    if (!canRun) {
      if (needsRunReview) primaryAction();
      return;
    }
    setError("");
    setToolboxGroup(null);
    if (!beginAction()) return;
    try {
      let settings = draft;
      // Custom components are checked together before the model is recorded.
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
      setSettings(false);
      const next = await api.run(saved.id);
      setRun(next);
      setSurface("trace");
      if (next.trace.error) {
        openShelf("problems");
      }
      setHistory(await api.runs(saved.id));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      finishAction();
      setExecuting(false);
      setCheckingBeforeRun(false);
    }
  }
  // Run from anywhere in the workspace, as in a code editor.
  const runShortcut = useRef(execute);
  runShortcut.current = execute;
  useEffect(() => {
    const listener = (event: KeyboardEvent) => {
      if (
        event.key.toLowerCase() === "k" &&
        (event.metaKey || event.ctrlKey) &&
        !document.querySelector("dialog[open]:not(.command-palette)")
      ) {
        event.preventDefault();
        setPalette((open) => !open);
        return;
      }
      // ? lists the shortcuts, unless someone is typing.
      if (
        (event.key === "?" || (event.code === "Slash" && event.shiftKey)) &&
        !event.metaKey &&
        !event.ctrlKey &&
        !(event.target as Element | null)?.closest?.(
          "input, textarea, select, [contenteditable]",
        ) &&
        !document.querySelector("dialog[open]")
      ) {
        event.preventDefault();
        setShortcuts(true);
        return;
      }
      if (
        event.key === "Enter" &&
        (event.metaKey || event.ctrlKey) &&
        !event.defaultPrevented &&
        !document.querySelector("dialog[open]")
      ) {
        event.preventDefault();
        if (event.shiftKey) void checkShortcut.current();
        else void runShortcut.current();
      }
    };
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);
  async function switchProject(next: Project) {
    if (next.id === project?.id || busy || pendingAction.current) return;
    if (!valid || !draft?.name.trim()) {
      setError(
        !builderEditingValid && draft?.blueprint
          ? "Fix or revert the component arguments before switching projects."
          : !editorValid
            ? "Finish the code settings before switching projects."
            : "Fix the project settings before switching projects.",
      );
      primaryAction();
      return;
    }
    if (!beginAction()) return;
    try {
      if (dirty) await save();
      await openProject(next);
      setSettings(false);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      finishAction();
    }
  }
  async function create(d: Draft, notes?: string[]) {
    if (!beginAction()) return;
    try {
      if (dirty) {
        if (!valid || !draft?.name.trim())
          throw new Error(
            "Fix the current project settings before creating another project.",
          );
        await save();
      }
      const created = await api.create(d);
      setProjects((items) => [created, ...items]);
      await openProject(created);
      // Code read through its declarations is ready to run; settings open
      // only when the reader left something to fix.
      setSettings(notes ? notes.length > 0 : !d.blueprint && d.script == null);
      if (notes?.length) setNotice(notes[0]);
      setShowNew(false);
    } finally {
      finishAction();
    }
  }

  async function restoreLibrary() {
    if (!beginAction()) return;
    try {
      const added = await api.installLibrary();
      if (added.length) setProjects(await api.projects());
      setNotice(
        added.length
          ? `Added ${added.length} library project${added.length === 1 ? "" : "s"}`
          : "Every library project is already here",
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      finishAction();
    }
  }

  const paletteFolds = useMemo(() => (run ? loopFolds(run.trace) : []), [run]);
  const commands: Command[] = !palette
    ? []
    : [
        {
          id: "run",
          group: "Actions",
          label: needsRunReview ? nextAction.label : "Run",
          detail: needsRunReview
            ? nextAction.detail
            : "Save and record the current code",
          shortcut: "⌘↵",
          disabled: !canRun && !needsRunReview,
          run: () => void execute(),
        },
        {
          id: "check",
          group: "Actions",
          label: "Check shapes",
          detail: "Dry-run the current code without values or saving a run",
          shortcut: "⇧⌘↵",
          disabled: busy || loading || !draft || !!draft.blueprint,
          run: () => void checkShapes(),
        },
        {
          id: "shortcuts",
          group: "Actions",
          label: "Keyboard shortcuts",
          detail: "Every shortcut, grouped by where it works",
          shortcut: "?",
          run: () => setShortcuts(true),
        },
        {
          id: "breakpoints",
          group: "Actions",
          label: "Clear breakpoints",
          detail: "Remove every breakpoint in this project",
          disabled:
            !project ||
            !Object.values(breakpoints[project.id] ?? {}).some(
              (lines) => lines.length,
            ),
          run: () =>
            project && saveBreakpoints({ ...breakpoints, [project.id]: {} }),
        },
        {
          id: "new",
          group: "Actions",
          label: "New project",
          disabled: busy || loading,
          run: () => setShowNew(true),
        },
        {
          id: "library",
          group: "Actions",
          label: "Restore library projects",
          detail: "Add any library project missing from the list",
          disabled: busy || loading,
          run: () => void restoreLibrary(),
        },
        {
          id: "settings",
          group: "Actions",
          label: "Inputs & settings",
          detail: "Inputs, weights, environment, module configuration",
          run: () => openSettings(),
        },
        {
          id: "explorer",
          group: "Actions",
          label: "Projects",
          detail: "Projects, files, and steps",
          run: () => {
            if (side !== "explorer") rememberOpener("side");
            setSide("explorer");
            focusPanel("side");
          },
        },
        {
          id: "history",
          group: "Actions",
          label: "Run history",
          detail: "Reopen or compare saved runs",
          run: () => {
            if (side !== "runs") rememberOpener("side");
            setSide("runs");
            focusPanel("side");
          },
        },
        ...(["problems", "variables", "output"] as const).map((id) => ({
          id: `panel-${id}`,
          group: "Actions" as const,
          label: {
            problems: "Run notes",
            variables: "Tensor shelf",
            output: "Printed output",
          }[id],
          run: () => openShelf(id),
        })),
        {
          id: "link",
          group: "Actions",
          label: "Copy a link to this view",
          run: () => void copyLink(),
        },
        // Loops first among steps: a folded loop opens on its repeats.
        ...[
          ...(run
            ? loopLines(run.trace, paletteFolds, () => true).values()
            : []),
        ].map((loop: LoopLine) => ({
          id: `loop-${loop.id}`,
          group: "Steps" as const,
          label: `↻ ${loop.text}`,
          detail: `${loop.passes} passes · ${loop.folded ? "drawn once" : "passes differ"}`,
          run: () => {
            setSurface("trace");
            select(loop.folded ? `loop:${loop.id}` : loop.firstOperation);
          },
        })),
        ...(run?.trace.operations ?? []).map((op) => {
          const output = run!.trace.tensors[op.outputs[0]];
          // The module path and loop pass are searchable too: "cross_attention", "pass 2".
          const call = op.module.split(" / ").at(-1);
          const pass = op.loops?.at(-1);
          return {
            id: `step-${op.id}`,
            group: "Steps" as const,
            label: `Step ${op.index + 1} · ${kindName(op.kind)}${output && ownName(output.name, op.kind) ? ` → ${output.name}` : ""}`,
            detail: `${output ? `[${output.shape.join(", ")}]` : op.status === "error" ? "failed" : ""}${op.source ? ` · line ${op.source.line}` : ""}${call ? ` · ${call}` : ""}${pass ? ` · pass ${pass.iteration}` : ""}`,
            run: () => {
              setSurface("trace");
              select(op.id);
            },
          };
        }),
        // Module calls: jump to a call's first step.
        ...(run ? journeyStages(run) : []).map((stage) => ({
          id: `call-${stage.id}`,
          group: "Calls" as const,
          label: stage.path,
          detail: `${stage.title} · steps ${stage.start_index + 1}–${stage.end_index}`,
          run: () => {
            setSurface("trace");
            if (stage.operationIds[0]) select(stage.operationIds[0]);
          },
        })),
        ...(run ? variables(run.trace) : [])
          .filter((item) => !item.anonymous)
          .map((item) => ({
            id: `tensor-${item.name}`,
            group: "Tensors" as const,
            label: item.name,
            detail: `[${item.tensor.shape.join(", ")}] ${item.tensor.dtype}`,
            run: () => {
              setSurface("trace");
              select(item.nodeId);
            },
          })),
        ...projects
          .filter((item) => item.id !== project?.id)
          .map((item) => ({
            id: `project-${item.id}`,
            group: "Projects" as const,
            label: item.name,
            disabled: busy || loading,
            detail:
              item.script != null
                ? "console"
                : item.blueprint
                  ? "canvas"
                  : "code",
            run: () => void switchProject(item),
          })),
        ...(isConsole && draft
          ? EXAMPLES.map((example) => ({
              id: `example-${example.id}`,
              group: "Examples" as const,
              label: example.title,
              disabled: busy || loading,
              detail: example.description,
              run: () =>
                setDraft({
                  ...draft,
                  script: example.script,
                  input: exampleInput(draft.input, example),
                }),
            }))
          : []),
      ];

  // The editor shows one source file; its margin is filled from the run.
  const kind = draft ? projectKind(draft) : null;
  const entry = draft ? entryPath(draft) : "model.py";
  const file =
    draft && (activeFile === entry || draft.files?.[activeFile] !== undefined)
      ? activeFile
      : entry;
  const fileText = !draft
    ? ""
    : isConsole
      ? (draft.script ?? "")
      : sourceCode(draft, file);
  const inputsChanged =
    !!run &&
    !!draft &&
    executionContextSignature(draft, file) !==
      executionContextSignature(run.project, file);
  const signature = draft ? draftSignature(draft) : "";
  // A check speaks for the code only while the draft it ran is unchanged.
  const currentCheck =
    check && check.signature === signature ? check.run : null;
  const results = useMemo(() => {
    const recorded = lineResults(
      run,
      fileText,
      inputsChanged,
      isConsole ? null : file,
    );
    return currentCheck && (stale || !run)
      ? withShapeCheck(
          recorded,
          lineResults(currentCheck, fileText, false, isConsole ? null : file),
        )
      : recorded;
  }, [run, fileText, inputsChanged, isConsole, file, currentCheck, stale]);
  const canCheck = !!draft && !draft.blueprint && valid && !!draft.name.trim();
  async function checkShapes() {
    if (!draft || !canCheck || busy) return;
    checkRequest.current?.abort();
    const controller = new AbortController();
    checkRequest.current = controller;
    const checked = draftSignature(draft);
    setChecking(true);
    try {
      const result = await api.shapeCheck(draft, controller.signal);
      if (!controller.signal.aborted)
        setCheck({ run: result, signature: checked });
    } catch (e) {
      // A newer check replaced this one, or a run holds the worker.
      if (!controller.signal.aborted && !liveCheck)
        setError((e as Error).message);
    } finally {
      if (checkRequest.current === controller) {
        checkRequest.current = null;
        setChecking(false);
      }
    }
  }
  const checkShortcut = useRef(checkShapes);
  checkShortcut.current = checkShapes;
  // Opt-in: re-check shapes shortly after the code or inputs stop changing.
  useEffect(() => {
    if (!liveCheck || !canCheck || busy || !(stale || !run) || currentCheck)
      return;
    const timer = window.setTimeout(() => void checkShortcut.current(), 900);
    return () => clearTimeout(timer);
  }, [liveCheck, canCheck, busy, stale, !run, signature, !!currentCheck]);
  useEffect(() => () => checkRequest.current?.abort(), []);
  const activeLine = useMemo(() => {
    for (const [line, result] of results)
      if (result.operations.some((op) => op.id === currentOperation))
        return line;
    return null;
  }, [results, currentOperation]);
  const runVariables = useMemo(() => (run ? variables(run.trace) : []), [run]);
  const peekTensors = useMemo(
    () => new Map(runVariables.map((item) => [item.name, item.tensor])),
    [runVariables],
  );
  // One lineage lookup per trace, shared by every hover in the editor. Inlays
  // can show a shape check's tensors, whose ids repeat the run's, so a tensor
  // is traced only in the trace that holds that very object.
  const lineageFor = useMemo(() => {
    const traces = [run?.trace, currentCheck?.trace].flatMap((trace) =>
      trace ? [{ trace, of: lineageOf(trace) }] : [],
    );
    return (tensor: Tensor) => {
      const found = traces.find(({ trace }) => inTrace(trace, tensor));
      return found ? found.of(tensor.id) : null;
    };
  }, [run, currentCheck]);
  const inkFor = useMemo(
    () => inkOfAll([run?.trace, currentCheck?.trace, compared?.trace]),
    [run, currentCheck, compared],
  );
  const paintFor = useMemo(
    () => cellPaintOfAll([run?.trace, currentCheck?.trace, compared?.trace]),
    [run, currentCheck, compared],
  );
  // The input summary is the ink legend, while it describes the shown run.
  const inputInk = useMemo(() => {
    const input = run?.trace.tensors[run.trace.input_ids?.[0] ?? ""];
    return input && draft && input.shape.join() === draft.input.shape.join()
      ? inkFor(input)
      : null;
  }, [run, draft, inkFor]);
  const knownShapeMap = useMemo(
    () => ({
      ...(draft
        ? Object.fromEntries(
            forwardInputs(draft).map((item) => [item.name, item.input.shape]),
          )
        : {}),
      ...knownShapes(results),
    }),
    [draft, results],
  );
  const cursorSymbol = results.get(cursor?.line ?? activeLine ?? -1)?.output;
  const tensorNames = useMemo(
    () =>
      new Set([
        ...(draft ? forwardInputs(draft).map((item) => item.name) : []),
        ...(run ? variables(run.trace).map((item) => item.name) : []),
      ]),
    [run, draft],
  );
  const problems = useMemo(
    () =>
      collectProblems(run, {
        stale,
        check: stale || !run ? currentCheck : null,
        inputIssue: draft
          ? forwardIssue(forwardInputs(draft), draft.capture_mode) || undefined
          : undefined,
      }).map((problem) =>
        // The console's one file is shown under the name on its tab.
        isConsole && problem.file
          ? { ...problem, file: "console.py" }
          : problem,
      ),
    [run, stale, draft, isConsole, currentCheck],
  );
  // Shape contracts (`# shape: B, T, D`) in the open file, checked against
  // the recorded run or a current shape check.
  const contracts = useMemo(
    () => checkContracts(fileText.split("\n"), results),
    [fileText, results],
  );
  const contractProblems = useMemo(
    () =>
      contracts
        .filter((item) => !item.ok)
        .map((item) => ({
          id: `contract-${item.line}`,
          severity: "warning" as const,
          title: item.text
            ? `Shape contract ${item.text} not met`
            : "Unreadable shape contract",
          detail: item.message,
          file: isConsole ? "console.py" : file,
          line: item.line,
          node: results.get(item.line)?.operations.at(-1)?.id ?? null,
          diagnosis: null,
        })),
    [contracts, isConsole, file, results],
  );
  const contractMarks = useMemo(
    () => new Map(contracts.map((item) => [item.line, item] as const)),
    [contracts],
  );
  const notes = useMemo(
    () => [...problems, ...contractProblems],
    [problems, contractProblems],
  );
  // Steps a run note points at: playback can pause there.
  const warningSteps = useMemo(
    () =>
      new Set(
        notes
          .filter((note) => note.node && note.severity !== "info")
          .map((note) => note.node!),
      ),
    [notes],
  );
  const counts = problemCounts(notes);
  // Tensor-insight warnings for the open file, keyed by line, for the editor margin.
  const lintLines = useMemo(() => {
    const lines = new Map<number, string[]>();
    for (const problem of notes) {
      if (
        problem.severity !== "warning" ||
        !/^(insight|contract)-/.test(problem.id) ||
        problem.line === null ||
        (!isConsole && problem.file !== file)
      )
        continue;
      lines.set(problem.line, [
        ...(lines.get(problem.line) ?? []),
        problem.title,
      ]);
    }
    return lines;
  }, [notes, isConsole, file]);

  const fileBreakpoints = useMemo(
    () => new Set(project ? (breakpoints[project.id]?.[file] ?? []) : []),
    [breakpoints, project?.id, file],
  );
  function toggleBreakpoint(line: number) {
    if (!project) return;
    const forProject = { ...(breakpoints[project.id] ?? {}) };
    const lines = new Set(forProject[file] ?? []);
    if (lines.has(line)) lines.delete(line);
    else lines.add(line);
    forProject[file] = [...lines].sort((a, b) => a - b);
    saveBreakpoints({ ...breakpoints, [project.id]: forProject });
  }
  // The recorded steps that pause playback: those of any breakpoint line.
  const breakpointSteps = useMemo(() => {
    const steps = new Set<string>();
    const lines = project ? breakpoints[project.id] : undefined;
    if (!run || !lines) return steps;
    const runEntry = entryPath(run.project);
    for (const op of run.trace.operations)
      if (
        op.source &&
        lines[op.source.file ?? runEntry]?.includes(op.source.line)
      )
        steps.add(op.id);
    return steps;
  }, [run, breakpoints, project?.id]);
  // Loop headers of the open file, while it still matches the recorded run.
  const loopMarks = useMemo(() => {
    if (!run) return undefined;
    const runEntry = entryPath(run.project);
    const path = (isConsole ? null : file) ?? runEntry;
    if (sourceCode(run.project, path) !== fileText) return undefined;
    return loopLines(
      run.trace,
      loopFolds(run.trace),
      (source) => (source ?? runEntry) === path,
    );
  }, [run, isConsole, file, fileText]);
  // While a loop's repeats play, its header is the line in focus.
  const loopLine = currentLoop
    ? ([...(loopMarks ?? [])].find(
        ([, loop]) => loop.id === currentLoop,
      )?.[0] ?? null)
    : null;
  function selectLoop(loop: LoopLine) {
    setSurface("trace");
    // A folded loop plays its repeats; one shown in full opens its first step.
    select(loop.folded ? `loop:${loop.id}` : loop.firstOperation);
  }
  function selectLine(line: number) {
    const found = results.get(line)?.operations ?? [];
    if (!found.length) return;
    // Repeated clicks step through every operation recorded on this line.
    const position = found.findIndex((op) => op.id === currentOperation);
    setSurface("trace");
    if (position >= 0) {
      select(found[(position + 1) % found.length].id);
      return;
    }
    // Coming from elsewhere, like running to the cursor: the line's next
    // execution after the current step (its loop pass), shown at that
    // execution's last step, the line's result. Without a current step, the
    // line's last step.
    const at =
      run?.trace.operations.find((op) => op.id === currentOperation)?.index ??
      Infinity;
    const next = found.find((op) => op.index > at);
    if (!next) {
      select(found[found.length - 1].id);
      return;
    }
    const pass = (op: (typeof found)[number]) =>
      (op.loops ?? []).map((step) => `${step.id}#${step.iteration}`).join();
    select(found.filter((op) => pass(op) === pass(next)).at(-1)!.id);
  }
  function openFile(path: string) {
    openCode();
    setActiveFile(path);
    setOpenFiles((files) => (files.includes(path) ? files : [...files, path]));
  }
  function editFile(text: string) {
    if (!draft) return;
    setDraft(
      isConsole ? { ...draft, script: text } : updateFile(draft, file, text),
    );
  }
  async function showRun(id: string) {
    if (!beginAction()) return;
    try {
      setRun(await api.getRun(id));
      setSurface("trace");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      finishAction();
    }
  }
  async function compareRun(id: string) {
    if (!beginAction()) return;
    try {
      setCompared(await api.getRun(id));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      finishAction();
    }
  }
  const tabs = openFiles.filter(
    (path) => path === entry || draft?.files?.[path] !== undefined,
  );
  const fileLabel = (path: string) =>
    isConsole ? "console.py" : path.split("/").at(-1)!;

  return (
    <AxisInkContext value={inkFor}>
      <CellPaintContext value={paintFor}>
        <div className="ide tensor-studio">
          {/* Shared paint servers: any SVG in the workbench can burn a selection. */}
          <svg className="tv-defs" aria-hidden="true" focusable="false">
            <defs>
              <linearGradient id="tv-fire" x1="0%" y1="0%" x2="70%" y2="100%">
                <stop offset="0%" stopColor="#fff1a8" />
                <stop offset="45%" stopColor="#ffb347" />
                <stop offset="100%" stopColor="#f0542c" />
              </linearGradient>
            </defs>
          </svg>
          <header className="title-bar">
            <div className="title-brand" aria-label="TensorViewer">
              <TensorMark />
              <span>
                Tensor<span className="brand-light">Viewer</span>
              </span>
            </div>
            <button
              className="command-center"
              disabled={loading}
              onClick={() => setPalette(true)}
              title="Go to a step, tensor, project, or action (Ctrl/⌘ + K)"
            >
              <Search size={13} />
              <span>{draft?.name || "Opening workspace…"}</span>
              <kbd>⌘K</kbd>
            </button>
            <div className="title-actions">
              <div
                className="run-controls"
                role="group"
                aria-label="Run and step"
              >
                <button
                  className="run-button"
                  disabled={!canRun && !needsRunReview}
                  title={
                    canRun
                      ? `${nextAction.detail} (Ctrl/⌘ + Enter)`
                      : nextAction.detail
                  }
                  onClick={primaryAction}
                >
                  {executing ||
                  checkingBeforeRun ||
                  (!!draft && nextAction.kind === "wait") ? (
                    <LoaderCircle size={14} className="spin" />
                  ) : nextAction.kind === "run" ? (
                    <Play size={13} fill="currentColor" />
                  ) : needsRunReview && !valid ? (
                    <CircleAlert size={14} />
                  ) : (
                    <ArrowRight size={14} />
                  )}
                  {checkingBeforeRun
                    ? "Checking model…"
                    : executing
                      ? "Running"
                      : nextAction.kind === "run"
                        ? "Run"
                        : nextAction.label}
                </button>
              </div>
              {project && (
                <button
                  className={`save-state ${dirty ? "dirty" : ""}`}
                  aria-label={dirty ? "Save project" : "All changes saved"}
                  title={dirty ? "Save project changes" : "All changes saved"}
                  disabled={!dirty || busy || !valid || !draft?.name.trim()}
                  onClick={async () => {
                    if (!beginAction()) return;
                    try {
                      await save();
                      setNotice("Project saved");
                    } catch (e) {
                      setError((e as Error).message);
                    } finally {
                      finishAction();
                    }
                  }}
                >
                  {dirty ? "● Save" : "Saved"}
                </button>
              )}
              <div
                className="studio-tools"
                role="group"
                aria-label="Workspace tools"
              >
                <button
                  className="studio-tool"
                  aria-label="Toggle tensor shelf"
                  aria-pressed={panelOpen && panelTab === "variables"}
                  onClick={() => {
                    if (panelOpen && panelTab === "variables")
                      closePanel("shelf");
                    else openShelf("variables");
                  }}
                >
                  <Boxes size={16} />
                  <span>Tensors</span>
                </button>
                <button
                  className="studio-tool"
                  aria-label="Toggle code"
                  aria-pressed={editorOpen}
                  onClick={() =>
                    editorOpen ? closePanel("editor") : openCode()
                  }
                >
                  <Code2 size={16} />
                  <span>Code</span>
                </button>
                <button
                  className="studio-tool"
                  aria-label="New project"
                  disabled={busy || loading}
                  onClick={() => setShowNew(true)}
                >
                  <Plus size={16} />
                  <span>New</span>
                </button>
              </div>
              {project && (
                <button
                  className="icon-button"
                  aria-label="Copy a link to this view"
                  title="Copy a link to this view"
                  onClick={() => void copyLink()}
                >
                  <Link2 size={15} />
                </button>
              )}
            </div>
          </header>
          <div className="ide-body">
            <nav className="activity-bar" aria-label="Views">
              {(
                [
                  ["explorer", Files, "Projects"],
                  ["runs", History, "Run history"],
                ] as const
              ).map(([id, Icon, label]) => (
                <button
                  key={id}
                  className={side === id ? "active" : ""}
                  aria-label={label}
                  aria-pressed={side === id}
                  title={label}
                  onClick={() => {
                    if (side === id) closePanel("side");
                    else {
                      rememberOpener("side");
                      setSide(id);
                      focusPanel("side");
                    }
                  }}
                >
                  <Icon size={19} />
                </button>
              ))}
              <span className="activity-spacer" />
              <button
                className={settings ? "active" : ""}
                aria-label="Inputs and settings"
                aria-pressed={settings}
                title="Inputs and settings"
                disabled={!draft}
                onClick={() =>
                  settings ? closePanel("settings") : openSettings()
                }
              >
                <Settings size={19} />
              </button>
            </nav>
            {side && (
              <aside
                className="side-bar"
                tabIndex={-1}
                aria-label={side === "explorer" ? "Projects" : "Run history"}
                onKeyDown={(event) => panelKeyDown(event, "side")}
              >
                <header className="side-heading">
                  {side === "explorer" ? "Projects" : "Run history"}
                  <button
                    className="icon-button"
                    aria-label="Close side bar"
                    title="Close side bar (Escape)"
                    onClick={() => closePanel("side")}
                  >
                    <X size={14} />
                  </button>
                </header>
                {side === "explorer" ? (
                  <Explorer
                    projects={projects}
                    project={project}
                    draft={draft}
                    run={run}
                    busy={busy || loading}
                    activeFile={file}
                    currentNode={currentOperation}
                    activatedThrough={activated}
                    currentLoop={currentLoop}
                    onProject={(next) => void switchProject(next)}
                    onNewProject={() => setShowNew(true)}
                    onOpenFile={openFile}
                    onChange={setDraft}
                    onSelect={(id) => {
                      setSurface("trace");
                      select(id);
                    }}
                  />
                ) : (
                  <RunsView
                    history={history}
                    run={run}
                    busy={busy}
                    onOpen={(id) => void showRun(id)}
                    onCompare={(id) => void compareRun(id)}
                  />
                )}
              </aside>
            )}
            <main className="workbench">
              {error && (
                <div className="ide-alert" role="alert">
                  <CircleAlert size={15} />
                  <span>{error}</span>
                  <button
                    className="icon-button"
                    aria-label="Dismiss error"
                    onClick={() => setError("")}
                  >
                    <X size={14} />
                  </button>
                </div>
              )}
              {draft && (
                <details
                  className="experiment-disclosure"
                  open={setupOpen || !inputBarValid}
                  onToggle={(event) => {
                    if (!event.currentTarget.open && !inputBarValid) {
                      event.currentTarget.open = true;
                      setSetupOpen(true);
                    } else setSetupOpen(event.currentTarget.open);
                  }}
                >
                  <summary aria-label="Configure starting tensors">
                    <ShapeGlyph shape={draft.input.shape} />
                    <span>Input</span>
                    <code>
                      {draft.input_name ?? "x"}{" "}
                      <InkShape
                        shape={draft.input.shape}
                        ink={inputInk}
                        separator=" × "
                      />
                    </code>
                    {!!draft.additional_inputs?.length && (
                      <span className="experiment-more-inputs">
                        +{" "}
                        {draft.additional_inputs
                          .map((item) => item.name)
                          .join(", ")}
                      </span>
                    )}
                    <small>{draft.input.dtype}</small>
                    <ChevronDown size={13} />
                  </summary>
                  <section
                    className="experiment-strip"
                    aria-label="Experiment setup"
                  >
                    <div className="experiment-input">
                      <InputBar
                        key={project?.id}
                        draft={draft}
                        busy={busy}
                        onChange={setDraft}
                        onSettings={() => openSettings()}
                        onValidity={setInputBarValid}
                      />
                    </div>
                    <div className="experiment-tools">
                      {isConsole && (
                        <>
                          <select
                            className="tab-select"
                            aria-label="Insert an operation"
                            title="Add an operation using the latest variable"
                            value=""
                            disabled={busy}
                            onChange={(event) => {
                              const snippet = SNIPPETS.find(
                                (item) => item.id === event.target.value,
                              );
                              if (snippet)
                                editFile(
                                  appendSnippet(
                                    fileText,
                                    snippet,
                                    draft.input_name ?? "x",
                                    {
                                      [draft.input_name ?? "x"]:
                                        draft.input.shape,
                                      ...knownShapes(results),
                                    },
                                  ),
                                );
                            }}
                          >
                            <option value="">＋ Operation</option>
                            {(
                              ["Shape", "Combine", "Compute", "Memory"] as const
                            ).map((group) => (
                              <optgroup key={group} label={group}>
                                {SNIPPETS.filter(
                                  (item) => item.group === group,
                                ).map((item) => (
                                  <option key={item.id} value={item.id}>
                                    {item.label}
                                  </option>
                                ))}
                              </optgroup>
                            ))}
                          </select>
                          <select
                            className="tab-select"
                            aria-label="Load an example"
                            value=""
                            disabled={busy}
                            onChange={(event) => {
                              const example = EXAMPLES.find(
                                (item) => item.id === event.target.value,
                              );
                              if (example)
                                setDraft({
                                  ...draft,
                                  script: example.script,
                                  input: exampleInput(draft.input, example),
                                });
                            }}
                          >
                            <option value="">Examples</option>
                            {EXAMPLES.map((example) => (
                              <option
                                key={example.id}
                                value={example.id}
                                title={example.description}
                              >
                                {example.title}
                              </option>
                            ))}
                          </select>
                        </>
                      )}
                    </div>
                  </section>
                </details>
              )}
              <div className="workbench-columns">
                <section className="canvas-column" aria-label="Canvas">
                  {(draft?.blueprint || stale) && (
                    <header className="studio-canvas-heading">
                      <div className="studio-canvas-title">
                        <Workflow size={17} />
                        <span>Tensor journey</span>
                      </div>
                      {draft?.blueprint && (
                        <div
                          className="studio-modes"
                          role="tablist"
                          aria-label="Canvas mode"
                        >
                          {draft?.blueprint && (
                            <button
                              role="tab"
                              aria-selected={surface === "build"}
                              onClick={() => {
                                setToolboxGroup(null);
                                setSurface("build");
                              }}
                            >
                              Builder
                            </button>
                          )}
                          <button
                            role="tab"
                            aria-selected={
                              surface === "trace" || !draft?.blueprint
                            }
                            onClick={() => {
                              setToolboxGroup(null);
                              setSurface("trace");
                            }}
                          >
                            Explore
                            {run && (
                              <span
                                className={`status-dot ${run.trace.error ? "error-dot" : ""}`}
                              />
                            )}
                          </button>
                        </div>
                      )}
                      <span className="tab-spacer" />
                      {stale && surface === "trace" && (
                        <span className="tab-note">
                          <History size={12} /> saved run · edited since
                        </span>
                      )}
                    </header>
                  )}
                  <div className="canvas-body">
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
                            inert={showNew}
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
                              onShowRun={() => setSurface("trace")}
                              onInputs={openInputs}
                              active={surface === "build" && !showNew}
                            />
                          </div>
                        )}
                        {(surface === "trace" || !draft?.blueprint) && (
                          <Walkthrough
                            key={`${run?.id}/${viewRevision}`}
                            run={run}
                            busy={busy}
                            recording={executing}
                            stale={stale}
                            active={!settings && !showNew}
                            onInspect={() => setSettings(false)}
                            onEditModel={fixRun}
                            onEditInputs={openInputs}
                            focusOperation={focusOperation}
                            onCurrentOperation={setCurrentOperation}
                            onCurrentLoop={setCurrentLoop}
                            onActivated={setActivated}
                            thread={thread}
                            breakpoints={breakpointSteps}
                            warningSteps={warningSteps}
                            pauseOnWarnings={pauseOnWarnings}
                            onPauseOnWarnings={changePauseOnWarnings}
                            onCell={(node, index) => setCell({ node, index })}
                          />
                        )}
                      </>
                    )}
                  </div>
                  {panelOpen && (
                    <BottomPanel
                      tab={panelTab}
                      onTab={setPanelTab}
                      onClose={() => closePanel("shelf")}
                      run={run}
                      problems={notes}
                      selected={currentOperation}
                      onSelect={(id) => {
                        setSurface("trace");
                        select(id);
                      }}
                      onThread={setThread}
                      onProblem={(problem) => {
                        if (problem.file && problem.file !== file && !isConsole)
                          openFile(problem.file);
                        if (problem.node) {
                          setSurface("trace");
                          select(problem.node);
                        }
                      }}
                    />
                  )}
                </section>
                {draft && editorOpen && (
                  <div
                    className={`column-splitter ${dragging ? "dragging" : ""}`}
                    role="separator"
                    aria-orientation="vertical"
                    aria-label="Resize editor"
                    tabIndex={0}
                    title="Drag to resize; double-click to reset"
                    onPointerDown={(event) => {
                      event.currentTarget.setPointerCapture(event.pointerId);
                      setDragging(true);
                    }}
                    onPointerMove={(event) => {
                      if (!dragging) return;
                      const column = event.currentTarget.nextElementSibling!;
                      resizeEditor(
                        column.getBoundingClientRect().right - event.clientX,
                      );
                    }}
                    onPointerUp={() => setDragging(false)}
                    onPointerCancel={() => setDragging(false)}
                    onDoubleClick={() => {
                      setEditorWidth(null);
                      try {
                        localStorage.removeItem("tensorviewer.editorWidth");
                      } catch {
                        // Nothing was stored.
                      }
                    }}
                    onKeyDown={(event) => {
                      if (
                        event.key !== "ArrowLeft" &&
                        event.key !== "ArrowRight"
                      )
                        return;
                      event.preventDefault();
                      const column = event.currentTarget.nextElementSibling!;
                      resizeEditor(
                        column.getBoundingClientRect().width +
                          (event.key === "ArrowLeft" ? 24 : -24),
                      );
                    }}
                  />
                )}
                {draft && editorOpen && (
                  <section
                    className="editor-column"
                    tabIndex={-1}
                    aria-label="Editor"
                    onKeyDown={(event) => panelKeyDown(event, "editor")}
                    style={
                      editorWidth
                        ? ({
                            "--editor-width": `${editorWidth}px`,
                          } as React.CSSProperties)
                        : undefined
                    }
                  >
                    <div className="source-heading">
                      <span>
                        <Code2 size={15} /> Code
                      </span>
                      <button
                        className="icon-button"
                        aria-label="Close code"
                        title="Close code (Escape)"
                        onClick={() => closePanel("editor")}
                      >
                        <X size={15} />
                      </button>
                    </div>
                    <header className="ide-tabs" role="tablist">
                      {tabs.map((path) => (
                        <div
                          key={path}
                          className="file-tab"
                          role="tab"
                          aria-selected={path === file}
                        >
                          <button
                            onClick={() => setActiveFile(path)}
                            title={path}
                          >
                            {fileLabel(path)}
                            {kind === "canvas" && <small>generated</small>}
                          </button>
                          {path !== entry && (
                            <button
                              className="tab-close"
                              aria-label={`Close ${path}`}
                              onClick={() => {
                                setOpenFiles((files) =>
                                  files.filter((item) => item !== path),
                                );
                                if (path === file) setActiveFile(entry);
                              }}
                            >
                              <X size={12} />
                            </button>
                          )}
                        </div>
                      ))}
                      <span className="tab-spacer" />
                    </header>
                    <div className="breadcrumbs" aria-label="Location">
                      <span>{draft.name}</span>
                      <span>{isConsole ? "console.py" : file}</span>
                      {cursorSymbol && (
                        <span className="crumb-tensor">
                          <ShapeGlyph shape={cursorSymbol.shape} />
                          {cursorSymbol.name}
                          <small>
                            <InkShape
                              shape={cursorSymbol.shape}
                              ink={inkFor(cursorSymbol)}
                            />
                          </small>
                        </span>
                      )}
                    </div>
                    {kind === "canvas" && (
                      <p className="editor-note-bar">
                        Generated from the canvas. Edit components in Builder,
                        or convert the project in Settings to edit this code.
                      </p>
                    )}
                    <CodeEditor
                      key={`${project?.id}/${file}`}
                      label={
                        isConsole ? "Tensor statements" : `Source of ${file}`
                      }
                      value={fileText}
                      readOnly={busy || kind === "canvas"}
                      placeholder={
                        isConsole
                          ? `y = ${draft.input_name ?? "x"}.reshape(-1)`
                          : undefined
                      }
                      results={results}
                      activeLine={loopLine ?? activeLine}
                      tensors={tensorNames}
                      shapes={knownShapeMap}
                      latest={lastVariable(fileText, draft.input_name ?? "x")}
                      peekTensors={peekTensors}
                      lineageFor={lineageFor}
                      lints={lintLines}
                      contracts={contractMarks}
                      loops={loopMarks}
                      breakpoints={fileBreakpoints}
                      // Once playback starts, inlays light up with the canvas;
                      // a fresh run keeps them readable for editing.
                      activatedThrough={
                        surface === "trace" && run && activated >= 0
                          ? activated
                          : undefined
                      }
                      onBreakpoint={toggleBreakpoint}
                      activeLoop={currentLoop}
                      onLoop={selectLoop}
                      onChange={editFile}
                      onRun={() => void execute()}
                      onCheck={() => void checkShapes()}
                      onSelectLine={selectLine}
                      onCursor={(line, column) => setCursor({ line, column })}
                      reveal={
                        editorNavigation &&
                        (isConsole || editorNavigation.file === file)
                          ? editorNavigation
                          : null
                      }
                    />
                  </section>
                )}
                {draft && (
                  <aside
                    className="workspace-drawer editor-drawer"
                    tabIndex={-1}
                    hidden={!settings}
                    aria-label="Inputs and settings"
                    onKeyDown={(event) => panelKeyDown(event, "settings")}
                  >
                    <header className="drawer-heading">
                      <div>
                        <span className="eyebrow">PROJECT</span>
                        <h2>Inputs & settings</h2>
                      </div>
                      <button
                        className="icon-button"
                        aria-label="Close settings"
                        title="Close settings (Escape)"
                        onClick={() => closePanel("settings")}
                      >
                        <X size={18} />
                      </button>
                    </header>
                    <div className="drawer-scroll">
                      {(draft.blueprint || isConsole) && (
                        <div className="generated-code-note">
                          <p>
                            {isConsole
                              ? "The console wraps your statements in a module. Convert it to a custom Python project to edit the whole module, constructor, and source files."
                              : "This project's code is generated from the canvas. Convert it to a custom Python project to edit the code directly."}
                          </p>
                          <button
                            className="secondary-button"
                            disabled={
                              busy ||
                              (!!draft.blueprint && !builderEditingValid)
                            }
                            onClick={() => {
                              setDraft({
                                ...draft,
                                blueprint: null,
                                script: null,
                              });
                              setSurface("trace");
                              openCode();
                            }}
                          >
                            Use as custom code
                          </button>
                          {!!draft.blueprint && !builderEditingValid && (
                            <p className="field-error">
                              Finish or revert the component arguments first.{" "}
                              <button
                                className="text-button"
                                onClick={reviewModel}
                              >
                                Review model
                              </button>
                            </p>
                          )}
                        </div>
                      )}
                      <ProjectEditor
                        key={`${project?.id}/${kind}`}
                        active={settings}
                        draft={draft}
                        onChange={setDraft}
                        onValidity={setValid}
                        busy={busy}
                        readOnly={!!draft.blueprint}
                        reviewRequest={editorReviewRequest}
                        inputReviewRequest={inputReviewRequest}
                        builderReady={!draft.blueprint || builderValid}
                      />
                    </div>
                  </aside>
                )}
              </div>
            </main>
          </div>
          <StatusBar
            draft={draft}
            run={run}
            busy={executing}
            stale={stale}
            errors={counts.errors}
            warnings={counts.warnings}
            currentNode={currentOperation}
            cell={
              cell?.node === currentOperation ? (cell?.index ?? null) : null
            }
            cursor={editorOpen ? cursor : null}
            check={
              !draft || draft.blueprint
                ? null
                : {
                    state: checking
                      ? "checking"
                      : !currentCheck || (run && !stale)
                        ? "none"
                        : needsValues(currentCheck.trace.error)
                          ? "partial"
                          : currentCheck.trace.error
                            ? "failed"
                            : "passed",
                    live: liveCheck,
                    onCheck: () => void checkShapes(),
                    onLive: () => {
                      const next = !liveCheck;
                      setLiveCheck(next);
                      try {
                        localStorage.setItem(
                          "tensorviewer.liveCheck",
                          next ? "on" : "off",
                        );
                      } catch {
                        // The choice still applies for this session.
                      }
                    },
                  }
            }
            onProblems={() => openShelf("problems")}
            onCaptureMode={(mode) =>
              draft && !busy && setDraft({ ...draft, capture_mode: mode })
            }
          />
          {compared && run && (
            <RunCompare
              current={run}
              other={compared}
              onSelect={(id) => {
                setSurface("trace");
                select(id);
              }}
              onClose={() => setCompared(null)}
            />
          )}
          {palette && (
            <CommandPalette
              commands={commands}
              onClose={() => setPalette(false)}
            />
          )}
          {showNew && (
            <NewProject
              onClose={() => setShowNew(false)}
              onCreate={create}
              existing={(name) => projects.find((item) => item.name === name)}
              onOpen={(item) => void switchProject(item)}
            />
          )}
          {shortcuts && <ShortcutsDialog onClose={() => setShortcuts(false)} />}
          {notice && (
            <div className="toast" role="status">
              {notice}
            </div>
          )}
        </div>
      </CellPaintContext>
    </AxisInkContext>
  );
}
