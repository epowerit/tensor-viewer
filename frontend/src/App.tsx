import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRight,
  CircleAlert,
  ChevronDown,
  Code2,
  Boxes,
  Plus,
  Workflow,
  Files,
  FlaskConical,
  History,
  Link2,
  LoaderCircle,
  Play,
  Search,
  Settings,
  X,
} from "lucide-react";
import {
  api,
  toDraft,
  draftSignature,
  isWhatIf,
  type InputEdit,
  type Knockout,
  type LearnStep,
  type Precision,
} from "./api/client";

/** A share as the notices write it: 1.7, 23, 100. */
const percent = (value: number) =>
  value >= 10 ? String(Math.round(value)) : value.toPrecision(2);

/** The value a training step aimed at: `softmax[0, 10, 1]`. */
const learnName = (run: Run, step: LearnStep) => {
  const tensor = run.trace.tensors[step.tensor_id];
  return tensor
    ? `${tensor.name}[${unravel(step.index, tensor.shape).join(", ")}]`
    : "a value";
};
import type { LeftOff } from "./journey/reload";
import { renamePins } from "./console/pins";
import { axisSymbol, symbolicShapes, type Sweep } from "./journey/symbolic";
import { RunHistoryContext } from "./tensors/runTimeline";
import { tokenize, withSentence } from "./inputs/samples";
import { unravel } from "./tensors/coordinates";
import { WhatIfContext, type WhatIfControl } from "./tensors/WhatIf";
import { knockoutText } from "./tensors/knockoutText";
import { useWatchBreaks } from "./shell/watchStore";
import { draftChanges } from "./workspace/runChanges";
import type {
  CompositionPlan,
  Draft,
  LatestRun,
  Project,
  Run,
  RunSummary,
  Tensor,
} from "./api/client";
import {
  Walkthrough,
  type CurrentCard,
  type FoldControls,
} from "./components/Walkthrough";
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
  withoutComment,
} from "./console/script";
import { variables } from "./console/variables";
import { journeyStages } from "./journey/stages";
import { ShortcutsDialog } from "./workspace/ShortcutsDialog";
import { inTrace, lineageOf } from "./tensors/axisLineage";
import { inkOfAll } from "./tensors/axisInk";
import {
  AxisInkContext,
  CellPaintContext,
  InkShape,
  SymbolicContext,
} from "./tensors/InkShape";
import { cellPaintOfAll } from "./tensors/cellPaint";
import { CodeEditor } from "./editor/CodeEditor";
import { checkContracts } from "./editor/contracts";
import {
  acceptRecorded,
  checkValueContracts,
  mergeChecks,
  sumsQuestions,
  type SumsAnswer,
} from "./editor/valueContracts";
import { ContractContext, contractIndex } from "./editor/ContractContext";
import { useDiff, useHeat } from "./tensors/heat";
import {
  SIDE_WIDTH,
  SideResizer,
  useSideView,
  useSideWidth,
} from "./shell/SideResizer";
import { EdgeResizer, useStoredSize } from "./shell/EdgeResizer";
import { ShapeGlyph } from "./editor/ShapeGlyph";
import { BottomPanel, type PanelTab } from "./shell/BottomPanel";
import { Explorer, projectKind } from "./shell/Explorer";
import type { ProjectAction } from "./shell/ProjectMenu";
import { libraryProjectName } from "./components/LibraryPicker";
import { InputBar } from "./shell/InputBar";
import { collectProblems, problemCounts } from "./shell/problems";
import { RunsView } from "./shell/RunsView";
import { StatusBar } from "./shell/StatusBar";
import {
  entryPath,
  projectFiles,
  sourceCode,
  updateFile,
} from "./sources/files";
import { applyFix, fixCandidates, type CodeFix } from "./operations/codeFixes";
import { diagnose, diagnoseError, recordedNames } from "./operations/diagnosis";
import { followLines, lineMap } from "./editor/lineMap";
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
  // A Flow table row under the pointer, traced on the canvas, and the
  // canvas node under the pointer, highlighted in the table.
  const [flowPreview, setFlowPreview] = useState<string | null>(null);
  // The code line under the pointer, traced on the canvas as its step.
  const [codeHover, setCodeHover] = useState<number | null>(null);
  const [canvasHover, setCanvasHover] = useState<string[] | null>(null);
  // Each project's newest run, so lists can mark projects that stopped.
  const [latestRuns, setLatestRuns] = useState<Map<string, LatestRun>>(
    new Map(),
  );
  useEffect(() => {
    let current = true;
    api
      .latestRuns()
      .then(
        (found) =>
          current &&
          setLatestRuns(new Map(found.map((item) => [item.project_id, item]))),
      )
      .catch(() => {});
    return () => {
      current = false;
    };
  }, [projects.length]);
  // The open project's history is always fresh: it updates after each run.
  const lastRunOf = useCallback(
    (id: string) => {
      const newest = id === project?.id ? history[0] : undefined;
      return newest
        ? { failed: newest.failed, operation_count: newest.operation_count }
        : latestRuns.get(id);
    },
    [latestRuns, history, project?.id],
  );
  const [side, setSide, lastSide] = useSideView();
  const [settings, setSettings] = useState(false);
  const [editorOpen, setEditorOpen] = useState(false);
  useEffect(() => {
    if (!editorOpen) setCodeHover(null);
  }, [editorOpen]);
  const [sideWidth, setSideWidth] = useSideWidth();
  const [panelHeight, setPanelHeight] = useStoredSize(
    "tensorviewer.panelHeight",
  );
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
  // The folded call or capsule playback is on, when it plays as one step.
  const [currentCard, setCurrentCard] = useState<CurrentCard | null>(null);
  // The canvas's folding, offered in the command palette.
  const [foldControls, setFoldControls] = useState<FoldControls | null>(null);
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
  // A name does not change what runs: renaming keeps the run current.
  const stale =
    draft && run
      ? draftSignature({ ...draft, name: "" }) !==
        draftSignature({ ...run.project, name: "" })
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
    setCurrentCard(null);
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
    setCurrentCard(null);
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
    // A what-if run is temporary: the address keeps the project's latest.
    run:
      run?.project_id === project?.id && !isWhatIf(run?.id)
        ? run?.id
        : undefined,
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

  // A notice may offer one action, such as showing what a save changed.
  const [noticeAction, setNoticeAction] = useState<{
    label: string;
    act: () => void;
  } | null>(null);
  useEffect(() => {
    if (!notice) {
      setNoticeAction(null);
      return;
    }
    const timer = window.setTimeout(() => setNotice(""), 6000);
    return () => clearTimeout(timer);
  }, [notice]);

  async function save(settings = draft) {
    if (!settings || !project) return null;
    const sent = draftSignature(settings);
    const saved = await api.save(project.id, settings);
    setProject(saved);
    // Typing goes on while a save is on its way; what was typed meanwhile
    // stays in the editor, unsaved, rather than being replaced by the save.
    setDraft((current) =>
      !current || draftSignature(current) === sent ? toDraft(saved) : current,
    );
    setProjects((items) => items.map((p) => (p.id === saved.id ? saved : p)));
    return saved;
  }
  /**
   * Record a run. A save keeps you where you are (the builder, settings);
   * Run shows the new run.
   */
  async function execute({ stay = false } = {}) {
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
      if (!stay) setSettings(false);
      const next = await api.run(saved.id);
      reloading.current = true;
      setRun(next);
      if (!stay) setSurface("trace");
      else if (surface === "build") {
        // Saved from the builder: the diagram redrew behind it, one click away;
        // it says what changed when it opens.
        setNotice("Diagram updated");
        setNoticeAction({ label: "Explore", act: () => setSurface("trace") });
      }
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
  /**
   * Saving keeps the code and the diagram in step, as hot reload does: when
   * the save changes what runs (code, inputs, settings), the model runs again
   * and the diagram redraws from the new run where it was left. A save made
   * while a run is recording waits for it, then runs.
   */
  async function saveAndUpdate() {
    if (!draft || !project || loading) return;
    // A run recording, or a canvas model still being checked after an
    // edit: the save waits for it, then runs.
    if (busy || pendingAction.current || nextAction.kind === "wait") {
      if (dirty || stale || nextAction.kind === "wait") {
        saveWaiting.current = true;
        setQueued(true);
      }
      return;
    }
    const runs = (item: Draft) => draftSignature({ ...item, name: "" });
    if (canRun && (!run || runs(draft) !== runs(run.project))) {
      await execute({ stay: true });
      return;
    }
    if (!dirty || !valid || !draft.name.trim() || !beginAction()) return;
    try {
      await save();
      setNotice("Project saved");
    } catch (e) {
      setError((e as Error).message);
    } finally {
      finishAction();
    }
  }
  // Unsaved edits are marked in the tab's title, as editors do, and leaving
  // the page with them asks first.
  useEffect(() => {
    const base = "TensorViewer — see the transformation";
    document.title = project
      ? `${dirty ? "● " : ""}${project.name} — TensorViewer`
      : base;
    return () => {
      document.title = base;
    };
  }, [project?.name, dirty]);
  useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);
  // Opt-in: save and update shortly after typing pauses, as live reload does.
  // It runs the code on every pause, so it is off until turned on.
  const [autoUpdate, setAutoUpdate] = useState(() => {
    try {
      return localStorage.getItem("tensorviewer.autoUpdate") === "on";
    } catch {
      return false;
    }
  });
  function changeAutoUpdate(on: boolean) {
    setAutoUpdate(on);
    try {
      localStorage.setItem("tensorviewer.autoUpdate", on ? "on" : "off");
    } catch {
      // Private browsing: the choice still applies for this session.
    }
  }
  const draftNow = draft ? draftSignature(draft) : "";
  useEffect(() => {
    if (!autoUpdate || !dirty || !valid || busy) return;
    const timer = window.setTimeout(() => void saveShortcut.current(), 1200);
    return () => clearTimeout(timer);
  }, [autoUpdate, draftNow, dirty, valid, busy]);
  /**
   * Symbolic shapes: the recorded run's shapes in terms of its input's axes,
   * found by checking the same code with each input axis doubled.
   */
  const [symbolic, setSymbolic] = useState<{
    runId: string;
    projectId: string;
    shapes: Map<string, (string | null)[]>;
    symbols: string[];
    shown: boolean;
  } | null>(null);
  const symbolicFor = useMemo(
    () =>
      symbolic && symbolic.shown && symbolic.runId === run?.id
        ? (tensorId: string) => symbolic.shapes.get(tensorId) ?? null
        : null,
    [symbolic, run?.id],
  );
  async function findSymbolicShapes(quiet = false) {
    if (!run || run.project.blueprint || checking) return;
    const recorded = run.project;
    const input = recorded.input;
    const inputTensor = run.trace.input_ids
      .map((id) => run.trace.tensors[id])
      .find((tensor) => tensor?.name === (recorded.input_name ?? "x"));
    if (!inputTensor || input.uploaded) {
      setNotice("Symbolic shapes need a generated or sentence input");
      return;
    }
    setChecking(true);
    if (!quiet) setNotice("Checking the code at other input sizes…");
    try {
      const sweeps: Sweep[] = [];
      const used = new Set<string>();
      // A sentence doubles its tokens; any other input each of its axes.
      const axes = input.text
        ? [inputTensor.shape.length - 1]
        : inputTensor.shape.map((_, axis) => axis);
      for (const axis of axes) {
        let symbol = axisSymbol(inputTensor.axes[axis], axis);
        while (used.has(symbol)) symbol = `${symbol}'`;
        const size = inputTensor.shape[axis];
        const shape = inputTensor.shape.map((each, at) =>
          at === axis ? each * 2 : each,
        );
        const doubled = input.text
          ? withSentence(input, `${input.text} ${input.text}`)
          : { ...input, shape };
        const checked = await api
          .shapeCheck({ ...recorded, input: doubled })
          .catch(() => null);
        // An axis the code cannot grow (a fixed size) keeps its number.
        if (!checked || checked.trace.error) continue;
        used.add(symbol);
        sweeps.push({ symbol, size, trace: checked.trace });
      }
      if (!sweeps.length) {
        setNotice(
          "No input axis could change size here: shapes stay as recorded",
        );
        return;
      }
      const shapes = symbolicShapes(run.trace, sweeps);
      // The input itself reads in its own symbols.
      shapes.set(
        inputTensor.id,
        inputTensor.shape.map(
          (size, axis) =>
            sweeps.find(
              (sweep) =>
                sweep.symbol === axisSymbol(inputTensor.axes[axis], axis),
            )?.symbol ?? `${size}`,
        ),
      );
      const symbols = sweeps.map((sweep) => sweep.symbol);
      setSymbolic({
        runId: run.id,
        projectId: run.project_id,
        shapes,
        symbols,
        shown: true,
      });
      const read = [...shapes.values()].filter((shape) =>
        shape.some((term) => term && /[A-Z]/.test(term)),
      ).length;
      if (!quiet)
        setNotice(
          `Shapes now read in ${symbols.join(", ")}: ${read} of ${shapes.size} tensors follow the input`,
        );
    } finally {
      setChecking(false);
    }
  }
  // A new run of the same project keeps reading in symbols: check again.
  const symbolicStale =
    !!run &&
    !!symbolic?.shown &&
    symbolic.runId !== run.id &&
    symbolic.projectId === run.project_id &&
    !run.project.blueprint &&
    !isWhatIf(run.id);
  useEffect(() => {
    if (symbolicStale && !checking) void findSymbolicShapes(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbolicStale, run?.id, checking]);
  /**
   * A what-if: the recorded run's code again with some input cells set,
   * shown in its place and compared with it, and never saved.
   */
  const [whatIf, setWhatIf] = useState<{
    base: Run;
    edits: InputEdit[];
    /** The dtype the model computed in, when varied. */
    precision: Precision | null;
    /** A training step the weights took first. */
    learn: LearnStep | null;
    /** One step's result replaced as it was made. */
    knockout: Knockout | null;
    /** Whether Diff was on before, to leave it as it was. */
    diff: boolean;
  } | null>(null);
  // Any other run shown ends the what-if.
  useEffect(() => {
    if (whatIf && run && !isWhatIf(run.id)) setWhatIf(null);
  }, [run, whatIf]);
  async function tryWhatIf(change: {
    edit?: InputEdit;
    precision?: Precision | null;
    learn?: LearnStep | null;
    knockout?: Knockout | null;
  }) {
    const base = whatIf && isWhatIf(run?.id) ? whatIf.base : run;
    if (!base || !beginAction()) return;
    const kept = whatIf?.base.id === base.id ? whatIf : null;
    const edits = [
      ...(kept?.edits ?? []).filter(
        (each) => each.index !== change.edit?.index,
      ),
      ...(change.edit ? [change.edit] : []),
    ];
    const precision =
      change.precision !== undefined
        ? change.precision
        : (kept?.precision ?? null);
    const learn =
      change.learn !== undefined ? change.learn : (kept?.learn ?? null);
    const knockout =
      change.knockout !== undefined
        ? change.knockout
        : (kept?.knockout ?? null);
    if (!edits.length && !precision && !learn && !knockout) {
      setNotice("Nothing is varied: that is the recorded run");
      finishAction();
      if (whatIf) leaveWhatIf();
      return;
    }
    try {
      const varied = await api.whatIf(
        base.id,
        edits,
        precision,
        learn,
        knockout,
      );
      setWhatIf({
        base,
        edits,
        precision,
        learn,
        knockout,
        diff: whatIf?.diff ?? diffOn,
      });
      setRun(varied);
      setDiff(true);
      const changed = varied.trace.operations.filter((op, i) => {
        const before = base.trace.operations[i];
        return (
          before &&
          op.outputs.some((id, k) => {
            const now = varied.trace.tensors[id],
              was = base.trace.tensors[before.outputs[k]];
            return (
              !!now &&
              !!was &&
              (now.histogram?.mean !== was.histogram?.mean ||
                now.histogram?.std !== was.histogram?.std)
            );
          })
        );
      }).length;
      // How far the output moved: for a precision, its rounding error.
      const output = varied.trace.tensors[varied.trace.output_ids[0]];
      const exact = base.trace.tensors[base.trace.output_ids[0]];
      let largest = 0,
        scale = 0;
      if (
        output?.values?.length === output?.numel &&
        exact?.values?.length === exact?.numel &&
        output.numel === exact.numel
      )
        output.values.forEach((value, at) => {
          const a = Number(value),
            b = Number(exact.values[at]);
          if (Number.isFinite(a) && Number.isFinite(b)) {
            largest = Math.max(largest, Math.abs(a - b));
            scale = Math.max(scale, Math.abs(b));
          }
        });
      const moved =
        largest && scale
          ? ` The output moved by up to ${largest.toPrecision(2)} (${percent((largest / scale) * 100)}% of its largest value).`
          : "";
      // A training step: how far the value it aimed at moved.
      const aimed = learn
        ? [
            base.trace.tensors[learn.tensor_id]?.values?.[learn.index],
            varied.trace.tensors[learn.tensor_id]?.values?.[learn.index],
          ].map(Number)
        : null;
      const curve = varied.trace.learn_curve ?? [];
      const learned =
        learn?.sentence && curve.length > 1
          ? ` Over ${learn.steps ?? 1} ${learn.steps === 1 ? "step" : "steps"} the next-word loss went ${Number(curve[0]).toPrecision(3)} → ${Number(curve.at(-1)).toPrecision(3)}.`
          : learn && aimed && aimed.every(Number.isFinite)
            ? ` ${learn.steps && learn.steps > 1 ? `${learn.steps} steps` : "One step"} ${learn.direction > 0 ? "up" : "down"} on ${learnName(base, learn)} moved it ${aimed[0].toPrecision(3)} → ${aimed[1].toPrecision(3)}.`
            : "";
      setNotice(
        `What if${knockout ? ` ${knockoutText(base.trace, knockout)}` : ""}${precision ? ` in ${precision}` : ""}${learn ? ` after ${learn.steps && learn.steps > 1 ? `${learn.steps} training steps` : "one training step"}` : ""}: ${changed} of ${varied.trace.operations.length} steps changed.${learned}${moved} Diff and the change lens show where; nothing is saved.`,
      );
    } catch (e) {
      setError((e as Error).message);
    } finally {
      finishAction();
    }
  }
  /** Saves a run as a pytest file, or says why it cannot be one. */
  async function exportPytest(runId: string) {
    try {
      const { name, text } = await api.pytest(runId);
      const url = URL.createObjectURL(
        new Blob([text], { type: "text/x-python" }),
      );
      const link = document.createElement("a");
      link.href = url;
      link.download = name;
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setNotice(
        `Saved ${name}: run it with pytest; it rebuilds this run's inputs and checks every module's shapes and the result's values.`,
      );
    } catch (e) {
      setError((e as Error).message);
    }
  }
  function leaveWhatIf() {
    if (!whatIf) return;
    setRun(whatIf.base);
    setDiff(whatIf.diff);
    setWhatIf(null);
  }
  const whatIfControl = useMemo<WhatIfControl | null>(() => {
    if (!run || run.project.blueprint) return null;
    const input = run.project.input;
    if (input.generator === "uploaded") return null;
    const inputId = run.trace.input_ids.find(
      (id) => run.trace.tensors[id]?.name === (run.project.input_name ?? "x"),
    );
    if (!inputId) return null;
    const base = whatIf && isWhatIf(run.id) ? whatIf.base : run;
    const recordedInput = base.trace.tensors[inputId];
    return {
      inputId,
      vocabulary: input.text ? tokenize(input.text).vocabulary : null,
      edits: isWhatIf(run.id) ? (whatIf?.edits ?? []) : [],
      recorded: (index) => {
        const value = recordedInput?.values?.[index];
        return value === undefined ? undefined : Number(value);
      },
      busy,
      run: (edit) => void tryWhatIf({ edit }),
      learn: (step) => void tryWhatIf({ learn: step }),
      learned:
        isWhatIf(run.id) && whatIf?.learn && run.trace.learn_curve
          ? { step: whatIf.learn, curve: run.trace.learn_curve }
          : null,
      knockout: (knockout) => void tryWhatIf({ knockout }),
      knocked: isWhatIf(run.id) ? (whatIf?.knockout ?? null) : null,
      // A patch comes from the run recorded before the one varied.
      patchFrom:
        history[history.findIndex((item) => item.id === base.id) + 1]?.id ??
        null,
      baseRunId: base.id,
      sweep: async (sweep) => {
        if (!beginAction()) return null;
        try {
          return await api.knockoutSweep(base.id, sweep);
        } catch (e) {
          setError((e as Error).message);
          return null;
        } finally {
          finishAction();
        }
      },
      leave: isWhatIf(run.id) ? leaveWhatIf : null,
    };
    // tryWhatIf and leaveWhatIf read state through setters and refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, whatIf, busy, history]);
  const runHistory = useMemo(
    () =>
      run && history.length > 1
        ? { run, history, compare: (id: string) => void compareRun(id) }
        : null,
    // compareRun only reads state through its setters.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [run, history],
  );
  /**
   * Fixes for a failed run: one-line edits, each checked by a shapes-only run
   * of the edited code, offered only when the run then gets further.
   */
  const fixesFor = useRef<string | null>(null);
  useEffect(() => {
    if (run?.id !== fixesFor.current) fixesFor.current = null;
  }, [run?.id]);
  const [runFixes, setRunFixes] = useState<{
    runId: string;
    checking: boolean;
    fixes: (CodeFix & { verdict: string })[];
  } | null>(null);
  const fixable =
    !!run?.trace.error &&
    !busy &&
    !!draft &&
    !draft.blueprint &&
    draft.script == null &&
    runFixes?.runId !== run.id;
  useEffect(() => {
    if (!fixable || !run || !draft) return;
    const failed = run.trace.operations.find((op) => op.status === "error");
    const diagnosis = failed
      ? diagnose(failed, run.trace.tensors, run.trace.operations)
      : diagnoseError(run.trace.error!, recordedNames(run.trace));
    const recorded = projectFiles(run.project);
    const current = projectFiles(draft);
    // Only fixes to files as they were when the run was recorded.
    const candidates = fixCandidates(run, diagnosis).filter(
      (fix) => recorded[fix.file] === current[fix.file],
    );
    setRunFixes({ runId: run.id, checking: !!candidates.length, fixes: [] });
    if (!candidates.length) return;
    // Checking goes on until another run is shown.
    fixesFor.current = run.id;
    const live = () => fixesFor.current === run.id;
    const reached = run.trace.operations.filter(
      (op) => op.status === "ok",
    ).length;
    // Fixes that get furthest come first; one that runs through, first of all.
    const further = new Map<string, number>();
    const keyOf = (fix: CodeFix) => `${fix.file}:${fix.line}:${fix.after}`;
    (async () => {
      const verified: (CodeFix & { verdict: string })[] = [];
      for (const fix of candidates) {
        const code = applyFix(current[fix.file], fix);
        if (code === null) continue;
        const checked = await api
          .shapeCheck(updateFile(draft, fix.file, code))
          .catch(() => null);
        if (!live()) return;
        if (!checked) continue;
        const error = checked.trace.error;
        const through = checked.trace.operations.filter(
          (op) => op.status === "ok",
        ).length;
        further.set(keyOf(fix), error ? through : Infinity);
        if (!error)
          verified.push({ ...fix, verdict: "shapes check through the end" });
        else if (through > reached)
          verified.push({
            ...fix,
            verdict: `gets ${through - reached} ${through - reached === 1 ? "step" : "steps"} further, then stops${error.line ? ` at line ${error.line}` : ""}`,
          });
      }
      verified.sort(
        (a, b) => (further.get(keyOf(b)) ?? 0) - (further.get(keyOf(a)) ?? 0),
      );
      if (live())
        setRunFixes({ runId: run.id, checking: false, fixes: verified });
    })();
    // `fixable` changes when the run does.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fixable, run?.id]);
  const runAfterFix = useRef(false);
  function applyRunFix(fix: CodeFix) {
    if (!draft) return;
    const code = applyFix(projectFiles(draft)[fix.file] ?? "", fix);
    if (code === null) {
      setNotice("That line has changed since the run; edit it in the code");
      return;
    }
    runAfterFix.current = true;
    setDraft(updateFile(draft, fix.file, code));
    setNotice(`Applied: ${fix.label} on line ${fix.line}. Running again…`);
  }
  useEffect(() => {
    if (!runAfterFix.current) return;
    runAfterFix.current = false;
    void saveAndUpdate();
    // Runs once the applied fix is in the draft.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft]);
  const saveWaiting = useRef(false);
  const [queued, setQueued] = useState(false);
  // What the update that recorded the run on screen changed, kept for the
  // palette after its note has gone.
  const [lastChange, setLastChange] = useState<{
    runId: string;
    summary: string;
    first: string | null;
    changed: ReadonlySet<string>;
  } | null>(null);
  // What the last update changed, marked in the outline for a moment.
  const [changedSteps, setChangedSteps] = useState<ReadonlySet<string> | null>(
    null,
  );
  useEffect(() => {
    if (!changedSteps) return;
    const timer = window.setTimeout(() => setChangedSteps(null), 2600);
    return () => clearTimeout(timer);
  }, [changedSteps]);
  // The diagram says what a run changed once it opens.
  const reloading = useRef(false);
  const leftOff = useRef<LeftOff | null>(null);
  const saveShortcut = useRef(saveAndUpdate);
  saveShortcut.current = saveAndUpdate;
  useEffect(() => {
    if (busy || nextAction.kind === "wait" || !saveWaiting.current) return;
    saveWaiting.current = false;
    setQueued(false);
    void saveShortcut.current();
  }, [busy, nextAction.kind]);
  // Run from anywhere in the workspace, as in a code editor.
  const runShortcut = useRef(execute);
  runShortcut.current = execute;
  // ⌘/Ctrl+B side bar, J tensor shelf, E code: open, or close when open.
  const toggleShortcut = useRef((key: "b" | "j" | "e") => {
    void key;
  });
  toggleShortcut.current = (key) => {
    if (key === "b") {
      if (side) closePanel("side");
      else {
        rememberOpener("side");
        setSide(lastSide);
        focusPanel("side");
      }
    } else if (key === "j") {
      if (panelOpen) closePanel("shelf");
      else openShelf(panelTab);
    } else if (editorOpen) closePanel("editor");
    else openCode();
  };
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
      const toggle = event.key.toLowerCase();
      if (
        (toggle === "b" || toggle === "j" || toggle === "e") &&
        (event.metaKey || event.ctrlKey) &&
        !event.shiftKey &&
        !event.altKey &&
        !document.querySelector("dialog[open]")
      ) {
        event.preventDefault();
        toggleShortcut.current(toggle);
        return;
      }
      // Ctrl/⌘ + S saves and updates the diagram, from anywhere.
      // Never the browser's Save page; a dialog's edits wait until it closes.
      if (
        event.key.toLowerCase() === "s" &&
        (event.metaKey || event.ctrlKey) &&
        !event.shiftKey &&
        !event.altKey
      ) {
        event.preventDefault();
        if (!document.querySelector("dialog[open]"))
          void saveShortcut.current();
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
  /**
   * Rename, duplicate or delete a project from the side bar or the palette.
   * Renaming keeps any unsaved edits of the open project; deleting waits a
   * few seconds with Undo before anything is removed.
   */
  const pendingDeletes = useRef(
    new Map<string, { timer: number; url: string }>(),
  );
  const [renameRequest, setRenameRequest] = useState<{
    id: string;
    key: number;
  } | null>(null);
  function requestRename(id: string) {
    if (side !== "explorer") setSide("explorer");
    setRenameRequest((previous) => ({ id, key: (previous?.key ?? 0) + 1 }));
  }
  async function renameProject(item: Project, name: string) {
    try {
      const saved = await api.save(item.id, { ...toDraft(item), name });
      setProjects((items) =>
        items.map((other) => (other.id === saved.id ? saved : other)),
      );
      if (item.id === project?.id) {
        setProject(saved);
        setDraft((current) => (current ? { ...current, name } : current));
      }
      setNotice(`Renamed to “${name}”`);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  async function duplicateProject(item: Project) {
    if (busy || pendingAction.current) return;
    const names = new Set(projects.map((other) => other.name));
    let name = `${item.name} copy`;
    for (let n = 2; names.has(name); n++) name = `${item.name} copy ${n}`;
    try {
      const created = await api.create({ ...toDraft(item), name });
      setProjects((items) => [created, ...items]);
      setNotice(`Duplicated as “${name}”`);
      await switchProject(created);
    } catch (e) {
      setError((e as Error).message);
    }
  }
  /**
   * A delete that waits a few seconds with Undo before it happens. Leaving
   * the page carries out the ones still waiting.
   */
  function deleteLater({
    key,
    url,
    label,
    perform,
    restore,
  }: {
    key: string;
    url: string;
    label: string;
    perform: () => Promise<unknown>;
    restore: () => void;
  }) {
    const timer = window.setTimeout(async () => {
      pendingDeletes.current.delete(key);
      try {
        await perform();
      } catch (e) {
        restore();
        setError(`Could not delete ${label}: ${(e as Error).message}`);
      }
    }, 6500);
    pendingDeletes.current.set(key, { timer, url });
    setNotice(`Deleted ${label}`);
    setNoticeAction({
      label: "Undo",
      act: () => {
        window.clearTimeout(timer);
        pendingDeletes.current.delete(key);
        restore();
        setNotice("");
      },
    });
  }
  useEffect(() => {
    const flush = () => {
      for (const { timer, url } of pendingDeletes.current.values()) {
        window.clearTimeout(timer);
        void fetch(url, { method: "DELETE", keepalive: true });
      }
      pendingDeletes.current.clear();
    };
    window.addEventListener("pagehide", flush);
    return () => window.removeEventListener("pagehide", flush);
  }, []);
  function deleteProject(item: Project) {
    if (projects.length <= 1 || pendingDeletes.current.has(item.id)) return;
    const at = projects.findIndex((other) => other.id === item.id);
    const remaining = projects.filter((other) => other.id !== item.id);
    const wasOpen = item.id === project?.id;
    const name = wasOpen ? draft?.name || item.name : item.name;
    setProjects(remaining);
    // The project beside it opens in its place.
    if (wasOpen)
      void openProject(remaining[Math.min(at, remaining.length - 1)]);
    deleteLater({
      key: item.id,
      url: `/api/v1/projects/${item.id}`,
      label: `“${name}”`,
      perform: () => api.deleteProject(item.id),
      restore: () => {
        setProjects((items) => {
          const restored = items.filter((other) => other.id !== item.id);
          restored.splice(Math.min(at, restored.length), 0, item);
          return restored;
        });
        if (wasOpen) void openProject(item);
      },
    });
  }
  // The open project's history as the server has it, after an Undo.
  const projectRef = useRef<string | null>(null);
  projectRef.current = project?.id ?? null;
  function reloadHistory(projectId: string) {
    void api
      .runs(projectId)
      .then((runs) => {
        if (projectRef.current === projectId) setHistory(runs);
      })
      .catch(() => {});
  }
  const runTime = (item: RunSummary) =>
    new Date(item.created_at).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  /** Delete one run; the run on screen gives way to the one beside it. */
  function deleteRun(id: string) {
    if (!project || pendingDeletes.current.has(`run:${id}`)) return;
    const at = history.findIndex((item) => item.id === id);
    if (at < 0) return;
    const projectId = project.id;
    const remaining = history.filter((item) => item.id !== id);
    const wasShown = run?.id === id;
    setHistory(remaining);
    if (wasShown) {
      const next = remaining[Math.min(at, remaining.length - 1)];
      if (next) void showRun(next.id);
      else setRun(null);
    }
    deleteLater({
      key: `run:${id}`,
      url: `/api/v1/runs/${id}`,
      label: `the run of ${runTime(history[at])}`,
      perform: () => api.deleteRun(id),
      restore: () => {
        reloadHistory(projectId);
        if (wasShown && projectRef.current === projectId) void showRun(id);
      },
    });
  }
  /** Keep the newest run and clear the history before it. */
  function clearOlderRuns() {
    if (!project || history.length < 2) return;
    const projectId = project.id;
    const latest = history[0];
    setHistory([latest]);
    if (run && run.project_id === projectId && run.id !== latest.id)
      void showRun(latest.id);
    deleteLater({
      key: `older:${projectId}`,
      url: `/api/v1/projects/${projectId}/runs?before=${encodeURIComponent(latest.id)}`,
      label: "the older runs",
      perform: () => api.deleteRunsBefore(projectId, latest.id),
      restore: () => reloadHistory(projectId),
    });
  }
  /**
   * A library model edited since it was installed goes back to the library's
   * original: its code, inputs and settings. Undo restores what was saved.
   */
  async function resetProject(item: Project) {
    try {
      const entry = (await api.library()).find(
        (found) => libraryProjectName(found) === item.name,
      );
      if (!entry) return;
      const original = {
        ...(await api.readSource(entry.code, item.name)).draft,
        name: item.name,
      };
      const open = item.id === project?.id;
      if (
        draftSignature(original) === draftSignature(item) &&
        !(open && dirty)
      ) {
        setNotice(`“${item.name}” is already the library original`);
        return;
      }
      const before = toDraft(item);
      const apply = async (next: Draft) => {
        const saved = await api.save(item.id, next);
        setProjects((items) =>
          items.map((other) => (other.id === saved.id ? saved : other)),
        );
        if (projectRef.current === saved.id) {
          setProject(saved);
          setDraft(toDraft(saved));
        }
      };
      await apply(original);
      setNotice(`Reset “${item.name}” to the library original`);
      setNoticeAction({
        label: "Undo",
        act: () => {
          setNotice("");
          void apply(before).catch((e) => setError((e as Error).message));
        },
      });
    } catch (e) {
      setError((e as Error).message);
    }
  }
  function manageProject(item: Project, action: ProjectAction, name?: string) {
    if (action === "rename" && name) void renameProject(item, name);
    else if (action === "rename") requestRename(item.id);
    else if (action === "duplicate") void duplicateProject(item);
    else if (action === "reset") void resetProject(item);
    else if (action === "delete") deleteProject(item);
  }
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
  // The run recorded before the displayed one, for every comparison.
  // A what-if run compares with the recorded run it varies.
  const previousRunId =
    whatIf && isWhatIf(run?.id)
      ? whatIf.base.id
      : (history[history.findIndex((item) => item.id === run?.id) + 1]?.id ??
        null);
  // Tensor views shared by every grid, toggled from the palette too.
  const [heat, setHeat] = useHeat();
  const [diffOn, setDiff] = useDiff();
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
          detail:
            run && !stale
              ? "The recorded run matches the code; its shapes are current"
              : "Dry-run the current code without values or saving a run",
          shortcut: "⇧⌘↵",
          disabled:
            busy || loading || !draft || !!draft.blueprint || (!!run && !stale),
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
          detail: "Write code, import a repository, or start from the library",
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
          shortcut: "⌘B",
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
        ...(
          [
            "problems",
            "variables",
            "flow",
            "watch",
            "weights",
            "lens",
            "output",
          ] as const
        ).map((id) => ({
          id: `panel-${id}`,
          group: "Actions" as const,
          label: {
            problems: "Run notes",
            variables: "Tensor shelf",
            flow: "Flow table",
            watch: "Watch expressions",
            weights: "Weights and their spectra",
            lens: "Logit lens",
            output: "Printed output",
          }[id],
          detail: {
            problems: "Errors, insights, and contract checks",
            variables: "Every named tensor at the playback position",
            flow: "Every step's shape and values, in order",
            watch: "Python over the run's tensors at the playback position",
            weights: "Every weight's norm, singular values, condition and rank",
            lens: "What each layer of a language model would predict",
            output: "What the run printed",
          }[id],
          ...(id === "variables" ? { shortcut: "⌘J" } : {}),
          run: () => openShelf(id),
        })),
        ...(project
          ? [
              {
                id: "project-rename",
                group: "Actions" as const,
                label: "Rename project",
                detail: `“${draft?.name || project.name}”, in the side bar (F2 on its row)`,
                run: () => requestRename(project.id),
              },
              {
                id: "project-duplicate",
                group: "Actions" as const,
                label: "Duplicate project",
                detail: "A copy of its saved code, inputs and settings, opened",
                run: () => void duplicateProject(project),
              },
              {
                id: "project-delete",
                group: "Actions" as const,
                label: "Delete project",
                detail:
                  projects.length > 1
                    ? "With its runs; Undo for a few seconds"
                    : "The workspace keeps at least one project",
                disabled: projects.length <= 1,
                run: () => deleteProject(project),
              },
            ]
          : []),
        {
          id: "code",
          group: "Actions",
          label: editorOpen ? "Hide code" : "Show code",
          detail: "The project's source files beside the canvas",
          shortcut: "⌘E",
          run: () => (editorOpen ? closePanel("editor") : openCode()),
        },
        ...(run && !run.project.blueprint
          ? (["bfloat16", "float16"] as const).map((precision) => ({
              id: `what-if-${precision}`,
              group: "Actions" as const,
              label:
                whatIf?.precision === precision && isWhatIf(run.id)
                  ? `Back to the input's own precision`
                  : `What if this ran in ${precision}?`,
              detail:
                "Run the same code in that dtype; Diff and the change lens show each step's rounding error",
              disabled: busy,
              run: () =>
                void tryWhatIf({
                  precision:
                    whatIf?.precision === precision && isWhatIf(run.id)
                      ? null
                      : precision,
                }),
            }))
          : []),
        ...(run && !isWhatIf(run.id) && !run.trace.error
          ? [
              {
                id: "export-pytest",
                group: "Actions" as const,
                label: "Save this run as a pytest file",
                detail:
                  "A test that rebuilds the inputs and checks every module's shapes and the result",
                run: () => void exportPytest(run.id),
              },
            ]
          : []),
        ...(whatIf && isWhatIf(run?.id)
          ? [
              {
                id: "what-if-leave",
                group: "Actions" as const,
                label: "Back to the recorded run",
                detail: "Leave the what-if; it was never saved",
                run: leaveWhatIf,
              },
            ]
          : []),
        ...(run && !run.project.blueprint
          ? [
              symbolic?.runId === run.id
                ? {
                    id: "symbolic-shapes",
                    group: "Actions" as const,
                    label: symbolic.shown
                      ? "Show recorded sizes"
                      : `Show shapes in ${symbolic.symbols.join(", ")}`,
                    detail: symbolic.shown
                      ? "[1, 13, 48] rather than [B, T, 48]"
                      : "Every shape in terms of the input's axes",
                    run: () =>
                      setSymbolic((current) =>
                        current
                          ? { ...current, shown: !current.shown }
                          : current,
                      ),
                  }
                : {
                    id: "symbolic-shapes",
                    group: "Actions" as const,
                    label: "Find symbolic shapes",
                    detail:
                      "Check the code at other input sizes; shapes then read [B, T, 48]",
                    disabled: busy || checking,
                    run: () => void findSymbolicShapes(),
                  },
            ]
          : []),
        {
          id: "auto-update",
          group: "Actions",
          label: autoUpdate
            ? "Stop updating the diagram as you type"
            : "Update the diagram as you type",
          detail: autoUpdate
            ? "Ctrl/⌘ + S still saves and updates it"
            : "Each pause in typing saves and runs your code; for code you trust",
          run: () => changeAutoUpdate(!autoUpdate),
        },
        ...(lastChange && lastChange.runId === run?.id
          ? [
              {
                id: "last-change",
                group: "Actions" as const,
                label: "Show what the last save changed",
                detail: lastChange.summary,
                run: () => {
                  const { first, changed } = lastChange;
                  setSurface("trace");
                  if (first) select(first);
                  // Light it again, everywhere it shows.
                  setChangedSteps(new Set(changed));
                },
              },
            ]
          : []),
        ...(foldControls
          ? [
              {
                id: "fold-here",
                group: "Actions" as const,
                label: "Fold the call around the current step",
                detail: "Into one card, played as one step",
                shortcut: "⌘⌥[",
                disabled: !foldControls.canFold,
                run: foldControls.fold,
              },
              {
                id: "unfold-here",
                group: "Actions" as const,
                label: "Unfold the card on screen",
                detail: "One level, onto its first step",
                shortcut: "⌘⌥]",
                disabled: !foldControls.canUnfold,
                run: foldControls.unfold,
              },
              ...foldControls.levels.map((level, i) => ({
                id: `detail-${i}`,
                group: "Actions" as const,
                label: `Detail: ${level.label}${level.current ? " (current)" : ""}`,
                detail: level.detail,
                run: level.choose,
              })),
            ]
          : []),
        {
          id: "shelf-maximize",
          group: "Actions",
          label: "Maximize tensor shelf",
          detail: "The shelf takes all but a strip of canvas",
          run: () => {
            setPanelHeight(window.innerHeight);
            openShelf(panelTab);
          },
        },
        {
          id: "layout-reset",
          group: "Actions",
          label: "Reset layout",
          detail: "Side bar, tensor shelf, and code back to their usual sizes",
          run: () => {
            setSideWidth(SIDE_WIDTH.initial);
            setPanelHeight(null);
            setEditorWidth(null);
            try {
              localStorage.removeItem("tensorviewer.editorWidth");
            } catch {
              // Nothing was stored.
            }
          },
        },
        {
          id: "link",
          group: "Actions",
          label: "Copy a link to this view",
          detail: "The project, run, and step on screen",
          run: () => void copyLink(),
        },
        {
          id: "heat",
          group: "Actions",
          label: heat ? "Stop heat shading" : "Heat shading",
          detail: "Shade every tensor grid's cells by value",
          run: () => setHeat(!heat),
        },
        {
          id: "diff",
          group: "Actions",
          label: diffOn
            ? "Stop showing changes"
            : "Show changes since the run before",
          detail: previousRunId
            ? "Every grid and the shelf compare with the previous run"
            : "This project has no earlier run",
          disabled: !previousRunId,
          run: () => setDiff(!diffOn),
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
  // What the next save would record that the run on screen did not: "model.py
  // · 2 lines", "input tokens".
  const pendingEdits = useMemo(
    () => (stale && run && draft ? draftChanges(run.project, draft) : []),
    [stale, run, draft],
  );
  // What the last update changed, as lines of the file in the editor.
  const changedLines = useMemo(() => {
    if (!changedSteps || !run) return null;
    const runEntry = entryPath(run.project);
    return new Set(
      run.trace.operations
        .filter(
          (op) =>
            changedSteps.has(op.id) &&
            op.source &&
            (isConsole || (op.source.file ?? runEntry) === file),
        )
        .map((op) => op.source!.line),
    );
  }, [changedSteps, run, file, isConsole]);
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
  // The card under the pointer on the canvas, as the lines that wrote it.
  const canvasHoverLines = useMemo(() => {
    if (!canvasHover?.length) return null;
    const hovered = new Set(canvasHover);
    const lines = new Set<number>();
    for (const [line, result] of results)
      if (result.operations.some((op) => hovered.has(op.id))) lines.add(line);
    return lines;
  }, [canvasHover, results]);
  // A line's last recorded step stands for it: the tensor the line produced.
  const codePreview =
    (codeHover !== null && results.get(codeHover)?.operations.at(-1)?.id) ||
    null;
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
  // Where a line of the run's code is now, in code edited since the run:
  // notes, loop chips, and breakpoints read the run's lines through it.
  const nowLine = useMemo(() => {
    const maps = new Map<string, (line: number) => number | null>();
    return (path: string | null, line: number): number | null => {
      if (!run || !draft) return line;
      const key = path ?? entryPath(run.project);
      if (!maps.has(key))
        maps.set(
          key,
          run.project.script != null
            ? lineMap(run.project.script, draft.script ?? "")
            : lineMap(sourceCode(run.project, key), sourceCode(draft, key)),
        );
      return maps.get(key)!(line);
    };
  }, [run, draft]);
  const problems = useMemo(
    () =>
      collectProblems(run, {
        stale,
        place: stale ? nowLine : undefined,
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
    [run, stale, draft, isConsole, currentCheck, nowLine],
  );
  // Shape contracts (`# shape: B, T, D`) in the open file, checked against
  // the recorded run or a current shape check.
  const contracts = useMemo(
    () => checkContracts(fileText.split("\n"), results),
    [fileText, results],
  );
  // Value contracts (`# range: 0..1; sums(-1): 1`), checked against the
  // recorded values.
  // Sums contracts on tensors kept in snapshots are added up by the backend.
  const [sumsAnswers, setSumsAnswers] = useState<Map<string, SumsAnswer>>(
    new Map(),
  );
  const sumsAsked = useMemo(
    () => sumsQuestions(fileText.split("\n"), results),
    [fileText, results],
  );
  useEffect(() => {
    if (!run) return;
    const missing = sumsAsked.filter(
      (question) => !sumsAnswers.has(`${run.id}|${question.key}`),
    );
    if (!missing.length) return;
    let current = true;
    Promise.all(
      missing.map((question) =>
        api
          .tensorQuery<SumsAnswer>(run.id, question.tensorId, "sums", {
            axis: question.axis,
            value: question.value,
          })
          .then((answer) => [`${run.id}|${question.key}`, answer] as const)
          .catch(() => null),
      ),
    ).then((found) => {
      if (!current) return;
      setSumsAnswers((previous) => {
        const next = new Map(previous);
        for (const pair of found) if (pair) next.set(pair[0], pair[1]);
        return next;
      });
    });
    return () => {
      current = false;
    };
  }, [run, sumsAsked, sumsAnswers]);
  const valueContracts = useMemo(() => {
    // Answers are kept per run; the check reads this run's.
    const prefix = run ? `${run.id}|` : "";
    const answers = new Map(
      [...sumsAnswers]
        .filter(([key]) => key.startsWith(prefix))
        .map(([key, answer]) => [key.slice(prefix.length), answer]),
    );
    return checkValueContracts(fileText.split("\n"), results, answers);
  }, [fileText, results, sumsAnswers, run]);
  // Accept what the run recorded for a broken value contract, like a new
  // snapshot: the line's broken bounds are redrawn around the recording.
  const acceptContract = useRef<(line: number) => void>(() => {});
  acceptContract.current = (line) => {
    const lines = fileText.split("\n");
    const tensor = results.get(line)?.output;
    const rewritten = tensor && acceptRecorded(lines[line - 1] ?? "", tensor);
    if (!rewritten) return;
    lines[line - 1] = rewritten;
    editFile(lines.join("\n"));
  };
  const contractProblems = useMemo(
    () =>
      [
        ...contracts.map((item) => ({ item, kind: "Shape" })),
        ...valueContracts.map((item) => ({ item, kind: "Value" })),
      ]
        .filter(({ item }) => !item.ok)
        .map(({ item, kind }) => ({
          id: `${kind.toLowerCase()}-contract-${item.line}`,
          severity: "warning" as const,
          title: item.text
            ? `${kind} contract ${item.text} not met`
            : `Unreadable ${kind.toLowerCase()} contract`,
          detail: item.message,
          file: isConsole ? "console.py" : file,
          line: item.line,
          node: results.get(item.line)?.operations.at(-1)?.id ?? null,
          diagnosis: null,
          fix:
            kind === "Value" && item.text
              ? {
                  label: "Accept recorded values",
                  apply: () => acceptContract.current(item.line),
                }
              : undefined,
        })),
    [contracts, valueContracts, isConsole, file, results],
  );
  const contractMarks = useMemo(
    () => mergeChecks(contracts, valueContracts),
    [contracts, valueContracts],
  );
  const contractsByTensor = useMemo(
    () => ({
      ...contractIndex(contractMarks, results),
      accept: (line: number) => acceptContract.current(line),
    }),
    [contractMarks, results],
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
  // Code edited since the run moved its lines; each recorded line is read
  // where it is now.
  // Steps where a watch set to pause holds, even with its tab closed.
  const watchBreaks = useWatchBreaks(run, project?.id ?? null);
  const breakpointSteps = useMemo(() => {
    // Steps where a pausing watch holds pause playback too.
    const steps = new Set<string>(watchBreaks);
    const lines = project ? breakpoints[project.id] : undefined;
    if (!run || !lines) return steps;
    const runEntry = entryPath(run.project);
    for (const op of run.trace.operations) {
      if (!op.source) continue;
      const path = op.source.file ?? runEntry;
      const line = lines[path]?.length ? nowLine(path, op.source.line) : null;
      if (line !== null && lines[path]?.includes(line)) steps.add(op.id);
    }
    return steps;
  }, [run, breakpoints, project?.id, nowLine, watchBreaks]);
  // Loop headers of the open file, while it still matches the recorded run.
  const loopMarks = useMemo(() => {
    if (!run) return undefined;
    const runEntry = entryPath(run.project);
    const path = (isConsole ? null : file) ?? runEntry;
    const recorded = loopLines(
      run.trace,
      loopFolds(run.trace),
      (source) => (source ?? runEntry) === path,
    );
    // A header keeps its chip where its code is now, while it still reads
    // the same; an edited loop waits for the next run.
    const current = fileText.split("\n");
    const marks = new Map<number, LoopLine>();
    for (const [line, loop] of recorded) {
      const now = nowLine(path, line);
      const header = withoutComment(current[(now ?? 0) - 1] ?? "")
        .trim()
        .replace(/:$/, "");
      if (now !== null && header === loop.text.trim()) marks.set(now, loop);
    }
    return marks;
  }, [run, isConsole, file, fileText, nowLine]);
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
    // Breakpoints stay on their code as lines are added or removed above it.
    const marks = project ? breakpoints[project.id]?.[file] : undefined;
    if (project && marks?.length) {
      const moved = followLines(marks, fileText, text);
      if (moved.join() !== marks.join())
        saveBreakpoints({
          ...breakpoints,
          [project.id]: { ...breakpoints[project.id], [file]: moved },
        });
    }
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
      <SymbolicContext value={symbolicFor}>
        <RunHistoryContext value={runHistory}>
          <WhatIfContext value={whatIfControl}>
            <CellPaintContext value={paintFor}>
              <ContractContext value={contractsByTensor}>
                <div className="ide tensor-studio">
                  {/* Shared paint servers: any SVG in the workbench can burn a selection. */}
                  <svg className="tv-defs" aria-hidden="true" focusable="false">
                    <defs>
                      <linearGradient
                        id="tv-fire"
                        x1="0%"
                        y1="0%"
                        x2="70%"
                        y2="100%"
                      >
                        <stop offset="0%" stopColor="#fff1a8" />
                        <stop offset="45%" stopColor="#ffb347" />
                        <stop offset="100%" stopColor="#f0542c" />
                      </linearGradient>
                    </defs>
                  </svg>
                  <header className="title-bar">
                    <div
                      className="title-brand"
                      role="img"
                      aria-label="TensorViewer"
                    >
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
                          aria-label={
                            dirty ? "Save project" : "All changes saved"
                          }
                          title={
                            dirty
                              ? "Save, and update the diagram from the saved code (Ctrl/⌘ + S)"
                              : "All changes saved"
                          }
                          disabled={
                            !dirty || busy || !valid || !draft?.name.trim()
                          }
                          onClick={() => void saveAndUpdate()}
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
                          title="Tensors (Ctrl/⌘ + J)"
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
                          title="Code (Ctrl/⌘ + E)"
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
                          title={
                            id === "explorer" ? `${label} (Ctrl/⌘ + B)` : label
                          }
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
                      <>
                        <aside
                          className="side-bar"
                          style={{ width: sideWidth }}
                          tabIndex={-1}
                          aria-label={
                            side === "explorer" ? "Projects" : "Run history"
                          }
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
                              lastRunOf={lastRunOf}
                              project={project}
                              draft={draft}
                              run={run}
                              busy={busy || loading}
                              activeFile={file}
                              currentNode={currentOperation}
                              currentRange={currentCard?.operationIds}
                              changed={changedSteps}
                              placeLine={stale ? nowLine : undefined}
                              canvasFolds={foldControls}
                              activatedThrough={activated}
                              currentLoop={currentLoop}
                              onProject={(next) => void switchProject(next)}
                              onProjectAction={manageProject}
                              renameRequest={renameRequest}
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
                              onDelete={deleteRun}
                              onExport={(id) => void exportPytest(id)}
                              onClearOlder={clearOlderRuns}
                            />
                          )}
                        </aside>
                        <SideResizer
                          width={sideWidth}
                          onWidth={setSideWidth}
                          onCollapse={() => closePanel("side")}
                        />
                      </>
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
                                        (item) =>
                                          item.id === event.target.value,
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
                                      [
                                        "Shape",
                                        "Combine",
                                        "Compute",
                                        "Memory",
                                      ] as const
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
                                        (item) =>
                                          item.id === event.target.value,
                                      );
                                      if (example)
                                        setDraft({
                                          ...draft,
                                          script: example.script,
                                          input: exampleInput(
                                            draft.input,
                                            example,
                                          ),
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
                          {(draft?.blueprint || stale || isWhatIf(run?.id)) && (
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
                              {surface === "trace" &&
                                (executing || queued ? (
                                  <span
                                    className="tab-note updating"
                                    role="status"
                                  >
                                    <LoaderCircle className="spin" size={12} />
                                    {queued
                                      ? "updating again after this run…"
                                      : "updating the diagram…"}
                                  </span>
                                ) : whatIf && isWhatIf(run?.id) ? (
                                  <span className="tab-note what-if-note">
                                    <FlaskConical size={12} />
                                    <span className="tab-note-edits">
                                      what if{" "}
                                      {whatIf.edits
                                        .map((edit) => {
                                          const input =
                                            whatIfControl &&
                                            run?.trace.tensors[
                                              whatIfControl.inputId
                                            ];
                                          const word =
                                            whatIfControl?.vocabulary?.[
                                              edit.value
                                            ];
                                          return `${input?.name ?? "x"}[${unravel(edit.index, input?.shape ?? []).join(", ")}] = ${word ?? edit.value}`;
                                        })
                                        .join(", ")}
                                      {whatIf.precision &&
                                        `${whatIf.edits.length ? ", " : ""}it ran in ${whatIf.precision}`}
                                      {whatIf.knockout &&
                                        `${whatIf.edits.length || whatIf.precision ? ", " : ""}${knockoutText(whatIf.base.trace, whatIf.knockout)}`}
                                      {whatIf.learn &&
                                        `${whatIf.edits.length || whatIf.precision || whatIf.knockout ? ", " : ""}it learned ${whatIf.learn.steps && whatIf.learn.steps > 1 ? `${whatIf.learn.steps} steps` : "one step"} ${whatIf.learn.sentence ? "on the sentence" : `${whatIf.learn.direction > 0 ? "↑" : "↓"} ${learnName(whatIf.base, whatIf.learn)}`}`}{" "}
                                      · not saved
                                    </span>
                                    <button
                                      type="button"
                                      onClick={leaveWhatIf}
                                      title="Show the recorded run again"
                                    >
                                      Back to recorded
                                    </button>
                                  </span>
                                ) : (
                                  stale && (
                                    <button
                                      type="button"
                                      className="tab-note"
                                      title={`Edited since this run: ${pendingEdits.join(", ") || "the model"}. Save, and update the diagram from the saved code (Ctrl/⌘ + S).`}
                                      disabled={busy || !valid}
                                      onClick={() => void saveAndUpdate()}
                                    >
                                      <History size={12} />
                                      <span className="tab-note-edits">
                                        saved run · edited since
                                        {pendingEdits.length > 0 &&
                                          `: ${pendingEdits[0]}${pendingEdits.length > 1 ? ` +${pendingEdits.length - 1}` : ""}`}
                                      </span>
                                      <b>Update</b>
                                    </button>
                                  )
                                ))}
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
                                    previewStep={flowPreview ?? codePreview}
                                    onHoverStep={setCanvasHover}
                                    previousRunId={previousRunId}
                                    busy={busy}
                                    recording={executing}
                                    stale={stale}
                                    active={!settings && !showNew}
                                    onInspect={() => setSettings(false)}
                                    onEditModel={fixRun}
                                    fixes={
                                      runFixes && runFixes.runId === run?.id
                                        ? {
                                            checking: runFixes.checking,
                                            items: runFixes.fixes,
                                            apply: applyRunFix,
                                          }
                                        : null
                                    }
                                    onRun={
                                      canRun ? () => void execute() : undefined
                                    }
                                    onShowCode={openCode}
                                    onLibrary={() => {
                                      if (side !== "explorer")
                                        rememberOpener("side");
                                      setSide("explorer");
                                      focusPanel("side");
                                    }}
                                    onEditInputs={openInputs}
                                    focusOperation={focusOperation}
                                    onCurrentOperation={setCurrentOperation}
                                    onCurrentLoop={setCurrentLoop}
                                    onCurrentCard={setCurrentCard}
                                    onFoldControls={setFoldControls}
                                    onActivated={setActivated}
                                    thread={thread}
                                    breakpoints={breakpointSteps}
                                    warningSteps={warningSteps}
                                    pauseOnWarnings={pauseOnWarnings}
                                    onPauseOnWarnings={changePauseOnWarnings}
                                    onCell={(node, index) =>
                                      setCell({ node, index })
                                    }
                                    leftOff={leftOff}
                                    placeLine={stale ? nowLine : undefined}
                                    relit={changedSteps}
                                    onCarried={(change) => {
                                      // Pinned names follow the tensors they name.
                                      if (change && project)
                                        renamePins(project.id, change.renames);
                                      if (!reloading.current) return;
                                      reloading.current = false;
                                      if (!change) return;
                                      setNotice(
                                        `Diagram updated · ${change.summary}`,
                                      );
                                      setChangedSteps(change.changed);
                                      if (run)
                                        setLastChange({
                                          runId: run.id,
                                          summary: change.summary,
                                          first: change.first,
                                          changed: change.changed,
                                        });
                                      // A change to the steps or names shows the
                                      // first one; values alone compare every
                                      // tensor instead.
                                      const first = change.first;
                                      setNoticeAction(
                                        first &&
                                          (change.structural ||
                                            change.renames.size)
                                          ? {
                                              label: "Show",
                                              act: () => {
                                                select(first);
                                                setNotice("");
                                              },
                                            }
                                          : change.valuesMayDiffer
                                            ? {
                                                label: "Compare values",
                                                act: () => {
                                                  change.compare();
                                                  setNotice("");
                                                },
                                              }
                                            : null,
                                      );
                                    }}
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
                              selectedRange={currentCard?.operationIds}
                              changed={changedSteps}
                              onSelect={(id) => {
                                setSurface("trace");
                                select(id);
                              }}
                              onThread={setThread}
                              onPreview={setFlowPreview}
                              hovered={canvasHover}
                              previousRunId={previousRunId}
                              projectId={project?.id ?? null}
                              height={panelHeight}
                              onHeight={setPanelHeight}
                              onProblem={(problem) => {
                                if (
                                  problem.file &&
                                  problem.file !== file &&
                                  !isConsole
                                )
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
                          <EdgeResizer
                            className="code-resizer"
                            label="Resize the code"
                            grow="left"
                            size={editorWidth ?? 0}
                            measure={() =>
                              document.querySelector<HTMLElement>(
                                ".editor-column",
                              )?.offsetWidth ?? 0
                            }
                            min={240}
                            max={Math.max(240, window.innerWidth - 420)}
                            onSize={resizeEditor}
                            onReset={() => {
                              setEditorWidth(null);
                              try {
                                localStorage.removeItem(
                                  "tensorviewer.editorWidth",
                                );
                              } catch {
                                // Nothing was stored.
                              }
                            }}
                            onCollapse={() => closePanel("editor")}
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
                                    {kind === "canvas" && (
                                      <small>generated</small>
                                    )}
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
                            <nav className="breadcrumbs" aria-label="Location">
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
                            </nav>
                            {kind === "canvas" && (
                              <p className="editor-note-bar">
                                Generated from the canvas. Edit components in
                                Builder, or convert the project in Settings to
                                edit this code.
                              </p>
                            )}
                            <CodeEditor
                              key={`${project?.id}/${file}`}
                              label={
                                isConsole
                                  ? "Tensor statements"
                                  : `Source of ${file}`
                              }
                              value={fileText}
                              // Editing goes on while a run records; the next
                              // save picks it up.
                              readOnly={
                                (busy && !executing) || kind === "canvas"
                              }
                              placeholder={
                                isConsole
                                  ? `y = ${draft.input_name ?? "x"}.reshape(-1)`
                                  : undefined
                              }
                              results={results}
                              activeLine={loopLine ?? activeLine}
                              changedLines={changedLines}
                              hoverLines={canvasHoverLines}
                              tensors={tensorNames}
                              shapes={knownShapeMap}
                              latest={lastVariable(
                                fileText,
                                draft.input_name ?? "x",
                              )}
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
                              onCursor={(line, column) =>
                                setCursor({ line, column })
                              }
                              onHoverLine={setCodeHover}
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
                            onKeyDown={(event) =>
                              panelKeyDown(event, "settings")
                            }
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
                                      (!!draft.blueprint &&
                                        !builderEditingValid)
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
                                  {!!draft.blueprint &&
                                    !builderEditingValid && (
                                      <p className="field-error">
                                        Finish or revert the component arguments
                                        first.{" "}
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
                    currentCard={currentCard}
                    cell={
                      cell?.node === currentOperation
                        ? (cell?.index ?? null)
                        : null
                    }
                    cursor={editorOpen ? cursor : null}
                    placeLine={stale ? nowLine : undefined}
                    autoUpdate={
                      autoUpdate
                        ? { onTurnOff: () => changeAutoUpdate(false) }
                        : null
                    }
                    check={
                      !draft || draft.blueprint
                        ? null
                        : {
                            state: checking
                              ? "checking"
                              : run && !stale
                                ? "recorded"
                                : !currentCheck
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
                      draft &&
                      !busy &&
                      setDraft({ ...draft, capture_mode: mode })
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
                      existing={(name) =>
                        projects.find((item) => item.name === name)
                      }
                      onOpen={(item) => void switchProject(item)}
                    />
                  )}
                  {shortcuts && (
                    <ShortcutsDialog onClose={() => setShortcuts(false)} />
                  )}
                  {notice && (
                    <div className="toast" role="status">
                      {notice}
                      {noticeAction && (
                        <button
                          type="button"
                          className="toast-action"
                          onClick={noticeAction.act}
                        >
                          {noticeAction.label}
                        </button>
                      )}
                    </div>
                  )}
                </div>
              </ContractContext>
            </CellPaintContext>
          </WhatIfContext>
        </RunHistoryContext>
      </SymbolicContext>
    </AxisInkContext>
  );
}
