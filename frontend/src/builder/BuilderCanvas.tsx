import { useEffect, useId, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Box,
  Check,
  Code2,
  CircleAlert,
  ChevronLeft,
  ChevronRight,
  LoaderCircle,
  Layers3,
  Maximize,
  Minus,
  Plus,
  SlidersHorizontal,
  Trash2,
  X,
} from "lucide-react";
import {
  api,
  type CustomComponentDraft,
  type Draft,
  type Tensor,
  type ToolboxItem,
} from "../api/client";
import { TensorGlyph } from "../journey/TensorGlyph";
import { TensorVolumeDialog } from "../tensors/TensorVolumeDialog";
import { previewTensor } from "./model";
import { componentSources, hasBranches } from "./connections";
import { BranchConnections } from "./BranchConnections";
import { ShapeSummary } from "../components/ShapeSummary";
import { componentTool, customTool } from "./custom";
import {
  buildReadiness,
  canApplyCompositionPreview,
  compositionSignature,
  type BuildReadiness,
  type CheckedPlan,
  type CompositionResult,
} from "./readiness";
import {
  constructorEdit,
  currentConstructorEdit,
  reconcileConstructorEdits,
  type ConstructorEdits,
} from "./constructorEdits";
import {
  CustomComponentDialog,
  ConstructorArguments,
} from "./CustomComponentDialog";
import { ParameterField } from "./ParameterField";
import {
  ToolboxPanel,
  ComponentToolbar,
  ComponentIcon,
  type ToolGroup,
} from "./Toolbox";

export function BuilderCanvas({
  draft,
  onChange,
  onReadiness,
  onEditingValidity,
  reviewRequest,
  checkedPlan,
  busy,
  onShowRun,
  onInputs,
  active,
  hasRun,
  toolboxGroup,
  onToolboxGroup,
}: {
  draft: Draft;
  onChange: (draft: Draft) => void;
  onReadiness: (readiness: BuildReadiness) => void;
  onEditingValidity: (valid: boolean) => void;
  reviewRequest: number;
  checkedPlan: CheckedPlan | null;
  busy: boolean;
  onShowRun: () => void;
  onInputs: () => void;
  active: boolean;
  hasRun: boolean;
  toolboxGroup: ToolGroup | null;
  onToolboxGroup: (group: ToolGroup | null) => void;
}) {
  const blueprint = draft.blueprint!;
  const projectNameMessageId = useId();
  const signature = compositionSignature(draft);
  const latestDraft = useRef(draft);
  latestDraft.current = draft;
  const latestSignature = useRef(signature);
  latestSignature.current = signature;
  const latestCheckedPlan = useRef(checkedPlan);
  latestCheckedPlan.current = checkedPlan;
  const acceptedCheckedPlan = useRef<CheckedPlan | null>(null);
  const branched = hasBranches(blueprint.components);
  const [catalog, setCatalog] = useState<ToolboxItem[]>([]);
  const [localResult, setLocalResult] = useState<CompositionResult | null>(
    null,
  );
  const incomingCheckedPlan =
    checkedPlan?.signature === signature &&
    (checkedPlan !== acceptedCheckedPlan.current ||
      localResult?.signature !== signature)
      ? checkedPlan
      : null;
  const result = incomingCheckedPlan ?? localResult;
  const plan = result?.signature === signature ? result.plan : null;
  const error =
    result?.signature === signature && "error" in result
      ? (result.error ?? "")
      : "";
  const [pending, setPending] = useState(false);
  const [checking, setChecking] = useState(false);
  const [constructorEdits, setConstructorEdits] = useState<ConstructorEdits>(
    {},
  );
  const invalidArguments = blueprint.components.find(
    (component) => currentConstructorEdit(constructorEdits, component)?.error,
  );
  const argumentsValid = !invalidArguments;
  const argumentsIssue = invalidArguments
    ? `Fix or revert the constructor arguments for ${invalidArguments.custom!.name} (step ${blueprint.components.indexOf(invalidArguments) + 1}).`
    : "";
  useEffect(() => {
    setConstructorEdits((edits) =>
      reconcileConstructorEdits(edits, blueprint.components),
    );
  }, [blueprint.components]);
  useEffect(() => {
    onEditingValidity(argumentsValid);
  }, [argumentsValid, onEditingValidity]);
  const [customEditor, setCustomEditor] = useState<{
    initial?: CustomComponentDraft;
    nodeId?: string;
  } | null>(null);
  const checkController = useRef<AbortController | null>(null);
  const composeController = useRef<AbortController | null>(null);
  const [catalogError, setCatalogError] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [focusProjectName, setFocusProjectName] = useState(false);
  const [connectionViews, setConnectionViews] = useState<
    Record<string, boolean>
  >({});
  const wasActive = useRef(active);
  useEffect(() => {
    const reopening = active && !wasActive.current;
    wasActive.current = active;
    if (reopening && invalidArguments) {
      setSelected(invalidArguments.id);
      onToolboxGroup(null);
    }
  }, [active, invalidArguments?.id, onToolboxGroup]);
  const [volume, setVolume] = useState<{
    tensor: Tensor;
    index: number;
  } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [retry, setRetry] = useState(0);
  const stageElement = useRef<HTMLDivElement>(null);
  const inspectorContent = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!active || selected !== "project" || !focusProjectName) return;
    inspectorContent.current
      ?.querySelector<HTMLInputElement>('[aria-label="Project name"]')
      ?.focus({ preventScroll: true });
    setFocusProjectName(false);
  }, [active, selected, focusProjectName]);
  const sequenceOverview = useRef<HTMLElement>(null);
  useEffect(() => {
    if (inspectorContent.current) inspectorContent.current.scrollTop = 0;
    sequenceOverview.current
      ?.querySelector<HTMLElement>("[aria-current=step]")
      ?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [selected]);
  useEffect(() => {
    if (!selected || selected === "project") return;
    const frame = requestAnimationFrame(() =>
      stageElement.current
        ?.querySelector<HTMLElement>(`[data-component-id="${selected}"]`)
        ?.scrollIntoView({
          block: "nearest",
          inline: "center",
          behavior: "instant",
        }),
    );
    return () => cancelAnimationFrame(frame);
  }, [selected, zoom]);
  useEffect(() => {
    checkController.current?.abort();
    setChecking(false);
    return () => checkController.current?.abort();
  }, [signature]);
  async function checkShapes() {
    if (checking || busy || pending || !argumentsValid) return;
    composeController.current?.abort();
    setPending(false);
    const controller = new AbortController();
    const preflightAtStart = latestCheckedPlan.current;
    checkController.current = controller;
    setChecking(true);
    const stillCurrent = () =>
      !controller.signal.aborted &&
      latestSignature.current === signature &&
      (latestCheckedPlan.current === preflightAtStart ||
        latestCheckedPlan.current?.signature !== signature);
    try {
      const next = await api.checkComposition(draft, controller.signal);
      if (!stillCurrent()) return;
      setLocalResult({ signature, plan: next });
      onChange({
        ...latestDraft.current,
        code: next.code,
        class_name: "ComposedModel",
        constructor: {},
      });
    } catch (e) {
      if (stillCurrent())
        setLocalResult({ signature, plan: null, error: (e as Error).message });
    } finally {
      if (stillCurrent()) setChecking(false);
    }
  }
  useEffect(() => {
    let live = true;
    setCatalogError("");
    api
      .toolbox()
      .then((items) => {
        if (live) setCatalog(items);
      })
      .catch((e) => {
        if (live) setCatalogError(e.message);
      });
    return () => {
      live = false;
    };
  }, [retry]);
  useEffect(() => {
    if (latestCheckedPlan.current?.signature === signature) return;
    const controller = new AbortController();
    composeController.current = controller;
    setPending(true);
    const stillCurrent = () =>
      !controller.signal.aborted &&
      canApplyCompositionPreview(
        signature,
        latestSignature.current,
        latestCheckedPlan.current?.signature,
      );
    const timer = window.setTimeout(() => {
      api
        .compose(draft, controller.signal)
        .then((next) => {
          if (!stillCurrent()) return;
          setLocalResult({ signature, plan: next });
          setPending(false);
          if (next.code !== draft.code)
            onChange({
              ...latestDraft.current,
              code: next.code,
              class_name: "ComposedModel",
              constructor: {},
            });
        })
        .catch((e) => {
          if (stillCurrent()) {
            setLocalResult({ signature, plan: null, error: e.message });
            setPending(false);
          }
        });
    }, 180);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [signature, retry]);
  useEffect(() => {
    if (!incomingCheckedPlan) return;
    acceptedCheckedPlan.current = incomingCheckedPlan;
    composeController.current?.abort();
    checkController.current?.abort();
    setLocalResult(incomingCheckedPlan);
    setPending(false);
    setChecking(false);
  }, [incomingCheckedPlan]);
  const readiness = buildReadiness(draft, result, {
    checking: (pending || checking) && !incomingCheckedPlan,
    ...(invalidArguments
      ? {
          editingIssue: {
            issue: argumentsIssue,
            componentId: invalidArguments.id,
          },
        }
      : {}),
  });
  useEffect(() => {
    onReadiness(readiness);
  }, [
    readiness.signature,
    readiness.state,
    readiness.issue,
    readiness.componentId,
    onReadiness,
  ]);
  const inputTensor = previewTensor(
    "builder-input",
    "Input",
    draft.input.shape,
    draft.input.axis_names,
    draft.input.dtype,
  );
  const selectedSpec = blueprint.components.find((c) => c.id === selected);
  const selectedArguments = selectedSpec
    ? currentConstructorEdit(constructorEdits, selectedSpec)
    : undefined;
  const selectedItem = componentTool(selectedSpec, catalog);
  const selectedStage = plan?.stages.find((s) => s.id === selected);
  const inputSelected = selected === "input";
  const selectionIndex = blueprint.components.findIndex(
    (c) => c.id === selected,
  );
  const selectedSources =
    selectionIndex >= 0
      ? componentSources(blueprint.components, selectionIndex)
      : ["input"];
  const precedingComponents = blueprint.components.slice(0, selectionIndex);
  const invalidSources = selectedSources.filter(
    (id) =>
      id !== "input" && !precedingComponents.some((item) => item.id === id),
  );
  const invalidSourceKey = JSON.stringify(invalidSources);
  const connectionsOpen = selectedSpec
    ? (connectionViews[selectedSpec.id] ??
      (["add_join", "concat_join", "stack_join"].includes(selectedSpec.kind) ||
        selectedSources.length !== 1 ||
        selectedSources[0] !== (precedingComponents.at(-1)?.id ?? "input")))
    : false;
  useEffect(() => {
    if (!selectedSpec || !invalidSources.length) return;
    setConnectionViews((views) => ({ ...views, [selectedSpec.id]: true }));
  }, [selectedSpec?.id, invalidSourceKey]);
  function sourceLabel(id: string) {
    if (id === "input") return "Input tensor";
    const index = blueprint.components.findIndex((item) => item.id === id);
    if (index < 0) return "Missing component";
    return `Step ${index + 1} · ${componentTool(blueprint.components[index], catalog)?.title ?? blueprint.components[index].kind}`;
  }
  const axesForSource = (id: string) =>
    id === "input"
      ? draft.input.axis_names
      : (plan?.stages.find((stage) => stage.id === id)?.axes ?? []);
  const incomingAxes = axesForSource(selectedSources[0]);
  const sequenceIds = ["input", ...blueprint.components.map((c) => c.id)];
  const selectedPosition = selected ? sequenceIds.indexOf(selected) : -1;
  const firstIssue = plan?.stages.find((s) => s.error);
  const reviewedRequest = useRef(reviewRequest);
  useEffect(() => {
    if (reviewRequest === reviewedRequest.current) return;
    reviewedRequest.current = reviewRequest;
    reviewComponent(
      !draft.name.trim()
        ? "project"
        : (invalidArguments?.id ?? firstIssue?.id ?? "input"),
    );
  }, [reviewRequest]);
  function closeToolbox() {
    onToolboxGroup(null);
    document
      .querySelector<HTMLButtonElement>(".component-tool[aria-expanded='true']")
      ?.focus();
  }
  function closeSettings() {
    if (selected && selected !== "project")
      stageElement.current
        ?.querySelector<HTMLButtonElement>(
          `[data-component-id="${selected}"] .builder-node-heading`,
        )
        ?.focus({ preventScroll: true });
    if (selected === "project")
      document
        .querySelector<HTMLButtonElement>(
          '.builder-heading [aria-label="Project settings"]',
        )
        ?.focus();
    setSelected(null);
  }
  function updateParameter(key: string, value: number) {
    changeComponents(
      blueprint.components.map((c) =>
        c.id === selected
          ? { ...c, parameters: { ...c.parameters, [key]: value } }
          : c,
      ),
    );
  }
  function selectComponent(id: string) {
    if (id === "input") {
      setSelected(null);
      onToolboxGroup(null);
      onInputs();
      return;
    }
    setSelected(id);
    if (id !== "project") setZoom((z) => Math.max(0.9, z));
    onToolboxGroup(null);
  }
  function reviewComponent(id: string) {
    selectComponent(id);
    if (id === "project") {
      setFocusProjectName(true);
      return;
    }
    const index = blueprint.components.findIndex((item) => item.id === id);
    if (index < 0) return;
    const previous = blueprint.components.slice(0, index);
    const invalid = componentSources(blueprint.components, index).some(
      (source) =>
        source !== "input" && !previous.some((item) => item.id === source),
    );
    if (!invalid) return;
    setConnectionViews((views) => ({ ...views, [id]: true }));
    requestAnimationFrame(() => {
      const field = inspectorContent.current?.querySelector<HTMLElement>(
        '.builder-connections [aria-invalid="true"]',
      );
      field?.focus({ preventScroll: true });
      field?.scrollIntoView({ block: "nearest" });
    });
  }
  function addInput() {
    onChange({ ...draft, blueprint: { ...blueprint, has_input: true } });
    selectComponent("input");
  }
  useEffect(() => {
    if (toolboxGroup) setSelected(null);
  }, [toolboxGroup]);
  function changeComponents(components: typeof blueprint.components) {
    onChange({ ...draft, blueprint: { ...blueprint, components } });
  }
  function changeConstructor(text: string) {
    if (!selectedSpec?.custom) return;
    const { edit, value } = constructorEdit(selectedSpec, text);
    setConstructorEdits((edits) => ({ ...edits, [selectedSpec.id]: edit }));
    if (
      value &&
      JSON.stringify(value) !==
        JSON.stringify(
          selectedSpec.arguments ?? selectedSpec.custom.constructor,
        )
    ) {
      const current = latestDraft.current;
      onChange({
        ...current,
        blueprint: {
          ...current.blueprint!,
          components: current.blueprint!.components.map((component) =>
            component.id === selectedSpec.id
              ? { ...component, arguments: value }
              : component,
          ),
        },
      });
    }
  }
  function add(item: ToolboxItem) {
    if (busy || blueprint.components.length >= 16) return;
    const id = crypto.randomUUID();
    const input =
      !blueprint.has_input &&
      (item.group === "Spatial" ||
        item.kind === "patch_embedding" ||
        item.kind === "vit" ||
        item.kind === "hierarchical_vit" ||
        item.kind === "batchnorm2d" ||
        item.kind === "tokens")
        ? {
            ...draft.input,
            shape: [2, 3, 8, 8],
            axis_names: ["batch", "channels", "height", "width"],
          }
        : !blueprint.has_input && item.kind === "window_reverse"
          ? {
              ...draft.input,
              shape: [8, 4, 8],
              axis_names: ["batch_windows", "window_tokens", "features"],
            }
          : !blueprint.has_input && item.kind === "conv1d"
            ? {
                ...draft.input,
                shape: [2, 3, 16],
                axis_names: ["batch", "channels", "length"],
              }
            : !blueprint.has_input && item.kind === "squeeze"
              ? {
                  ...draft.input,
                  shape: [2, 1, 8],
                  axis_names: ["batch", "unit", "features"],
                }
              : draft.input;
    onChange({
      ...draft,
      input,
      blueprint: {
        has_input: true,
        components: [
          ...blueprint.components,
          {
            id,
            kind: item.kind,
            ...(item.custom ? { custom: item.custom } : {}),
            parameters: Object.fromEntries(
              item.parameters.map((p) => [p.key, p.default]),
            ),
          },
        ],
      },
    });
    selectComponent(id);
  }
  function move(direction: number) {
    const list = [...blueprint.components],
      at = list.findIndex((c) => c.id === selected),
      to = at + direction;
    if (at < 0 || to < 0 || to >= list.length) return;
    [list[at], list[to]] = [list[to], list[at]];
    changeComponents(list);
  }
  function fit() {
    const stage = stageElement.current;
    const sequence = stage?.querySelector<HTMLElement>(".builder-sequence");
    if (stage && sequence)
      setZoom(
        Math.max(
          0.2,
          Math.min(
            1,
            (stage.clientWidth - 16) /
              (sequence.getBoundingClientRect().width / zoom),
          ),
        ),
      );
  }
  return (
    <section
      className="builder-workspace"
      aria-label="Model builder"
      onKeyDown={(e) => {
        if (e.key === "Escape" && selected) {
          e.stopPropagation();
          closeSettings();
        }
      }}
    >
      <header className="builder-heading">
        <div>
          <Layers3 size={17} />
          <h1>Model canvas</h1>
          <span className="canvas-mode">Design</span>
          <span className="canvas-component-count">
            {blueprint.components.length} component
            {blueprint.components.length === 1 ? "" : "s"}
          </span>
        </div>
        <div>
          <button
            className="secondary-button"
            aria-label="Project settings"
            aria-pressed={selected === "project"}
            onClick={() => selectComponent("project")}
          >
            <SlidersHorizontal size={14} /> Project
          </button>
          {hasRun && (
            <button className="secondary-button" onClick={onShowRun}>
              View last run <ArrowRight size={14} />
            </button>
          )}
        </div>
      </header>
      <ComponentToolbar
        active={toolboxGroup}
        disabled={busy}
        onSelect={(group) =>
          onToolboxGroup(toolboxGroup === group ? null : group)
        }
      />
      <div className="builder-body" inert={busy}>
        {toolboxGroup && (
          <ToolboxPanel
            group={toolboxGroup}
            catalog={catalog}
            onAdd={add}
            onInput={addInput}
            onClose={closeToolbox}
            limit={blueprint.components.length >= 16}
            error={catalogError}
            onRetry={() => setRetry((n) => n + 1)}
            onCreateCustom={() => setCustomEditor({})}
            onGroupSelect={onToolboxGroup}
          />
        )}
        <div className="builder-canvas-area">
          <div className="builder-stage" ref={stageElement}>
            {!blueprint.has_input && !blueprint.components.length ? (
              <div className="builder-empty">
                <div className="empty-orbit">
                  <Box size={30} />
                </div>
                <span className="eyebrow">YOUR TENSOR EXPERIMENT</span>
                <h2>Build an idea. See it unfold.</h2>
                <p>
                  Choose your input. Connect a few components.
                  <br />
                  Run the model to explore every transformation.
                </p>
                <div className="empty-actions">
                  <button className="primary-button" onClick={addInput}>
                    <Plus size={15} /> Set up input
                  </button>
                  <button
                    className="secondary-button"
                    onClick={() => onToolboxGroup("All")}
                  >
                    Browse tools <ArrowRight size={14} />
                  </button>
                </div>
                <div className="canvas-onboarding">
                  <span>
                    <b>01</b> Configure input
                  </span>
                  <span>
                    <b>02</b> Add components
                  </span>
                  <span>
                    <b>03</b> Run & explore
                  </span>
                </div>
              </div>
            ) : (
              <div
                className={`builder-sequence ${branched ? "has-branches" : ""}`}
                aria-label="Connected component sequence"
                style={{ zoom }}
              >
                {branched && (
                  <BranchConnections components={blueprint.components} />
                )}
                <article
                  className={`builder-node ${inputSelected ? "selected" : ""}`}
                  data-component-id="input"
                >
                  <button
                    className="builder-node-heading"
                    aria-label="Configure input tensor"
                    onClick={() => selectComponent("input")}
                  >
                    <span className="eyebrow">INPUT</span>
                    <b>Input tensor</b>
                    <SlidersHorizontal size={14} />
                  </button>
                  {blueprint.has_input ? (
                    <>
                      <TensorGlyph
                        tensor={inputTensor}
                        onSelect={(index) =>
                          setVolume({ tensor: inputTensor, index })
                        }
                      />
                      <ShapeSummary
                        shape={draft.input.shape}
                        axes={draft.input.axis_names}
                      />
                      <footer>
                        <span className="node-preview-label">
                          Input preview
                        </span>
                        <button
                          aria-label="Inspect input tensor"
                          onClick={() =>
                            setVolume({ tensor: inputTensor, index: 0 })
                          }
                        >
                          <Box size={12} /> Inspect 3D
                        </button>
                      </footer>
                    </>
                  ) : (
                    <p className="field-error">
                      Add an input tensor from the toolbox.
                    </p>
                  )}
                </article>
                {blueprint.components.map((component, i) => {
                  const item = componentTool(component, catalog),
                    stage = plan?.stages.find((s) => s.id === component.id);
                  const tensor = stage?.shape
                    ? previewTensor(
                        component.id,
                        item?.title ?? component.kind,
                        stage.shape,
                        stage.axes ?? [],
                        draft.input.dtype,
                      )
                    : null;
                  return (
                    <div className="builder-link" key={component.id}>
                      <div className="builder-connector" aria-hidden="true">
                        <span />
                        <ArrowRight size={15} />
                      </div>
                      <article
                        className={`builder-node ${selected === component.id ? "selected" : ""} ${stage?.error && !plan?.validation_required ? "invalid" : ""}`}
                        data-component-id={component.id}
                      >
                        <button
                          className="builder-node-heading"
                          aria-label={`Configure ${item?.title ?? component.kind} ${i + 1}`}
                          onClick={() => selectComponent(component.id)}
                        >
                          <span className="eyebrow">
                            {String(i + 1).padStart(2, "0")}
                          </span>
                          <span className="node-component-icon">
                            <ComponentIcon kind={component.kind} size={15} />
                          </span>
                          <b>{item?.title ?? component.kind}</b>
                          <SlidersHorizontal size={14} />
                        </button>
                        {branched && (
                          <div
                            className="builder-source-tags"
                            aria-label={`Inputs for step ${i + 1}`}
                          >
                            {componentSources(blueprint.components, i).map(
                              (id, slot) => (
                                <button
                                  key={slot}
                                  onClick={() => selectComponent(id)}
                                  title={`Inspect input ${slot + 1}`}
                                >
                                  <ArrowRight size={10} />
                                  {id === "input"
                                    ? "Input tensor"
                                    : `Step ${blueprint.components.findIndex((c) => c.id === id) + 1 || "?"} output`}
                                </button>
                              ),
                            )}
                          </div>
                        )}
                        {tensor && !pending ? (
                          <>
                            <TensorGlyph
                              tensor={tensor}
                              onSelect={(index) => setVolume({ tensor, index })}
                            />
                            <ShapeSummary
                              shape={tensor.shape}
                              axes={tensor.axes}
                            />
                          </>
                        ) : (
                          <div className="builder-node-message">
                            {pending || checking ? (
                              "Checking shape…"
                            ) : (
                              <>
                                <CircleAlert size={19} />
                                <p>{stage?.error || "Shape unavailable"}</p>
                              </>
                            )}
                          </div>
                        )}
                        <footer>
                          {pending || checking ? (
                            <span>Checking output…</span>
                          ) : plan?.validation_required ? (
                            <span>Awaiting shape check</span>
                          ) : stage?.error ? (
                            <>
                              <CircleAlert size={12} /> Needs attention
                            </>
                          ) : (
                            <>
                              <span className="node-preview-label">
                                Output preview
                              </span>
                            </>
                          )}
                          {tensor && !pending && (
                            <button
                              aria-label={`Inspect ${item?.title ?? component.kind} ${i + 1} tensor`}
                              onClick={() => setVolume({ tensor, index: 0 })}
                            >
                              <Box size={12} /> 3D
                            </button>
                          )}
                        </footer>
                      </article>
                    </div>
                  );
                })}
                <button
                  className="builder-add"
                  aria-label="Add next component"
                  onClick={() => onToolboxGroup("All")}
                >
                  <Plus size={22} />
                  <span>Add component</span>
                </button>
              </div>
            )}
          </div>
          {(blueprint.has_input || blueprint.components.length > 0) && (
            <div className="builder-dock">
              {blueprint.has_input && (
                <nav
                  className="sequence-overview"
                  ref={sequenceOverview}
                  aria-label="Sequence navigation"
                >
                  <button
                    aria-label="Go to input tensor"
                    aria-current={inputSelected ? "step" : undefined}
                    onClick={() => selectComponent("input")}
                  >
                    <Box size={13} />
                    <span>Input</span>
                  </button>
                  {blueprint.components.map((c, i) => (
                    <button
                      key={c.id}
                      aria-label={`Go to ${componentTool(c, catalog)?.title ?? c.kind}, step ${i + 1}`}
                      aria-current={selected === c.id ? "step" : undefined}
                      className={
                        plan?.stages.find((s) => s.id === c.id)?.error ||
                        currentConstructorEdit(constructorEdits, c)?.error
                          ? "has-issue"
                          : ""
                      }
                      onClick={() => selectComponent(c.id)}
                    >
                      <span className="sequence-number">
                        {String(i + 1).padStart(2, "0")}
                      </span>
                      <span>{componentTool(c, catalog)?.title ?? c.kind}</span>
                    </button>
                  ))}
                </nav>
              )}
              <div className="builder-status">
                <span
                  className={`builder-readiness ${readiness.state === "invalid" ? "has-issue" : ""}`}
                  role="status"
                >
                  {readiness.state === "checking" ? (
                    <LoaderCircle size={13} className="spin" />
                  ) : readiness.state === "invalid" ? (
                    <CircleAlert size={13} />
                  ) : (
                    <Check size={13} />
                  )}
                  {checking
                    ? "Checking custom shapes…"
                    : readiness.state === "ready"
                      ? "Ready to generate"
                      : readiness.state === "invalid" &&
                          draft.name.trim() &&
                          blueprint.has_input &&
                          blueprint.components.length
                        ? invalidArguments
                          ? "Finish component settings"
                          : "Model needs attention"
                        : readiness.issue}
                  <span className="preview-disclaimer">
                    {readiness.state === "ready"
                      ? "Shapes preview · run to inspect values"
                      : ""}
                  </span>
                  {blueprint.components.some((c) => c.custom) && (
                    <button
                      className="custom-check-button"
                      disabled={
                        readiness.state === "checking" ||
                        !argumentsValid ||
                        !blueprint.has_input ||
                        !blueprint.components.length ||
                        busy
                      }
                      onClick={checkShapes}
                    >
                      {checking ? (
                        <LoaderCircle size={13} className="spin" />
                      ) : (
                        <Check size={13} />
                      )}
                      {checking ? "Checking…" : "Preview custom shapes"}
                    </button>
                  )}
                  {readiness.state === "invalid" && (
                    <button
                      onClick={() =>
                        reviewComponent(
                          !draft.name.trim()
                            ? "project"
                            : (invalidArguments?.id ??
                                firstIssue?.id ??
                                "input"),
                        )
                      }
                    >
                      Review <ArrowRight size={12} />
                    </button>
                  )}
                </span>
                <div className="builder-zoom">
                  <span>{blueprint.components.length} / 16</span>
                  <button
                    aria-label="Zoom model out"
                    onClick={() => setZoom((z) => Math.max(0.2, z / 1.2))}
                  >
                    <Minus size={13} />
                  </button>
                  <button
                    className="zoom-percentage"
                    aria-label="Reset model zoom to 100 percent"
                    title="Reset to 100%"
                    onClick={() => setZoom(1)}
                  >
                    {Math.round(zoom * 100)}%
                  </button>
                  <button
                    aria-label="Zoom model in"
                    onClick={() => setZoom((z) => Math.min(1.5, z * 1.2))}
                  >
                    <Plus size={13} />
                  </button>
                  <button
                    aria-label="Fit model sequence"
                    title="Fit model sequence"
                    onClick={fit}
                  >
                    <Maximize size={13} />
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
        {selected && (
          <aside
            className="builder-settings"
            hidden={!active}
            aria-label={
              selected === "project" ? "Project settings" : "Component settings"
            }
          >
            <header className="inspector-header">
              <div className="inspector-eyebrow">
                <span>
                  {selected === "project"
                    ? "PROJECT"
                    : inputSelected
                      ? "STARTING TENSOR"
                      : `COMPONENT ${String(selectionIndex + 1).padStart(2, "0")}`}
                </span>
                <button
                  className="icon-button"
                  aria-label={
                    selected === "project"
                      ? "Close project settings"
                      : "Close component settings"
                  }
                  onClick={closeSettings}
                >
                  <X size={16} />
                </button>
              </div>
              <div className="inspector-title">
                <span className="inspector-icon">
                  {selected === "project" ? (
                    <SlidersHorizontal size={20} />
                  ) : inputSelected ? (
                    <Box size={20} />
                  ) : (
                    <ComponentIcon kind={selectedSpec?.kind ?? ""} size={20} />
                  )}
                </span>
                <h2>
                  {selected === "project"
                    ? "Project settings"
                    : inputSelected
                      ? "Input tensor"
                      : (selectedItem?.title ?? "Component")}
                </h2>
              </div>
              {selected !== "project" && (
                <div className="inspector-navigation">
                  <button
                    className="icon-button"
                    aria-label="Select previous component"
                    disabled={selectedPosition <= 0}
                    onClick={() =>
                      selectComponent(sequenceIds[selectedPosition - 1])
                    }
                  >
                    <ChevronLeft size={15} />
                  </button>
                  <span>
                    {inputSelected
                      ? "Input"
                      : `Step ${selectionIndex + 1} of ${blueprint.components.length}`}
                  </span>
                  <button
                    className="icon-button"
                    aria-label="Select next component"
                    disabled={selectedPosition >= sequenceIds.length - 1}
                    onClick={() =>
                      selectComponent(sequenceIds[selectedPosition + 1])
                    }
                  >
                    <ChevronRight size={15} />
                  </button>
                </div>
              )}
            </header>
            <div className="inspector-content" ref={inspectorContent}>
              {selected === "project" ? (
                <>
                  <p className="settings-description">
                    Name your project and manage its input.
                  </p>
                  <label>
                    Project name
                    <input
                      aria-label="Project name"
                      aria-invalid={!draft.name.trim()}
                      aria-describedby={
                        !draft.name.trim() ? projectNameMessageId : undefined
                      }
                      required
                      maxLength={100}
                      value={draft.name}
                      onChange={(e) =>
                        onChange({ ...draft, name: e.target.value })
                      }
                    />
                  </label>
                  {!draft.name.trim() && (
                    <p
                      id={projectNameMessageId}
                      className="field-error"
                      role="alert"
                    >
                      Enter a project name to save or generate a diagram.
                    </p>
                  )}
                  <div className="setup-summary">
                    <span>
                      Input{" "}
                      {blueprint.has_input ? (
                        <code>[{draft.input.shape.join(", ")}]</code>
                      ) : (
                        <b>Not set</b>
                      )}
                    </span>
                    <span>
                      Components <b>{blueprint.components.length}</b>
                    </span>
                    <span>
                      Execution <b>CPU · evaluation</b>
                    </span>
                  </div>
                  <button
                    className="secondary-button"
                    onClick={
                      blueprint.has_input
                        ? () => selectComponent("input")
                        : addInput
                    }
                  >
                    <Box size={14} />{" "}
                    {blueprint.has_input ? "Edit inputs" : "Set up input"}
                  </button>
                </>
              ) : selectedSpec && selectedItem ? (
                <>
                  <p className="settings-description">
                    {selectedItem.description}
                  </p>
                  <details
                    key={selectedSpec.id}
                    className="builder-connections"
                    open={connectionsOpen}
                    onToggle={(event) => {
                      const open = event.currentTarget.open;
                      setConnectionViews((views) => ({
                        ...views,
                        [selectedSpec.id]: open,
                      }));
                    }}
                  >
                    <summary>
                      <span>Connections</span>
                      <small>
                        {selectedSources.map(sourceLabel).join(" + ")}
                      </small>
                      <ChevronRight size={14} aria-hidden="true" />
                    </summary>
                    <fieldset
                      className="connection-fields"
                      aria-label="Input connections"
                    >
                      {selectedSources.map((source, slot) => (
                        <label key={slot}>
                          {["add_join", "concat_join", "stack_join"].includes(
                            selectedSpec.kind,
                          )
                            ? `Input ${slot + 1}`
                            : "Source tensor"}
                          <select
                            aria-label={`Component input ${slot + 1} source`}
                            value={source}
                            disabled={busy}
                            aria-invalid={
                              invalidSources.includes(source) || undefined
                            }
                            onChange={(e) => {
                              const sources = [
                                ...componentSources(
                                  blueprint.components,
                                  blueprint.components.indexOf(selectedSpec),
                                ),
                              ];
                              sources[slot] = e.target.value;
                              changeComponents(
                                blueprint.components.map((c) =>
                                  c.id === selectedSpec.id
                                    ? { ...c, sources }
                                    : c,
                                ),
                              );
                            }}
                          >
                            <option value="input">Input tensor</option>
                            {blueprint.components
                              .slice(
                                0,
                                blueprint.components.indexOf(selectedSpec),
                              )
                              .map((c, i) => (
                                <option key={c.id} value={c.id}>
                                  {i + 1} ·{" "}
                                  {componentTool(c, catalog)?.title ?? c.kind}
                                </option>
                              ))}
                            {source !== "input" &&
                              !blueprint.components
                                .slice(
                                  0,
                                  blueprint.components.indexOf(selectedSpec),
                                )
                                .some((c) => c.id === source) && (
                                <option value={source}>
                                  Missing or later component — reconnect
                                </option>
                              )}
                          </select>
                        </label>
                      ))}
                      <p className="settings-note">
                        Select an earlier output to branch. Join paths with Add,
                        Concatenate, or Stack branches. The last component is
                        the model output.
                      </p>
                    </fieldset>
                  </details>
                  {selectedSpec.kind === "vit" && (
                    <p className="settings-description">
                      Starts with untrained weights. Patch tokens receive a
                      class token and learned positions. The classifier reads
                      token 0 and returns logits. Image dimensions must be
                      divisible by the patch size.
                    </p>
                  )}
                  {["hierarchical_vit", "window_attention"].includes(
                    selectedSpec.kind,
                  ) && (
                    <p className="settings-description">
                      Fixed, non-overlapping windows with shared, initially
                      untrained weights. This example has no shifted windows or
                      positional bias. Window size must divide the spatial grid.
                    </p>
                  )}
                  {selectedSpec.kind === "patch_merging" && (
                    <p className="settings-description">
                      Even height and width are required. Four neighboring
                      C-feature vectors become one 4C vector, followed by
                      normalization and a learned projection to 2C. This changes
                      values as well as shape.
                    </p>
                  )}
                  {["shifted_window", "window_pair"].includes(
                    selectedSpec.kind,
                  ) && (
                    <p className="settings-description">
                      Learned relative-position bias and a wraparound mask are
                      applied before softmax. Shift must be smaller than the
                      window. An axis with only one window is not shifted.
                      Weights start untrained; no padding or dropout is added.
                    </p>
                  )}
                  {selectedSpec.kind === "class_readout" && (
                    <p className="settings-description">
                      Reads token 0 after normalization. Add Class token +
                      positions before your transformer when using a class
                      token. Outputs are logits, before softmax.
                    </p>
                  )}
                  {!!selectedItem.parameters.length && (
                    <div className="inspector-section-title">Parameters</div>
                  )}
                  {selectedItem.parameters.map((p) => (
                    <ParameterField
                      key={p.key}
                      parameter={p}
                      value={selectedSpec.parameters[p.key] ?? p.default}
                      title={selectedItem.title}
                      kind={selectedSpec.kind}
                      shape={selectedStage?.input_shape}
                      axes={incomingAxes}
                      onChange={(value) => updateParameter(p.key, value)}
                    />
                  ))}
                  {!selectedItem.parameters.length && !selectedSpec.custom && (
                    <p className="settings-note">
                      This component has no adjustable settings.
                    </p>
                  )}
                  {selectedSpec.custom && (
                    <>
                      <div className="custom-instance-label">
                        <Code2 size={13} />
                        <code>{selectedSpec.custom.class_name}</code>
                        <span>Saved copy</span>
                      </div>
                      <ConstructorArguments
                        key={selectedSpec.id + selectedSpec.custom.id}
                        text={
                          selectedArguments?.text ??
                          JSON.stringify(
                            selectedSpec.arguments ??
                              selectedSpec.custom.constructor,
                            null,
                            2,
                          )
                        }
                        error={selectedArguments?.error ?? ""}
                        onChange={changeConstructor}
                        onRevert={() =>
                          setConstructorEdits((edits) => {
                            const next = { ...edits };
                            delete next[selectedSpec.id];
                            return next;
                          })
                        }
                      />
                      <button
                        className="secondary-button"
                        onClick={() =>
                          setCustomEditor({
                            initial: {
                              ...selectedSpec.custom!,
                              constructor:
                                selectedSpec.arguments ??
                                selectedSpec.custom!.constructor,
                            },
                            nodeId: selectedSpec.id,
                          })
                        }
                      >
                        <Code2 size={14} /> Edit source as new version
                      </button>
                      <p className="settings-note">
                        Generating the diagram checks this local module on
                        shape-only tensors first. You can preview its shapes
                        here too. Operations that depend on values need a
                        custom-code project.
                      </p>
                    </>
                  )}
                  {selectedStage && (
                    <section
                      className="connection-preview"
                      aria-label="Shape transformation"
                    >
                      <div className="inspector-section-title">
                        Tensor transformation
                      </div>
                      {(selectedStage.source_shapes?.length
                        ? selectedStage.source_shapes
                        : [selectedStage.input_shape]
                      ).map((shape, i) => (
                        <div className="connection-tensor" key={i}>
                          <span>
                            {selectedSources.length > 1 ? `IN ${i + 1}` : "IN"}
                          </span>
                          <ShapeSummary
                            shape={shape}
                            axes={axesForSource(selectedSources[i] ?? "input")}
                            compact
                          />
                        </div>
                      ))}
                      <div className="connection-direction">
                        <ArrowRight size={13} />
                        <span>{pending ? "Checking shape…" : "Output"}</span>
                      </div>
                      <div className="connection-tensor">
                        <span>OUT</span>
                        {selectedStage.shape && !pending ? (
                          <ShapeSummary
                            shape={selectedStage.shape}
                            axes={selectedStage.axes ?? []}
                            compact
                          />
                        ) : (
                          <span className="connection-unavailable">
                            {pending
                              ? "Updating…"
                              : "Resolve connection to preview"}
                          </span>
                        )}
                      </div>
                    </section>
                  )}
                  {selectedStage?.error && !plan?.validation_required && (
                    <p className="field-error" role="alert">
                      {selectedStage.error}
                    </p>
                  )}
                  <div className="component-actions">
                    <button
                      className="secondary-button"
                      aria-label="Move component earlier"
                      disabled={blueprint.components[0]?.id === selected}
                      onClick={() => move(-1)}
                    >
                      <ArrowLeft size={14} /> Earlier
                    </button>
                    <button
                      className="secondary-button"
                      aria-label="Move component later"
                      disabled={blueprint.components.at(-1)?.id === selected}
                      onClick={() => move(1)}
                    >
                      Later <ArrowRight size={14} />
                    </button>
                  </div>
                  <button
                    className="remove-component"
                    onClick={() => {
                      changeComponents(
                        blueprint.components.filter((c) => c.id !== selected),
                      );
                      setSelected(null);
                    }}
                  >
                    <Trash2 size={14} /> Remove component
                  </button>
                </>
              ) : null}
            </div>
          </aside>
        )}
      </div>
      {(error ||
        (plan?.error &&
          !plan.validation_required &&
          blueprint.has_input &&
          !firstIssue)) && (
        <div className="builder-error" role="alert">
          <CircleAlert size={14} />
          <span>{error || plan?.error}</span>
          {error && (
            <button
              disabled={busy || pending || checking || !argumentsValid}
              onClick={() =>
                checkedPlan?.signature === signature
                  ? void checkShapes()
                  : setRetry((n) => n + 1)
              }
            >
              Retry
            </button>
          )}
        </div>
      )}
      {customEditor && (
        <CustomComponentDialog
          initial={customEditor.initial}
          onClose={() => setCustomEditor(null)}
          onSave={(component) => {
            setCatalog((items) => [...items, customTool(component)]);
            if (customEditor.nodeId) {
              changeComponents(
                blueprint.components.map((c) =>
                  c.id === customEditor.nodeId
                    ? { ...c, custom: component, arguments: null }
                    : c,
                ),
              );
            } else {
              add(customTool(component));
            }
            setCustomEditor(null);
          }}
        />
      )}
      {volume && (
        <TensorVolumeDialog
          tensor={volume.tensor}
          initialIndex={volume.index}
          onClose={() => setVolume(null)}
        />
      )}
    </section>
  );
}
