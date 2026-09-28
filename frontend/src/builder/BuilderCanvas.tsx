import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Box,
  Check,
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
  type CompositionPlan,
  type Draft,
  type Tensor,
  type ToolboxItem,
} from "../api/client";
import { TensorGlyph } from "../journey/TensorGlyph";
import { TensorVolumeDialog } from "../tensors/TensorVolumeDialog";
import { previewTensor } from "./model";
import { ShapeSummary } from "../components/ShapeSummary";
import { ParameterField } from "./ParameterField";
import { ToolboxPanel, ComponentIcon, type ToolGroup } from "./Toolbox";

export function BuilderCanvas({
  draft,
  onChange,
  onValidity,
  busy,
  onShowRun,
  hasRun,
  toolboxGroup,
  onToolboxGroup,
}: {
  draft: Draft;
  onChange: (draft: Draft) => void;
  onValidity: (valid: boolean) => void;
  busy: boolean;
  onShowRun: () => void;
  hasRun: boolean;
  toolboxGroup: ToolGroup | null;
  onToolboxGroup: (group: ToolGroup | null) => void;
}) {
  const blueprint = draft.blueprint!;
  const [catalog, setCatalog] = useState<ToolboxItem[]>([]);
  const [plan, setPlan] = useState<CompositionPlan | null>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const [catalogError, setCatalogError] = useState("");
  const [selected, setSelected] = useState<string | null>(null);
  const [shapeText, setShapeText] = useState(draft.input.shape.join(", "));
  const [shapeError, setShapeError] = useState("");
  const [volume, setVolume] = useState<{
    tensor: Tensor;
    index: number;
  } | null>(null);
  const [zoom, setZoom] = useState(1);
  const [retry, setRetry] = useState(0);
  const stageElement = useRef<HTMLDivElement>(null);
  const inspectorContent = useRef<HTMLDivElement>(null);
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
  const signature = JSON.stringify([
    blueprint,
    draft.input,
    draft.capture_mode,
    draft.name,
  ]);
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
    setShapeText(draft.input.shape.join(", "));
  }, [draft.input.shape.join(",")]);
  useEffect(() => {
    const controller = new AbortController();
    setPending(true);
    setError("");
    onValidity(false);
    const timer = window.setTimeout(() => {
      api
        .compose(draft, controller.signal)
        .then((next) => {
          if (controller.signal.aborted) return;
          setPlan(next);
          setPending(false);
          if (next.code !== draft.code)
            onChange({
              ...draft,
              code: next.code,
              class_name: "ComposedModel",
              constructor: {},
            });
        })
        .catch((e) => {
          if (!controller.signal.aborted) {
            setError(e.message);
            setPending(false);
            setPlan(null);
          }
        });
    }, 180);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [signature, retry]);
  useEffect(() => {
    onValidity(!pending && !shapeError && !!plan?.valid);
  }, [pending, shapeError, plan]);
  const inputTensor = previewTensor(
    "builder-input",
    "Input",
    draft.input.shape,
    draft.input.axis_names,
  );
  const selectedSpec = blueprint.components.find((c) => c.id === selected);
  const selectedItem = catalog.find((c) => c.kind === selectedSpec?.kind);
  const selectedStage = plan?.stages.find((s) => s.id === selected);
  const inputSelected = selected === "input";
  const selectionIndex = blueprint.components.findIndex(
    (c) => c.id === selected,
  );
  const incomingAxes =
    selectionIndex <= 0
      ? draft.input.axis_names
      : (plan?.stages[selectionIndex - 1]?.axes ?? []);
  const sequenceIds = ["input", ...blueprint.components.map((c) => c.id)];
  const selectedPosition = selected ? sequenceIds.indexOf(selected) : -1;
  const firstIssue = plan?.stages.find((s) => s.error);
  function closeToolbox() {
    onToolboxGroup(null);
    document
      .querySelector<HTMLButtonElement>(".rail-tool[aria-expanded='true']")
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
          '.builder-heading [aria-label="Project setup"]',
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
    setSelected(id);
    if (id !== "project") setZoom((z) => Math.max(0.9, z));
    onToolboxGroup(null);
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
  function add(item: ToolboxItem) {
    if (busy || blueprint.components.length >= 16) return;
    const id = crypto.randomUUID();
    const input =
      !blueprint.has_input &&
      (item.group === "Spatial" ||
        item.kind === "patch_embedding" ||
        item.kind === "batchnorm2d" ||
        item.kind === "tokens")
        ? {
            ...draft.input,
            shape: [2, 3, 8, 8],
            axis_names: ["batch", "channels", "height", "width"],
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
            parameters: Object.fromEntries(
              item.parameters.map((p) => [p.key, p.default]),
            ),
          },
        ],
      },
    });
    selectComponent(id);
    setShapeError("");
  }
  function shapeCommit() {
    const parts = shapeText.split(/[,\s]+/).filter(Boolean);
    const shape = parts.map(Number);
    if (
      !parts.length ||
      parts.length > 6 ||
      parts.some((p) => !/^\d+$/.test(p)) ||
      shape.some((n) => !Number.isSafeInteger(n) || n < 1) ||
      shape.reduce((a, b) => a * b, 1) > 2 ** 40
    ) {
      setShapeError("Use 1–6 positive dimensions, up to 2^40 elements.");
      return;
    }
    setShapeError("");
    const axes =
      shape.length === draft.input.axis_names.length
        ? draft.input.axis_names
        : shape.length === 4
          ? ["batch", "channels", "height", "width"]
          : shape.length === 3
            ? ["batch", "tokens", "features"]
            : shape.map((_, i) => (i === 0 ? "batch" : `axis ${i}`));
    // Large designs remain valid drafts; numeric allocation is an explicit mode choice.
    onChange({
      ...draft,
      input: { ...draft.input, shape, axis_names: axes },
      capture_mode:
        shape.reduce((a, b) => a * b, 1) > 8_388_608
          ? "shapes"
          : draft.capture_mode,
    });
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
            aria-label="Project setup"
            aria-pressed={selected === "project"}
            onClick={() => selectComponent("project")}
          >
            <SlidersHorizontal size={14} /> Setup
          </button>
          {hasRun && (
            <button className="secondary-button" onClick={onShowRun}>
              View last run <ArrowRight size={14} />
            </button>
          )}
        </div>
      </header>
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
                className="builder-sequence"
                aria-label="Connected component sequence"
                style={{ zoom }}
              >
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
                  const item = catalog.find((c) => c.kind === component.kind),
                    stage = plan?.stages.find((s) => s.id === component.id);
                  const tensor = stage?.shape
                    ? previewTensor(
                        component.id,
                        item?.title ?? component.kind,
                        stage.shape,
                        stage.axes ?? [],
                      )
                    : null;
                  return (
                    <div className="builder-link" key={component.id}>
                      <div className="builder-connector" aria-hidden="true">
                        <span />
                        <ArrowRight size={15} />
                      </div>
                      <article
                        className={`builder-node ${selected === component.id ? "selected" : ""} ${stage?.error ? "invalid" : ""}`}
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
                            {pending ? (
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
                          {pending ? (
                            <span>Checking output…</span>
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
                    aria-label={`Go to ${catalog.find((item) => item.kind === c.kind)?.title ?? c.kind}, step ${i + 1}`}
                    aria-current={selected === c.id ? "step" : undefined}
                    className={
                      plan?.stages.find((s) => s.id === c.id)?.error
                        ? "has-issue"
                        : ""
                    }
                    onClick={() => selectComponent(c.id)}
                  >
                    <span className="sequence-number">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <span>
                      {catalog.find((item) => item.kind === c.kind)?.title ??
                        c.kind}
                    </span>
                  </button>
                ))}
              </nav>
            )}
            <div className="builder-status">
              <span
                className={`builder-readiness ${firstIssue || error || shapeError ? "has-issue" : ""}`}
                role="status"
              >
                {pending ? (
                  <LoaderCircle size={13} className="spin" />
                ) : plan?.valid && !shapeError ? (
                  <Check size={13} />
                ) : (
                  <CircleAlert size={13} />
                )}
                {pending
                  ? "Checking connections"
                  : error
                    ? "Connection check unavailable"
                    : shapeError
                      ? "Apply input changes"
                      : plan?.valid
                        ? "Ready to run"
                        : blueprint.has_input
                          ? "Check your connections"
                          : "Start with an input"}
                <span className="preview-disclaimer">
                  {plan?.valid && !pending && !shapeError
                    ? "Shapes preview · run to inspect values"
                    : ""}
                </span>
                {(firstIssue || shapeError) && !pending && (
                  <button
                    onClick={() =>
                      selectComponent(shapeError ? "input" : firstIssue!.id)
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
        </div>
        {selected && (
          <aside
            className="builder-settings"
            aria-label={
              selected === "project" ? "Project setup" : "Component settings"
            }
          >
            <header className="inspector-header">
              <div className="inspector-eyebrow">
                <span>
                  {selected === "project"
                    ? "EXPERIMENT"
                    : inputSelected
                      ? "STARTING TENSOR"
                      : `COMPONENT ${String(selectionIndex + 1).padStart(2, "0")}`}
                </span>
                <button
                  className="icon-button"
                  aria-label="Close component settings"
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
                    ? "Project setup"
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
                    Configure this experiment before running it.
                  </p>
                  <label>
                    Project name
                    <input
                      aria-label="Project name"
                      maxLength={100}
                      value={draft.name}
                      onChange={(e) =>
                        onChange({ ...draft, name: e.target.value })
                      }
                    />
                  </label>
                  <label>
                    Recording mode
                    <select
                      aria-label="Project recording mode"
                      value={draft.capture_mode ?? "values"}
                      onChange={(e) =>
                        onChange({
                          ...draft,
                          capture_mode: e.target.value as "values" | "shapes",
                        })
                      }
                    >
                      <option value="values">Values & shapes</option>
                      <option value="shapes">Shapes only</option>
                    </select>
                  </label>
                  <p className="settings-note">
                    Shapes only supports large tensors without allocating their
                    values. Values & shapes records the actual computation.
                  </p>
                  <div className="setup-summary">
                    <span>
                      Input <code>[{draft.input.shape.join(", ")}]</code>
                    </span>
                    <span>
                      Components <b>{blueprint.components.length}</b>
                    </span>
                    <span>
                      Execution <b>CPU · evaluation</b>
                    </span>
                  </div>
                  <button className="secondary-button" onClick={addInput}>
                    <Box size={14} /> Configure input
                  </button>
                </>
              ) : inputSelected ? (
                <>
                  <div className="builder-presets">
                    <button
                      onClick={() => {
                        onChange({
                          ...draft,
                          input: {
                            ...draft.input,
                            shape: [2, 4, 8],
                            axis_names: ["batch", "tokens", "features"],
                          },
                        });
                        setShapeError("");
                      }}
                    >
                      Sequence
                    </button>
                    <button
                      onClick={() => {
                        onChange({
                          ...draft,
                          input: {
                            ...draft.input,
                            shape: [2, 3, 28, 28],
                            axis_names: [
                              "batch",
                              "channels",
                              "height",
                              "width",
                            ],
                          },
                        });
                        setShapeError("");
                      }}
                    >
                      Images
                    </button>
                  </div>
                  <label>
                    Shape
                    <input
                      aria-label="Builder input shape"
                      value={shapeText}
                      onChange={(e) => {
                        setShapeText(e.target.value);
                        setShapeError(
                          "Apply the new shape with Enter or by leaving the field.",
                        );
                      }}
                      onBlur={shapeCommit}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") shapeCommit();
                      }}
                    />
                  </label>
                  {shapeError && (
                    <p className="field-error" role="alert">
                      {shapeError}
                    </p>
                  )}
                  <label>
                    Values
                    <select
                      value={draft.input.generator}
                      onChange={(e) =>
                        onChange({
                          ...draft,
                          input: {
                            ...draft.input,
                            generator: e.target
                              .value as Draft["input"]["generator"],
                          },
                        })
                      }
                    >
                      <option value="arange">Sequential numbers</option>
                      <option
                        value="random"
                        disabled={draft.input.dtype === "int64"}
                      >
                        Seeded random
                      </option>
                      <option value="zeros">Zeros</option>
                      <option value="ones">Ones</option>
                    </select>
                  </label>
                  <div className="settings-field-row">
                    <label>
                      Data type
                      <select
                        aria-label="Input data type"
                        value={draft.input.dtype}
                        onChange={(e) =>
                          onChange({
                            ...draft,
                            input: {
                              ...draft.input,
                              dtype: e.target.value as Draft["input"]["dtype"],
                              generator:
                                e.target.value === "int64" &&
                                draft.input.generator === "random"
                                  ? "arange"
                                  : draft.input.generator,
                            },
                          })
                        }
                      >
                        <option value="float32">float32</option>
                        <option value="float64">float64</option>
                        <option value="int64">int64</option>
                      </select>
                    </label>
                    <label>
                      Random seed
                      <input
                        aria-label="Input random seed"
                        type="number"
                        min={0}
                        max={4294967295}
                        value={draft.input.seed}
                        onChange={(e) => {
                          const seed = Number(e.target.value);
                          if (
                            Number.isInteger(seed) &&
                            seed >= 0 &&
                            seed <= 4294967295
                          )
                            onChange({
                              ...draft,
                              input: { ...draft.input, seed },
                            });
                        }}
                      />
                    </label>
                  </div>
                  <label>
                    Recording mode
                    <select
                      value={draft.capture_mode ?? "values"}
                      onChange={(e) =>
                        onChange({
                          ...draft,
                          capture_mode: e.target.value as "values" | "shapes",
                        })
                      }
                    >
                      <option value="values">Values & shapes</option>
                      <option value="shapes">Shapes only</option>
                    </select>
                  </label>
                  <p className="settings-note">
                    Dimensions follow the axis order shown on the node. Large
                    dimensions use indexed gaps; every index remains accessible.
                  </p>
                </>
              ) : selectedSpec && selectedItem ? (
                <>
                  <p className="settings-description">
                    {selectedItem.description}
                  </p>
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
                  {!selectedItem.parameters.length && (
                    <p className="settings-note">
                      This component has no adjustable settings.
                    </p>
                  )}
                  {selectedStage && (
                    <section
                      className="connection-preview"
                      aria-label="Shape transformation"
                    >
                      <div className="inspector-section-title">
                        Tensor transformation
                      </div>
                      <div className="connection-tensor">
                        <span>IN</span>
                        <ShapeSummary
                          shape={selectedStage.input_shape}
                          axes={incomingAxes}
                          compact
                        />
                      </div>
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
                  {selectedStage?.error && (
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
      {(error || (plan?.error && blueprint.has_input)) && (
        <div className="builder-error" role="alert">
          <CircleAlert size={14} />
          <span>{error || plan?.error}</span>
          {error && (
            <button onClick={() => setRetry((n) => n + 1)}>Retry</button>
          )}
        </div>
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
