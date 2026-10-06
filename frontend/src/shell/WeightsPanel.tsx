import { useEffect, useMemo, useRef, useState } from "react";
import {
  api,
  isWhatIf,
  type Run,
  type WeightReport,
  type WeightSpectrum,
} from "../api/client";
import { formatValue } from "../tensors/coordinates";
import { tensorUses } from "../tensors/TensorUseContext";
import { groupWeights } from "../tensors/weightGroups";
import { useStepPreview } from "./stepPreview";
import { PanelLoading } from "./PanelLoading";

/** Reports already asked for, by run and the run compared with. */
const reports = new Map<string, Promise<WeightReport>>();

/** A badly conditioned matrix: its directions differ in scale this much. */
const ILL = 1e4;

const count = (n: number) =>
  n >= 1e6
    ? `${(n / 1e6).toPrecision(3)}M`
    : n >= 1e3
      ? `${(n / 1e3).toPrecision(3)}k`
      : String(n);
const percent = (share: number) =>
  share >= 0.1
    ? `${Math.round(share * 100)}%`
    : share >= 0.001
      ? `${(share * 100).toPrecision(2)}%`
      : share > 0
        ? `${(share * 100).toExponential(0)}%`
        : "0";

/**
 * Singular values, largest first, as bars on a log scale: a long tail shows
 * directions barely used.
 */
function Spectrum({ values }: { values: number[] }) {
  const positive = values.filter((value) => value > 0);
  if (!positive.length) return null;
  const top = Math.log10(positive[0]);
  const bottom = Math.min(top - 1, Math.log10(positive.at(-1)!));
  const width = 96,
    height = 18;
  const bar = width / values.length;
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      aria-hidden="true"
    >
      {values.map((value, at) => {
        const h =
          value > 0
            ? Math.max(
                1,
                ((Math.log10(value) - bottom) / (top - bottom)) * height,
              )
            : 0;
        return (
          <rect
            key={at}
            x={at * bar}
            y={height - h}
            width={Math.max(0.6, bar - 0.4)}
            height={h}
          />
        );
      })}
    </svg>
  );
}

type Weight = WeightSpectrum & { singular: number[] };

/**
 * The model's weights (and buffers, such as a causal mask, which the run
 * records alike), largest first: shape, size, norm, and as a matrix its
 * singular values, condition number, numerical and effective rank. A weight
 * that does not use all its directions, or whose directions differ hugely in
 * scale, is marked. With an earlier run to compare with (the recorded run of
 * a what-if, or the run before), **compare** shows what changed in each
 * weight instead: the update W − W₀, how large it is against the weight, and
 * its own spectrum and rank, most changed first. A row opens the first step
 * that reads the weight.
 */
export function WeightsPanel({
  run,
  previousRunId = null,
  selected = null,
  onSelect,
  onPreview,
}: {
  run: Run | null;
  previousRunId?: string | null;
  /** The step playback is on, whose weights are marked. */
  selected?: string | null;
  onSelect?: (node: string) => void;
  /** Trace the step reading a weight while its row is under the pointer. */
  onPreview?: (id: string | null) => void;
}) {
  const { rowPreview } = useStepPreview(onPreview);
  const whatIf = isWhatIf(run?.id);
  // A what-if is compared with its recorded run from the start.
  const [comparing, setComparing] = useState(whatIf);
  useEffect(() => setComparing(isWhatIf(run?.id)), [run?.id]);
  // The weights in the order the model reads them, by layer, or largest first.
  const [order, setOrder] = useState<"model" | "size">("model");
  const against = comparing && previousRunId ? previousRunId : null;
  const key = `${run?.id}\n${against ?? ""}`;
  const [report, setReport] = useState<{
    key: string;
    found?: WeightReport;
    error?: string;
  } | null>(null);
  useEffect(() => {
    if (!run) return;
    let live = true;
    let asked = reports.get(key);
    if (!asked) {
      asked = api.weightSpectra(run.id, against);
      asked.catch(() => reports.delete(key));
      reports.set(key, asked);
    }
    asked
      .then((found) => live && setReport({ key, found }))
      .catch(
        (error: Error) => live && setReport({ key, error: error.message }),
      );
    return () => {
      live = false;
    };
    // `key` names the run and the run compared with.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  const readers = useMemo(() => (run ? tensorUses(run.trace) : null), [run]);
  // The weights the step playback is on reads.
  const used = useMemo(
    () =>
      new Set(
        run?.trace.operations.find((op) => op.id === selected)?.inputs ?? [],
      ),
    [run, selected],
  );
  // Playback moving to another step brings its weights into view.
  const panel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    panel.current
      ?.querySelector("tr.is-current")
      ?.scrollIntoView({ block: "nearest" });
  }, [selected, report]);
  const classes = (...names: (string | false | undefined)[]) =>
    names.filter(Boolean).join(" ") || undefined;
  if (!run)
    return <p className="panel-empty">Run to see the model's weights.</p>;
  const current = report?.key === key ? report : null;
  const toggle = previousRunId && (
    <button
      type="button"
      className="weights-compare"
      aria-pressed={comparing}
      onClick={() => setComparing((on) => !on)}
      title="Show what changed in each weight since that run: W − W₀ and its spectrum"
    >
      compare with {whatIf ? "the recorded run" : "the run before"}
    </button>
  );
  if (!current)
    return <PanelLoading>Reading the weights' spectra… {toggle}</PanelLoading>;
  if (current.error || current.found?.error)
    return (
      <p className="panel-empty">{current.error ?? current.found?.error}</p>
    );
  const weights: Weight[] = (current.found?.weights ?? []).map((weight) => ({
    ...weight,
    singular: weight.singular ?? [],
  }));
  if (!weights.length)
    return (
      <p className="panel-empty">
        {run.project.capture_mode === "shapes"
          ? "A shapes-only run records no values, so there are no weights to read. Record values to see them."
          : "This model has no weights: nothing it ran reads a learned parameter."}
      </p>
    );
  // Buffers (a causal mask, running statistics) are recorded like weights
  // but not learned: they go last, apart, and an update never touches them.
  // Runs recorded before buffers were told apart list everything together.
  const known = run.trace.buffer_names;
  const buffers = new Set(known ?? []);
  const learned = weights.filter((weight) => !buffers.has(weight.name));
  const kept = weights.filter((weight) => buffers.has(weight.name));
  const total = learned.reduce((sum, weight) => sum + weight.numel, 0);
  // Only a matrix has a rank or a condition to worry about.
  const flagged = (weight: Weight) =>
    weight.singular.length > 0 &&
    ((weight.rank != null &&
      weight.full != null &&
      weight.rank < weight.full) ||
      weight.condition == null ||
      weight.condition > ILL);
  const open = (weight: Weight) => {
    const reader = readers?.(weight.tensor_id).read[0];
    return {
      ...rowPreview(reader?.id),
      onClick: () => reader && onSelect?.(reader.id),
      title: reader
        ? `Read at step ${reader.step} (${reader.kind})`
        : undefined,
    };
  };
  // A group's name, as the canvas names a block (blocks.0 › attention);
  // the model's own top-level weights go under its class name.
  const model = run.project.class_name || "Model";
  const crumbs = (path: string) =>
    path
      ? path
          .split(".")
          .reduce<string[]>(
            (parts, part) =>
              /^\d+$/.test(part) && parts.length
                ? [...parts.slice(0, -1), `${parts.at(-1)}.${part}`]
                : [...parts, part],
            [],
          )
          .join(" › ")
      : model;
  const firstRead = (weight: Weight) =>
    readers?.(weight.tensor_id).read[0]?.step ?? null;
  // The order switch, in both views: by layer, or the view's own ranking.
  const orderSwitch = (other: string, title: string) => (
    <span className="weights-order" role="group" aria-label="Order the weights">
      <button
        type="button"
        aria-pressed={order === "model"}
        onClick={() => setOrder("model")}
        title="In the order the run reads them, grouped by layer, as on the canvas"
      >
        by layer
      </button>
      <button
        type="button"
        aria-pressed={order === "size"}
        onClick={() => setOrder("size")}
        title={title}
      >
        {other}
      </button>
    </span>
  );
  if (against) {
    const changed = learned
      .filter((weight) => weight.update)
      .sort((a, b) => (b.update!.relative ?? 0) - (a.update!.relative ?? 0));
    const moved = changed.filter((weight) => weight.update!.norm > 0);
    // The headline is the most changed matrix: a small bias changes by a
    // large share without saying much, and has no rank.
    const most = moved.find((weight) => weight.singular.length) ?? moved[0];
    const updateRow = (weight: Weight, label = weight.name) => {
      const update = weight.update!;
      return (
        <tr
          key={weight.tensor_id}
          className={classes(
            update.norm <= 0 && "weight-still",
            used.has(weight.tensor_id) && "is-current",
          )}
          {...open(weight)}
        >
          <td className="weight-name" title={weight.name}>
            {label}
          </td>
          <td>[{weight.shape.join(", ")}]</td>
          <td>{percent(update.relative ?? 0)}</td>
          <td>{formatValue(update.norm)}</td>
          <td>
            {update.rank != null && update.norm > 0
              ? `${update.rank} / ${update.full}`
              : "—"}
          </td>
          <td>
            {update.effective_rank != null && update.norm > 0
              ? formatValue(update.effective_rank)
              : "—"}
          </td>
          <td>
            <Spectrum values={update.singular ?? []} />
          </td>
        </tr>
      );
    };
    return (
      <div className="weights-panel" ref={panel}>
        <p className="weights-summary">
          {moved.length} of {learned.length} weights changed
          {most?.update &&
            ` · most changed ${most.singular.length ? "matrix" : "weight"}: ${most.name}, by ${percent(most.update.relative ?? 0)}${most.update.effective_rank != null && most.update.full ? `, an update of effective rank ${formatValue(most.update.effective_rank)} of ${most.update.full}` : ""}`}{" "}
          {orderSwitch("most changed", "Most changed first, by ‖ΔW‖ / ‖W₀‖")}{" "}
          {toggle}
        </p>
        <table>
          <thead>
            <tr>
              <th>Weight</th>
              <th>Shape</th>
              <th title="‖W − W₀‖ / ‖W₀‖">Change</th>
              <th title="‖W − W₀‖">‖ΔW‖</th>
              <th title="Singular values of the update above float32 rounding of its largest, of the most there can be">
                Update rank
              </th>
              <th title="exp(entropy) of the update's normalized singular values: how many directions the learning went into">
                Eff. rank
              </th>
              <th>Update spectrum</th>
            </tr>
          </thead>
          {order === "size" ? (
            <tbody>{changed.map((weight) => updateRow(weight))}</tbody>
          ) : (
            groupWeights(changed, firstRead).map((group, at) => (
              <tbody key={`${group.path}-${at}`}>
                <tr className="weights-group">
                  <th colSpan={7}>{crumbs(group.path)}</th>
                </tr>
                {group.weights.map(({ weight, label }) =>
                  updateRow(weight, label),
                )}
              </tbody>
            ))
          )}
        </table>
      </div>
    );
  }
  // How many independent directions a matrix really uses, of the most it
  // could: its effective rank as a bar, and a numerical rank short of full.
  const directions = (weight: Weight) => {
    if (
      !weight.singular.length ||
      weight.effective_rank == null ||
      !weight.full
    )
      return null;
    const share = Math.max(0, Math.min(1, weight.effective_rank / weight.full));
    const short = weight.rank != null && weight.rank < weight.full;
    return (
      <span
        className="weight-directions"
        title={`Effective rank ${formatValue(weight.effective_rank)} of ${weight.full}: exp(entropy) of its normalized singular values. Numerical rank ${weight.rank ?? "?"} of ${weight.full}: singular values above float32 rounding of the largest.`}
      >
        <span className="weight-bar" aria-hidden="true">
          <span style={{ width: `${share * 100}%` }} />
        </span>
        {formatValue(weight.effective_rank)} of {weight.full}
        {short && <em> · rank {weight.rank}</em>}
      </span>
    );
  };
  const row = (weight: Weight, label = weight.name, buffer = false) => {
    const matrix = weight.singular.length > 0;
    return (
      <tr
        key={weight.tensor_id}
        className={classes(
          buffer ? "weight-buffer" : flagged(weight) && "weight-flagged",
          !matrix && "weight-vector",
          used.has(weight.tensor_id) && "is-current",
        )}
        {...open(weight)}
      >
        <td className="weight-name" title={weight.name}>
          {label}
        </td>
        <td>[{weight.shape.join(", ")}]</td>
        <td>{count(weight.numel)}</td>
        <td>{formatValue(weight.norm)}</td>
        <td>{matrix ? formatValue(weight.singular[0]) : ""}</td>
        <td>
          {!matrix
            ? ""
            : weight.condition == null
              ? "∞"
              : formatValue(weight.condition)}
        </td>
        <td>{directions(weight)}</td>
        <td>
          <Spectrum values={weight.singular} />
        </td>
      </tr>
    );
  };
  const flaggedCount = learned.filter(flagged).length;
  const COLUMNS = 8;
  return (
    <div className="weights-panel" ref={panel}>
      <p className="weights-summary">
        {known
          ? `${learned.length} ${learned.length === 1 ? "weight" : "weights"} · ${count(total)} values${kept.length ? ` · ${kept.length} ${kept.length === 1 ? "buffer" : "buffers"}` : ""}`
          : `${weights.length} weights and buffers · ${count(total)} values`}
        {flaggedCount > 0 &&
          ` · ${flaggedCount} rank-deficient or badly conditioned`}{" "}
        {orderSwitch("largest first", "Largest first")} {toggle}
      </p>
      <table>
        <thead>
          <tr>
            <th>Weight</th>
            <th>Shape</th>
            <th>Params</th>
            <th title="Frobenius norm: its overall size">‖W‖</th>
            <th title="Largest singular value: how far it can stretch a vector">
              σ max
            </th>
            <th title="Largest over smallest singular value: how unevenly it stretches">
              Condition
            </th>
            <th title="How many independent directions it really uses (effective rank), of the most it could">
              Directions used
            </th>
            <th>Spectrum</th>
          </tr>
        </thead>
        {order === "size" ? (
          <tbody>
            {[...learned]
              .sort((a, b) => b.numel - a.numel)
              .map((weight) => row(weight))}
          </tbody>
        ) : (
          groupWeights(learned, firstRead).map((group, at) => (
            <tbody key={`${group.path}-${at}`}>
              <tr className="weights-group">
                <th colSpan={COLUMNS}>{crumbs(group.path)}</th>
              </tr>
              {group.weights.map(({ weight, label }) => row(weight, label))}
            </tbody>
          ))
        )}
        {kept.length > 0 && (
          <tbody>
            <tr className="weights-group">
              <th colSpan={COLUMNS}>
                Buffers · recorded with the model, not learned
              </th>
            </tr>
            {[...kept]
              .sort((a, b) => b.numel - a.numel)
              .map((weight) => row(weight, weight.name, true))}
          </tbody>
        )}
      </table>
    </div>
  );
}
