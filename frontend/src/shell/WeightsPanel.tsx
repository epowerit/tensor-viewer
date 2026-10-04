import { useEffect, useMemo, useState } from "react";
import {
  api,
  isWhatIf,
  type Run,
  type WeightReport,
  type WeightSpectrum,
} from "../api/client";
import { formatValue } from "../tensors/coordinates";
import { tensorUses } from "../tensors/TensorUseContext";

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
  onSelect,
}: {
  run: Run | null;
  previousRunId?: string | null;
  onSelect?: (node: string) => void;
}) {
  const whatIf = isWhatIf(run?.id);
  // A what-if is compared with its recorded run from the start.
  const [comparing, setComparing] = useState(whatIf);
  useEffect(() => setComparing(isWhatIf(run?.id)), [run?.id]);
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
    return (
      <p className="panel-empty">Reading the weights' spectra… {toggle}</p>
    );
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
        This run read no weights with values (a shapes-only run records none).
      </p>
    );
  const total = weights.reduce((sum, weight) => sum + weight.numel, 0);
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
      onClick: () => reader && onSelect?.(reader.id),
      title: reader
        ? `Read at step ${reader.step} (${reader.kind})`
        : undefined,
    };
  };
  if (against) {
    const changed = weights
      .filter((weight) => weight.update)
      .sort((a, b) => (b.update!.relative ?? 0) - (a.update!.relative ?? 0));
    const moved = changed.filter((weight) => weight.update!.norm > 0);
    // The headline is the most changed matrix: a small bias changes by a
    // large share without saying much, and has no rank.
    const most = moved.find((weight) => weight.singular.length) ?? moved[0];
    return (
      <div className="weights-panel">
        <p className="weights-summary">
          {moved.length} of {weights.length} weights changed
          {most?.update &&
            ` · most changed ${most.singular.length ? "matrix" : "weight"}: ${most.name}, by ${percent(most.update.relative ?? 0)}${most.update.effective_rank != null && most.update.full ? `, an update of effective rank ${formatValue(most.update.effective_rank)} of ${most.update.full}` : ""}`}{" "}
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
          <tbody>
            {changed.map((weight) => {
              const update = weight.update!;
              return (
                <tr
                  key={weight.tensor_id}
                  className={update.norm > 0 ? undefined : "weight-still"}
                  {...open(weight)}
                >
                  <td className="weight-name">{weight.name}</td>
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
            })}
          </tbody>
        </table>
      </div>
    );
  }
  return (
    <div className="weights-panel">
      <p className="weights-summary">
        {weights.length} weights and buffers · {count(total)} values
        {weights.some(flagged) &&
          ` · ${weights.filter(flagged).length} rank-deficient or badly conditioned`}{" "}
        {toggle}
      </p>
      <table>
        <thead>
          <tr>
            <th>Weight</th>
            <th>Shape</th>
            <th>Params</th>
            <th title="Frobenius norm">‖W‖</th>
            <th title="Largest singular value">σ max</th>
            <th title="Largest over smallest singular value">Condition</th>
            <th title="Singular values above float32 rounding of the largest, of the most there can be">
              Rank
            </th>
            <th title="exp(entropy) of the normalized singular values: how many directions it really uses">
              Eff. rank
            </th>
            <th>Spectrum</th>
          </tr>
        </thead>
        <tbody>
          {[...weights]
            .sort((a, b) => b.numel - a.numel)
            .map((weight) => (
              <tr
                key={weight.tensor_id}
                className={flagged(weight) ? "weight-flagged" : undefined}
                {...open(weight)}
              >
                <td className="weight-name">{weight.name}</td>
                <td>[{weight.shape.join(", ")}]</td>
                <td>{count(weight.numel)}</td>
                <td>{formatValue(weight.norm)}</td>
                <td>
                  {weight.singular.length
                    ? formatValue(weight.singular[0])
                    : "—"}
                </td>
                <td>
                  {!weight.singular.length
                    ? "—"
                    : weight.condition == null
                      ? "∞"
                      : formatValue(weight.condition)}
                </td>
                <td>
                  {weight.rank != null
                    ? `${weight.rank} / ${weight.full}`
                    : "—"}
                </td>
                <td>
                  {weight.effective_rank != null
                    ? formatValue(weight.effective_rank)
                    : "—"}
                </td>
                <td>
                  <Spectrum values={weight.singular} />
                </td>
              </tr>
            ))}
        </tbody>
      </table>
    </div>
  );
}
