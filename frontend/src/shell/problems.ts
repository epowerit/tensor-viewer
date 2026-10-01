import type { Run } from "../api/client";
import { needsValues } from "../console/script";
import { diagnose, type Diagnosis } from "../operations/diagnosis";
import { entryPath } from "../sources/files";
import { tensorInsights } from "./insights";

export type Problem = {
  id: string;
  severity: "error" | "warning" | "info";
  title: string;
  detail: string;
  file: string | null;
  line: number | null;
  /** Journey node to show when the problem is opened. */
  node: string | null;
  diagnosis: Diagnosis | null;
};

/** Everything that needs attention in the draft and the displayed run. */
export function collectProblems(
  run: Run | null,
  options: {
    inputIssue?: string;
    stale?: boolean;
    /** A shapes-only check of the current code, when one is up to date. */
    check?: Run | null;
    /** Tensor insights about the run or check; on unless turned off. */
    insights?: boolean;
  } = {},
): Problem[] {
  const problems: Problem[] = [];
  const base = { file: null, line: null, node: null, diagnosis: null };
  if (options.inputIssue)
    problems.push({
      ...base,
      id: "input",
      severity: "error",
      title: "Input settings",
      detail: options.inputIssue,
    });
  if (run) {
    const error = run.trace.error;
    const failed = [...run.trace.operations]
      .reverse()
      .find((op) => op.status === "error");
    if (error) {
      const diagnosis = failed ? diagnose(failed, run.trace.tensors) : null;
      problems.push({
        id: "error",
        severity: "error",
        title: diagnosis?.title ?? error.type,
        detail: `${error.type}: ${error.message}`,
        file: error.file ?? entryPath(run.project),
        line: error.line ?? null,
        node: failed?.id ?? null,
        diagnosis,
      });
    }
    run.trace.warnings?.forEach((warning, i) =>
      problems.push({
        ...base,
        id: `warning-${i}`,
        severity: "warning",
        title: "Tracking notice",
        detail: warning,
      }),
    );
    if (run.trace.weight_check && !run.trace.weight_check.compatible)
      problems.push({
        ...base,
        id: "weights",
        severity: "error",
        title: "Checkpoint does not match the model",
        detail: run.trace.weight_check.issues.join(" "),
      });
    if (options.stale)
      problems.push({
        ...base,
        id: "stale",
        severity: "info",
        title: "Edited since this run",
        detail:
          "The canvas shows a saved run. Run again to record the current code and inputs.",
      });
  }
  const checked = options.check?.trace.error;
  if (options.check && checked) {
    const failed = [...options.check.trace.operations]
      .reverse()
      .find((op) => op.status === "error");
    const diagnosis = failed
      ? diagnose(failed, options.check.trace.tensors)
      : null;
    const limited = needsValues(checked);
    problems.unshift({
      id: "shape-check",
      // The code has not run; a failure here is what a run would hit.
      severity: limited ? "info" : "warning",
      title: limited
        ? "Shape check stopped: this step needs values"
        : `Shape check: ${diagnosis?.title ?? checked.type}`,
      detail: limited
        ? "The shape of this result depends on tensor values, which a shape check does not have. Run to see it and everything after it."
        : `${checked.type}: ${checked.message}`,
      file: checked.file ?? entryPath(options.check.project),
      line: checked.line ?? null,
      node: null,
      diagnosis: limited ? null : diagnosis,
    });
  }
  // Lint the code on screen: a current shape check when the run is out of date.
  const analyzed = options.check ?? run;
  if (analyzed && options.insights !== false)
    problems.push(
      ...tensorInsights(analyzed).map((insight) =>
        options.check
          ? // A checked step has no counterpart on the canvas yet.
            { ...insight, node: null, title: `Shape check: ${insight.title}` }
          : insight,
      ),
    );
  return problems;
}

export function problemCounts(problems: Problem[]) {
  return {
    errors: problems.filter((item) => item.severity === "error").length,
    warnings: problems.filter((item) => item.severity === "warning").length,
  };
}
