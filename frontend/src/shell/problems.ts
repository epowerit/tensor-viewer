import type { Run } from "../api/client";
import { needsValues } from "../console/script";
import {
  diagnose,
  diagnoseError,
  recordedNames,
  type Diagnosis,
} from "../operations/diagnosis";
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
  /** A one-click change to the code that resolves it. */
  fix?: { label: string; apply: () => void };
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
    /**
     * Where a line of the run's code is in the code now, edited since; null
     * when it was deleted. Notes about the run point at their code.
     */
    place?: (file: string | null, line: number) => number | null;
  } = {},
): Problem[] {
  const placed = (problem: Problem): Problem =>
    options.place && problem.line !== null
      ? { ...problem, line: options.place(problem.file, problem.line) }
      : problem;
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
      const diagnosis = failed
        ? diagnose(failed, run.trace.tensors, run.trace.operations)
        : diagnoseError(error, recordedNames(run.trace));
      problems.push(
        placed({
          id: "error",
          severity: "error",
          title: diagnosis?.title ?? error.type,
          detail: `${error.type}: ${error.message}`,
          file: error.file ?? entryPath(run.project),
          line: error.line ?? null,
          node: failed?.id ?? null,
          diagnosis,
        }),
      );
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
      ? diagnose(
          failed,
          options.check.trace.tensors,
          options.check.trace.operations,
        )
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
          : placed(insight),
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

/** Notes that say the same thing about different tensors, shown once. */
export type ProblemGroup = {
  key: string;
  severity: Problem["severity"];
  /** The shared title, such as "Computed but never used". */
  title: string;
  detail: string;
  /** What each member names, such as "active", in the order recorded. */
  members: { problem: Problem; label: string }[];
};

/**
 * Notes of one severity whose title differs only in the name after a colon
 * ("Computed but never used: active", "…: doubled") and whose explanation is
 * the same, in one group; any other note stands alone. Notes with a fix or a
 * shape diagnosis always stand alone, since those differ per note.
 */
export function groupProblems(problems: Problem[]): (Problem | ProblemGroup)[] {
  const shared = (problem: Problem) => {
    const colon = problem.title.indexOf(": ");
    if (colon < 0 || problem.fix || problem.diagnosis) return null;
    return {
      key: `${problem.severity}\n${problem.title.slice(0, colon)}\n${problem.detail}`,
      title: problem.title.slice(0, colon),
      label: problem.title.slice(colon + 2),
    };
  };
  const counts = new Map<string, number>();
  for (const problem of problems) {
    const key = shared(problem)?.key;
    if (key) counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  const groups = new Map<string, ProblemGroup>();
  const items: (Problem | ProblemGroup)[] = [];
  for (const problem of problems) {
    const found = shared(problem);
    if (!found || counts.get(found.key)! < 2) {
      items.push(problem);
      continue;
    }
    let group = groups.get(found.key);
    if (!group) {
      group = {
        key: found.key,
        severity: problem.severity,
        title: found.title,
        detail: problem.detail,
        members: [],
      };
      groups.set(found.key, group);
      items.push(group);
    }
    group.members.push({ problem, label: found.label });
  }
  return items;
}
