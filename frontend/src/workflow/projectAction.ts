import type { Draft } from "../api/client";
import {
  compositionSignature,
  type BuildReadiness,
} from "../builder/readiness";
import { forwardInputs, forwardIssue } from "../inputs/forward";

export type ProjectAction = {
  kind: "run" | "inputs" | "code" | "model" | "tools" | "wait";
  label: string;
  detail: string;
};

type ActionState = {
  draft: Draft | null;
  hasRun: boolean;
  editorValid: boolean;
  inputsValid: boolean;
  builderEditingValid: boolean;
  build: BuildReadiness | null;
};

const PYTHON_KEYWORDS = new Set(
  "False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split(
    " ",
  ),
);

export function validClassName(name: string): boolean {
  return (
    /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) &&
    name.length <= 100 &&
    !PYTHON_KEYWORDS.has(name)
  );
}

/** The same next step drives generation controls in every project surface. */
export function projectAction({
  draft,
  hasRun,
  editorValid,
  inputsValid,
  builderEditingValid,
  build,
}: ActionState): ProjectAction {
  if (!draft)
    return {
      kind: "wait",
      label: "Generate diagram",
      detail: "Create or open a project to get started.",
    };
  if (!draft.name.trim())
    return {
      kind: draft.blueprint ? "model" : "code",
      label: "Set project name",
      detail: "Give your project a name before generating the diagram.",
    };
  if (!editorValid)
    return {
      kind: "code",
      label: "Review code",
      detail: "Finish the code settings before generating the diagram.",
    };
  if (
    !draft.blueprint &&
    (!draft.code.trim() || !validClassName(draft.class_name))
  )
    return {
      kind: "code",
      label: "Review code",
      detail: !draft.code.trim()
        ? "Add your Python model before generating the diagram."
        : "Enter the Python class to run in Model settings.",
    };
  const inputIssue = forwardIssue(forwardInputs(draft), draft.capture_mode);
  if (!inputsValid || inputIssue)
    return {
      kind: "inputs",
      label: "Review inputs",
      detail:
        inputIssue ||
        "Finish the input settings before generating the diagram.",
    };
  if (draft.blueprint) {
    if (!builderEditingValid)
      return {
        kind: "model",
        label: "Review model",
        detail: "Fix or revert the unfinished component arguments.",
      };
    if (!draft.blueprint.has_input)
      return {
        kind: "inputs",
        label: "Set up input",
        detail: "Choose the tensor that enters your model.",
      };
    if (!draft.blueprint.components.length)
      return {
        kind: "tools",
        label: "Add component",
        detail: "Choose a component from the toolbar to start your model.",
      };
    if (
      !build ||
      build.signature !== compositionSignature(draft) ||
      build.state === "checking"
    )
      return {
        kind: "wait",
        label: "Checking model…",
        detail: "Checking the connections for your current model and inputs.",
      };
    if (build.state === "invalid")
      return {
        kind: "model",
        label: "Review model",
        detail:
          build.issue ||
          "Review component connections before generating the diagram.",
      };
    if (build.state === "needs-check")
      return {
        kind: "run",
        label: hasRun ? "Run again" : "Generate diagram",
        detail:
          "Check custom component shapes, then run your model with the current inputs.",
      };
  }
  return {
    kind: "run",
    label: hasRun ? "Run again" : "Generate diagram",
    detail: "Run your model with the current inputs.",
  };
}
