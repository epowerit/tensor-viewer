/** Python operator methods, named the way the code spells them. */
const OPERATORS: Record<string, string> = {
  __getitem__: "index",
  __setitem__: "index assign",
  __invert__: "invert",
  __neg__: "negate",
  __and__: "and",
  __or__: "or",
  __xor__: "xor",
  __radd__: "add",
  __rsub__: "subtract",
  __rmul__: "multiply",
  __rdiv__: "divide",
  __rtruediv__: "divide",
  __rfloordiv__: "floor divide",
  __rpow__: "power",
  __rmatmul__: "matmul",
  __rmod__: "remainder",
};

/**
 * The step name shown to people. `__getitem__` is how torch reports `x[:, 0]`;
 * the screen says what the code did instead. Other kinds are already the
 * function the code called.
 */
export function kindName(kind: string): string {
  return (
    OPERATORS[kind] ??
    (/^__\w+__$/.test(kind) ? kind.slice(2, -2).replace(/_/g, " ") : kind)
  );
}

/**
 * Whether a result has a name of its own. An unassigned result is named
 * after its operation (`softmax`, `chunk[1]`), so a label that already shows
 * the operation need not repeat it.
 */
export function ownName(name: string, kind: string): boolean {
  return name !== kind && !name.startsWith(`${kind}[`);
}

/** A step as people read it: `rsqrt → scale`, or `rsqrt` when unnamed. */
export function stepLabel(kind: string, name?: string): string {
  return name && ownName(name, kind)
    ? `${kindName(kind)} → ${name}`
    : kindName(kind);
}
