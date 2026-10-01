export type TokenKind =
  | "keyword"
  | "namespace"
  | "tensor"
  | "call"
  | "string"
  | "comment"
  | "axes"
  | "number"
  | "operator"
  | "plain";
export type Token = { text: string; kind: TokenKind };

const KEYWORDS = new Set(
  "def class return import from as for in if else elif while with try except finally raise pass break continue lambda yield not and or is None True False del global nonlocal assert async await".split(
    " ",
  ),
);
const NAMESPACES = new Set(["torch", "nn", "F", "math", "self", "np"]);
const PATTERN =
  /(#.*$)|("(?:[^"\\]|\\.)*"?|'(?:[^'\\]|\\.)*'?)|(\b\d+(?:\.\d*)?(?:e[+-]?\d+)?\b|\.\d+)|([A-Za-z_]\w*)|([@+\-*/%=<>!&|^~]+)|(\s+|.)/g;

/**
 * Color one line of Python. `tensors` are names known to hold tensors in the
 * displayed run, so variables the recorder saw stand out from other names.
 */
export function highlightLine(
  line: string,
  tensors: ReadonlySet<string> = new Set(),
): Token[] {
  const tokens: Token[] = [];
  const push = (text: string, kind: TokenKind) => {
    const last = tokens.at(-1);
    if (last && last.kind === kind && kind === "plain") last.text += text;
    else tokens.push({ text, kind });
  };
  for (const match of line.matchAll(PATTERN)) {
    const [text, comment, string, number, word, operator] = match;
    if (comment !== undefined) {
      // An axis annotation is an instruction to the recorder, not just a note.
      // Axis names and shape contracts are read by TensorViewer.
      const axes = /^#\s*(axes|shape):/.test(comment);
      push(comment, axes ? "axes" : "comment");
    } else if (string !== undefined) push(text, "string");
    else if (number !== undefined) push(text, "number");
    else if (word !== undefined) {
      const rest = line.slice(match.index + text.length);
      const member = line[match.index - 1] === ".";
      push(
        text,
        KEYWORDS.has(word) && !member
          ? "keyword"
          : NAMESPACES.has(word) && !member
            ? "namespace"
            : tensors.has(word) && !member
              ? "tensor"
              : /^\s*\(/.test(rest)
                ? "call"
                : "plain",
      );
    } else if (operator !== undefined) push(text, "operator");
    else push(text, "plain");
  }
  return tokens;
}

/** Columns a line occupies when tabs advance to the next multiple of four. */
export function visualWidth(line: string): number {
  let width = 0;
  for (const character of line)
    width = character === "\t" ? width + 4 - (width % 4) : width + 1;
  return width;
}

/** 1-based line and column of a caret offset. */
export function caretPosition(text: string, offset: number) {
  const before = text.slice(0, offset);
  const line = before.split("\n").length;
  return { line, column: offset - before.lastIndexOf("\n") };
}
