/** One parameter of a module's `__init__`, as the code declares it. */
export type Parameter = {
  name: string;
  /** The default as written in Python, or null when the argument is required. */
  python: string | null;
  /** The default as JSON, when it has one (numbers, booleans, None, strings). */
  json?: unknown;
};

export type Signature = {
  parameters: Parameter[];
  /** It takes `**kwargs`, so any name is accepted. */
  open: boolean;
};

const escape = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const indentOf = (line: string) => line.length - line.trimStart().length;

/** The text inside the parentheses that open at `start`, strings respected. */
function enclosed(code: string, start: number) {
  let depth = 0;
  let quote = "";
  for (let i = start; i < code.length; i++) {
    const char = code[i];
    if (quote) {
      if (char === "\\") i++;
      else if (char === quote) quote = "";
    } else if (char === '"' || char === "'") quote = char;
    else if (char === "#") {
      const end = code.indexOf("\n", i);
      i = end < 0 ? code.length : end;
    } else if ("([{".includes(char)) depth++;
    else if (")]}".includes(char) && --depth === 0)
      return code.slice(start + 1, i);
  }
  return null;
}

/** Splits at commas outside brackets and strings. */
function topLevel(text: string, separator: string) {
  const parts: string[] = [];
  let depth = 0,
    quote = "",
    from = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quote) {
      if (char === "\\") i++;
      else if (char === quote) quote = "";
    } else if (char === '"' || char === "'") quote = char;
    else if ("([{".includes(char)) depth++;
    else if (")]}".includes(char)) depth--;
    else if (char === separator && depth === 0) {
      parts.push(text.slice(from, i));
      from = i + 1;
    }
  }
  parts.push(text.slice(from));
  return parts;
}

function asJson(python: string): { json?: unknown } {
  const value = python.trim();
  if (/^[-+]?(\d[\d_]*\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(value))
    return { json: Number(value.replace(/_/g, "")) };
  if (value === "True") return { json: true };
  if (value === "False") return { json: false };
  if (value === "None") return { json: null };
  const quoted = value.match(/^(['"])((?:(?!\1)[^\\]|\\.)*)\1$/);
  if (quoted) return { json: quoted[2].replace(/\\(.)/g, "$1") };
  return {};
}

/**
 * The `__init__` parameters of `className`, read from whichever file defines
 * it, or null when no file does. A class without its own `__init__` takes
 * nothing it names.
 */
export function constructorSignature(
  files: Record<string, string>,
  className: string,
): Signature | null {
  if (!className.trim()) return null;
  const header = new RegExp(`^([ \\t]*)class\\s+${escape(className)}\\b`, "m");
  for (const code of Object.values(files)) {
    const found = header.exec(code);
    if (!found) continue;
    const lines = code.slice(found.index).split("\n");
    const classIndent = found[1].length;
    let offset = found.index + lines[0].length + 1;
    for (const line of lines.slice(1)) {
      const body = line.trim();
      if (body && !body.startsWith("#") && indentOf(line) <= classIndent) break;
      const init = line.match(/^\s*def\s+__init__\s*\(/);
      if (init) {
        const inside = enclosed(code, offset + init[0].length - 1);
        if (inside === null) break;
        return parse(inside);
      }
      offset += line.length + 1;
    }
    return { parameters: [], open: false };
  }
  return null;
}

function parse(inside: string): Signature {
  const parameters: Parameter[] = [];
  let open = false;
  topLevel(inside, ",").forEach((raw, index) => {
    const part = raw.replace(/#[^\n]*/g, "").trim();
    if (!part || part === "/" || part === "*") return;
    if (part.startsWith("**")) {
      open = true;
      return;
    }
    if (part.startsWith("*")) return;
    const [declared, ...rest] = topLevel(part, "=");
    const name = declared.split(":")[0].trim();
    if (index === 0 && name === "self") return;
    if (!/^\w+$/.test(name)) return;
    const python = rest.length ? rest.join("=").trim() : null;
    parameters.push({ name, python, ...(python ? asJson(python) : {}) });
  });
  return { parameters, open };
}

/**
 * The constructor JSON with `name` added, and where its value sits in the
 * text, so the field can select it for typing.
 */
export function withArgument(
  current: Record<string, unknown>,
  parameter: Parameter,
): { text: string; start: number; end: number } {
  const value = "json" in parameter ? parameter.json : null;
  const text = JSON.stringify({ ...current, [parameter.name]: value }, null, 2);
  const key = `${JSON.stringify(parameter.name)}: `;
  const start = text.indexOf(key) + key.length;
  return { text, start, end: start + JSON.stringify(value).length };
}

/** The parameter a mistyped name most likely meant, within two edits. */
export function meant(name: string, parameters: Parameter[]) {
  let best: { name: string; distance: number } | null = null;
  for (const parameter of parameters) {
    const distance = edits(name.toLowerCase(), parameter.name.toLowerCase());
    if (distance <= 2 && (!best || distance < best.distance))
      best = { name: parameter.name, distance };
  }
  return best?.name ?? null;
}

function edits(a: string, b: string) {
  let row = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const next = [i];
    for (let j = 1; j <= b.length; j++)
      next[j] = Math.min(
        row[j] + 1,
        next[j - 1] + 1,
        row[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1),
      );
    row = next;
  }
  return row[b.length];
}
