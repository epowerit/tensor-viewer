import type {
  ComponentSpec,
  CustomComponent,
  ToolboxItem,
} from "../api/client";

export function customTool(custom: CustomComponent): ToolboxItem {
  return {
    kind: "custom",
    title: custom.name,
    description: custom.description,
    group: "Custom",
    parameters: [],
    custom,
  };
}

export function componentTool(
  spec: ComponentSpec | undefined,
  catalog: ToolboxItem[],
) {
  return spec?.custom
    ? customTool(spec.custom)
    : catalog.find((item) => item.kind === spec?.kind);
}

export function parseArguments(text: string): Record<string, unknown> {
  const value = JSON.parse(text);
  if (value === null || Array.isArray(value) || typeof value !== "object")
    throw new Error('Use a JSON object, such as {"scale": 2}.');
  if (text.length > 8000)
    throw new Error("Keep arguments under 8,000 characters.");
  JSON.stringify(value, (_key, entry) => {
    if (typeof entry === "number" && !Number.isFinite(entry))
      throw new Error("Use finite numbers in constructor arguments.");
    return entry;
  });
  return value;
}
