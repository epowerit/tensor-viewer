import { useLayoutEffect, useRef, useState } from "react";
import type { ComponentSpec } from "../api/client";
import { componentSources } from "./connections";
export function BranchConnections({
  components,
}: {
  components: ComponentSpec[];
}) {
  const svg = useRef<SVGSVGElement>(null);
  const [layout, setLayout] = useState<{
    width: number;
    height: number;
    paths: { key: string; path: string; label: string }[];
  }>({ width: 1, height: 1, paths: [] });
  useLayoutEffect(() => {
    const parent = svg.current?.parentElement;
    if (!parent) return;
    const measure = () => {
      const bounds = parent.getBoundingClientRect(),
        scale = bounds.width / parent.offsetWidth || 1;
      const nodes = new Map(
        Array.from(
          parent.querySelectorAll<HTMLElement>("[data-component-id]"),
        ).map((el) => [el.dataset.componentId!, el.getBoundingClientRect()]),
      );
      const paths = components.flatMap((c, i) =>
        componentSources(components, i).flatMap((source, slot) => {
          const a = nodes.get(source),
            b = nodes.get(c.id);
          if (!a || !b) return [];
          const ax = (a.left + a.width / 2 - bounds.left) / scale,
            ay = (a.top - bounds.top) / scale,
            bx = (b.left + b.width / 2 - bounds.left) / scale,
            by = (b.top - bounds.top) / scale;
          const lane = Math.min(ay, by) - 24 - ((i + slot) % 6) * 18;
          return [
            {
              key: `${c.id}-${slot}`,
              label: `${source === "input" ? "Input" : `Step ${components.findIndex((n) => n.id === source) + 1}`} → step ${i + 1}, input ${slot + 1}`,
              path: `M ${ax} ${ay} V ${lane + 8} Q ${ax} ${lane} ${ax + 8} ${lane} H ${bx - 8} Q ${bx} ${lane} ${bx} ${lane + 8} V ${by - 5}`,
            },
          ];
        }),
      );
      const next = {
        width: parent.offsetWidth,
        height: parent.offsetHeight,
        paths,
      };
      setLayout((old) =>
        JSON.stringify(old) === JSON.stringify(next) ? old : next,
      );
    };
    const observer = new ResizeObserver(measure);
    observer.observe(parent);
    parent
      .querySelectorAll("[data-component-id]")
      .forEach((el) => observer.observe(el));
    measure();
    return () => observer.disconnect();
  }, [JSON.stringify(components)]);
  return (
    <svg
      className="builder-branch-connections"
      ref={svg}
      width={layout.width}
      height={layout.height}
      aria-label="Component data connections"
      role="img"
    >
      <defs>
        <marker
          id="branch-arrow"
          viewBox="0 0 8 8"
          refX="6"
          refY="4"
          markerWidth="7"
          markerHeight="7"
          orient="auto"
        >
          <path d="M0 0L8 4L0 8" fill="currentColor" />
        </marker>
      </defs>
      {layout.paths.map((p) => (
        <path key={p.key} d={p.path} markerEnd="url(#branch-arrow)">
          <title>{p.label}</title>
        </path>
      ))}
    </svg>
  );
}
