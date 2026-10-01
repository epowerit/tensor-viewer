import { useMemo } from "react";
import type { Tensor } from "../api/client";
import { pixelColors, type PixelPlan } from "../inputs/samples";
import { formatValue } from "./coordinates";
import { useTensorValues } from "./useTensorValues";

type Props = {
  tensor: Tensor;
  plan: PixelPlan;
  runId?: string;
  selected: number;
  onSelect: (index: number) => void;
};

/** The selected height × width plane drawn as a picture of its recorded values. */
export function PixelView({ tensor, plan, runId, selected, onSelect }: Props) {
  const indices = useMemo(() => plan.pixels.flat(), [plan]);
  const data = useTensorValues(tensor, runId, indices);
  const picture = pixelColors(plan, data.valueAt);
  const size = Math.max(4, Math.floor(168 / Math.max(plan.height, plan.width)));
  const channelAxis = tensor.shape.length - 3;
  return (
    <figure className="pixel-view">
      <svg
        viewBox={`0 0 ${plan.width * size} ${plan.height * size}`}
        width={plan.width * size}
        height={plan.height * size}
        role="img"
        aria-label={`${tensor.name} drawn as a ${plan.height} by ${plan.width} ${plan.color ? "color" : "grayscale"} picture`}
        shapeRendering="crispEdges"
      >
        {plan.pixels.map((pixel, p) => (
          <rect
            key={p}
            x={(p % plan.width) * size}
            y={Math.floor(p / plan.width) * size}
            width={size}
            height={size}
            fill={picture?.colors[p] ?? "transparent"}
            className={pixel.includes(selected) ? "selected" : ""}
            onClick={() => onSelect(pixel[0])}
          />
        ))}
      </svg>
      <figcaption>
        {data.loading
          ? "Loading recorded values…"
          : data.error
            ? data.error
            : picture
              ? `${plan.color ? "Channels 0, 1, 2 as red, green, blue" : channelAxis >= 0 ? `${tensor.axes[channelAxis] ?? "axis"} ${plan.prefix.at(-1)} in gray` : "Gray"} · darkest ${formatValue(picture.minimum)}, brightest ${formatValue(picture.maximum)}${plan.prefix.length > (plan.color ? 0 : 1) ? ` · at [${plan.prefix.join(", ")}]` : ""}`
              : "No finite values to draw."}
      </figcaption>
    </figure>
  );
}
