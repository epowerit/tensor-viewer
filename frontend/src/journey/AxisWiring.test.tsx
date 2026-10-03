import { renderToStaticMarkup } from "react-dom/server";
import { expect, test } from "vitest";
import type { Tensor } from "../api/client";
import { AxisWiring } from "./AxisWiring";

test("wiring draws a line for every part that travels, named both ends", () => {
  const input = {
    id: "q",
    name: "q",
    shape: [2, 5, 8],
    axes: ["batch", "tokens", "features"],
  } as unknown as Tensor;
  const html = renderToStaticMarkup(
    <AxisWiring
      data={{
        input,
        fromNames: ["batch", "tokens", "features"],
        to: [2, 2, 5, 4],
        toNames: ["batch", "heads", "tokens", "head_features"],
        map: [
          [{ axis: 0, size: 2 }],
          [{ axis: 2, size: 2, piece: { index: 0, of: 2 } }],
          [{ axis: 1, size: 5 }],
          [{ axis: 2, size: 4, piece: { index: 1, of: 2 } }],
        ],
      }}
    />,
  );
  expect(html).toContain(
    'aria-label="Axes [batch 2, tokens 5, features 8] become [batch 2, heads 2, tokens 5, head_features 4]"',
  );
  // Four lines: two whole axes, and the features axis fanning out in two.
  expect(html.match(/class="axis-wiring-line/g)).toHaveLength(4);
  expect(html.match(/axis-wiring-line is-piece/g)).toHaveLength(2);
});
