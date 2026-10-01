import type { Tensor } from "../api/client";
import { alignShapes } from "./relations";

/**
 * Operand shapes written right-aligned above the result, marking every axis
 * where one value is reused to fill a larger size.
 */
export function BroadcastAlignment({
  operands,
  output,
}: {
  operands: Tensor[];
  output: Tensor;
}) {
  const rows = alignShapes(
    operands.map((tensor) => tensor.shape),
    output.shape,
  );
  if (!rows) return null;
  return (
    <figure className="broadcast-alignment">
      <figcaption>
        Shapes line up from the last axis. A size of 1, or a missing axis, is
        reused to match the other operand.
      </figcaption>
      <table>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              <th scope="row">{operands[i].name}</th>
              {row.map((axis, j) => (
                <td
                  key={j}
                  className={
                    axis.size === null
                      ? "absent"
                      : axis.stretched
                        ? "stretched"
                        : ""
                  }
                  title={
                    axis.stretched
                      ? `Reused ${output.shape[j]} times along this axis`
                      : undefined
                  }
                >
                  {axis.size ?? "·"}
                  {axis.stretched && <small>→{output.shape[j]}</small>}
                </td>
              ))}
            </tr>
          ))}
          <tr className="result">
            <th scope="row">{output.name}</th>
            {output.shape.map((size, j) => (
              <td key={j}>{size}</td>
            ))}
          </tr>
        </tbody>
      </table>
    </figure>
  );
}
