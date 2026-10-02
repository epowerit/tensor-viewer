import { createContext, Fragment, useContext, type CSSProperties } from "react";
import type { Tensor } from "../api/client";
import type { Ink } from "./axisInk";
import type { CellPaint } from "./cellPaint";

/** Axis ink for the tensors on screen; empty outside a recorded run. */
export const AxisInkContext = createContext<
  ((tensor: Tensor) => (Ink | null)[] | null) | null
>(null);

/** Per-cell paint for the tensors on screen; empty outside a recorded run. */
export const CellPaintContext = createContext<CellPaint | null>(null);

export function useCellPaint(): CellPaint | null {
  return useContext(CellPaintContext);
}

export function useAxisInk(tensor: Tensor | null | undefined) {
  const ink = useContext(AxisInkContext);
  return tensor && ink ? ink(tensor) : null;
}

/** CSS variables that color one axis: text, and an underline bar. */
export function inkStyle(
  ink: Ink | null | undefined,
): CSSProperties | undefined {
  if (!ink) return undefined;
  return {
    "--ink": ink.colors[0],
    "--ink-bar":
      ink.colors.length > 1
        ? `linear-gradient(90deg, ${ink.colors.join(", ")})`
        : ink.colors[0],
  } as CSSProperties;
}

/** One axis size, inked by where the axis came from. */
export function InkSize({ size, ink }: { size: number; ink?: Ink | null }) {
  if (!ink) return <>{size.toLocaleString()}</>;
  return (
    <span
      className={`ink-size${ink.piece ? " ink-piece" : ""}${ink.colors.length > 1 ? " ink-merged" : ""}`}
      style={inkStyle(ink)}
      title={ink.text}
    >
      {size.toLocaleString()}
    </span>
  );
}

/** A shape such as [2, 4, 8], each size in the ink of its axis. */
export function InkShape({
  shape,
  ink,
  separator = ", ",
  brackets = true,
}: {
  shape: number[];
  ink?: (Ink | null)[] | null;
  separator?: string;
  brackets?: boolean;
}) {
  if (!shape.length) return <>scalar</>;
  const sizes = shape.map((size, axis) => (
    <Fragment key={axis}>
      {axis > 0 && separator}
      <InkSize size={size} ink={ink?.[axis]} />
    </Fragment>
  ));
  return <span className="ink-shape">{brackets ? <>[{sizes}]</> : sizes}</span>;
}

/** A recorded tensor's shape in axis ink, wherever it is shown. */
export function TensorShape({
  tensor,
  separator,
}: {
  tensor: Tensor;
  separator?: string;
}) {
  return (
    <InkShape
      shape={tensor.shape}
      ink={useAxisInk(tensor)}
      separator={separator}
    />
  );
}
