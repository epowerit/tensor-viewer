/** A .npy file's shape and values, as the backend writes them. */
export type Npy = { shape: number[]; values: Float64Array };

const READERS: Record<
  string,
  [number, (view: DataView, at: number) => number]
> = {
  "<f8": [8, (view, at) => view.getFloat64(at, true)],
  "<f4": [4, (view, at) => view.getFloat32(at, true)],
  "<i8": [8, (view, at) => Number(view.getBigInt64(at, true))],
  "<i4": [4, (view, at) => view.getInt32(at, true)],
  "<i2": [2, (view, at) => view.getInt16(at, true)],
  "|i1": [1, (view, at) => view.getInt8(at)],
  "|u1": [1, (view, at) => view.getUint8(at)],
  "|b1": [1, (view, at) => view.getUint8(at)],
};

/**
 * Reads a version 1–3 .npy file of little-endian numbers in C order: the
 * kinds `numpy.save` writes for TensorViewer's dtypes. Anything else is null.
 */
export function readNpy(buffer: ArrayBuffer): Npy | null {
  const bytes = new Uint8Array(buffer);
  if (bytes.length < 10 || bytes[0] !== 0x93) return null;
  const view = new DataView(buffer);
  const major = bytes[6];
  const headerLength =
    major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const start = major === 1 ? 10 : 12;
  const header = new TextDecoder().decode(
    bytes.subarray(start, start + headerLength),
  );
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*True/.test(header);
  const shapeText = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1];
  const reader = descr ? READERS[descr] : undefined;
  if (!reader || fortran || shapeText === undefined) return null;
  const shape = shapeText
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map(Number);
  const count = shape.reduce((a, b) => a * b, 1);
  const [size, read] = reader;
  const offset = start + headerLength;
  if (offset + count * size > bytes.length) return null;
  const values = new Float64Array(count);
  for (let i = 0; i < count; i++) values[i] = read(view, offset + i * size);
  return { shape, values };
}
