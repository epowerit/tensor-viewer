/** Solve x + pad = output * stride + kernel * dilation, without scanning either axis. */
export function nearestWindowSample(
  x: number,
  pad: number,
  stride: number,
  dilation: number,
  kernel: number,
  outputs: number,
  previous: number,
) {
  let a = BigInt(stride),
    b = BigInt(dilation),
    coefficient = 1n,
    next = 0n;
  while (b) {
    const q = a / b;
    [a, b] = [b, a - q * b];
    [coefficient, next] = [next, coefficient - q * next];
  }
  const target = x + pad,
    gcd = Number(a);
  if (target % gcd) return null;
  const period = dilation / gcd,
    modulus = BigInt(period);
  const residue = Number(
    (((BigInt(target / gcd) * coefficient) % modulus) + modulus) % modulus,
  );
  const lo = Math.max(
    0,
    Math.ceil((target - (kernel - 1) * dilation) / stride),
  );
  const hi = Math.min(outputs - 1, Math.floor(target / stride));
  const first = Math.ceil((lo - residue) / period),
    last = Math.floor((hi - residue) / period);
  if (first > last) return null;
  const step = Math.max(
    first,
    Math.min(last, Math.round((previous - residue) / period)),
  );
  const output = residue + step * period;
  return { output, kernel: (target - output * stride) / dilation };
}
