import { renderToStaticMarkup } from "react-dom/server";
import { expect, test } from "vitest";
import type { Tensor } from "../api/client";
import { ContractContext } from "./ContractContext";
import { TensorPeek } from "./TensorPeek";

const tensor = {
  id: "t3",
  name: "active",
  shape: [2, 2],
  axes: ["rows", "columns"],
  dtype: "float32",
  numel: 4,
  values: [0, 1.5, 0, 2],
  value_source: "inline",
  minimum: 0,
  maximum: 2,
  storage_id: "storage-1",
  contiguous: true,
  histogram: {
    low: 0,
    high: 2,
    counts: [2, 0, 2],
    zeros: 2,
    non_finite: 0,
    mean: 0.875,
    std: 0.9,
  },
} as unknown as Tensor;

test("a peek shows the value spread and the line's contract", () => {
  const check = {
    line: 8,
    text: "zeros: ..30%",
    ok: false,
    message: "active is 50% zeros, above 30%.",
  };
  const html = renderToStaticMarkup(
    <ContractContext
      value={{ byTensor: new Map([["t3", check]]), byOperation: new Map() }}
    >
      <TensorPeek tensor={tensor} />
    </ContractContext>,
  );
  expect(html).toContain("value-spread");
  expect(html).toContain("50% zeros");
  expect(html).toContain("peek-contract-broken");
  expect(html).toContain("active is 50% zeros, above 30%.");
});

test("a shape-only peek has no spread and no contract line", () => {
  const html = renderToStaticMarkup(
    <TensorPeek
      tensor={{ ...tensor, value_source: "shape", values: [], histogram: null }}
    />,
  );
  expect(html).not.toContain("value-spread");
  expect(html).not.toContain("peek-contract");
});
