import { expect, test } from "vitest";
import { elementBytes, formatBytes, storageBytes, tensorBytes } from "./memory";

test("element sizes follow the dtype, with or without the torch prefix", () => {
  expect(elementBytes("float32")).toBe(4);
  expect(elementBytes("torch.bfloat16")).toBe(2);
  expect(elementBytes("bool")).toBe(1);
  expect(elementBytes("quint4x2")).toBeNull();
  expect(tensorBytes({ dtype: "int64", numel: 7 })).toBe(56);
  expect(tensorBytes({ dtype: "mystery", numel: 7 })).toBeNull();
});

test("views of one storage are counted once, at their largest", () => {
  expect(
    storageBytes([
      { dtype: "float32", numel: 16, storage_id: "a" },
      { dtype: "float32", numel: 4, storage_id: "a" },
      { dtype: "float64", numel: 2, storage_id: "b" },
      { dtype: "mystery", numel: 9, storage_id: "c" },
    ]),
  ).toBe(64 + 16);
});

test("byte counts read in the nearest unit", () => {
  expect(formatBytes(56)).toBe("56 B");
  expect(formatBytes(1024)).toBe("1.0 KB");
  expect(formatBytes(1536)).toBe("1.5 KB");
  expect(formatBytes(300 * 1024 * 1024)).toBe("300 MB");
  expect(formatBytes(3 * 1024 ** 3)).toBe("3.0 GB");
});
