import { expect, test } from "vitest";
import { planeThumbnail, sliceThumbnails } from "./sliceThumbnails";

test("each slice of a hidden axis becomes a small averaged picture", () => {
  const images = {
    shape: [3, 4, 4],
    numel: 48,
    dtype: "float32",
    values: Array.from({ length: 48 }, (_, i) => i),
  };
  const film = sliceThumbnails(images, { row: 1, column: 2 }, [0, 0, 0], 0, 2)!;
  expect([film.rows, film.columns, film.total, film.thumbs.length]).toEqual([
    2, 2, 3, 3,
  ]);
  // Slice 1 holds 16..31; its top-left 2×2 block is 16, 17, 20, 21, as the
  // backend's numpy.array_split blocks are.
  expect(film.thumbs[1][0]).toBe((16 + 17 + 20 + 21) / 4);
  // Small planes keep every cell.
  expect(
    sliceThumbnails(images, { row: 1, column: 2 }, [0, 0, 0], 0)!.thumbs[0],
  ).toHaveLength(16);
  expect(
    sliceThumbnails(
      { ...images, values: [] },
      { row: 1, column: 2 },
      [0, 0, 0],
      0,
    ),
  ).toBeNull();
});

test("uneven planes split like numpy.array_split, larger blocks first", () => {
  const strip = {
    shape: [2, 1, 5],
    numel: 10,
    dtype: "float32",
    values: Array.from({ length: 10 }, (_, i) => i),
  };
  const film = sliceThumbnails(strip, { row: 1, column: 2 }, [0, 0, 0], 0, 2)!;
  // Columns 0..4 split into [0, 1, 2] and [3, 4].
  expect(film.thumbs[0]).toEqual([1, 3.5]);
});

test("a card's thumbnail averages the first plane down", () => {
  const tensor = {
    shape: [2, 4, 4],
    dtype: "float32",
    numel: 32,
    values: Array.from({ length: 32 }, (_, i) => i),
  };
  // The first 4 × 4 plane in 2 × 2 blocks; the second plane is not pictured.
  expect(planeThumbnail(tensor, 2)).toEqual({
    rows: 2,
    columns: 2,
    picture: [2.5, 4.5, 10.5, 12.5],
  });
});

test("a vector is one row, and a block with NaN or infinity is NaN", () => {
  const vector = {
    shape: [4],
    dtype: "float32",
    numel: 4,
    values: [1, "nan", 3, "inf"],
  };
  const thumb = planeThumbnail(vector, 4)!;
  expect(thumb.rows).toBe(1);
  expect(thumb.picture[0]).toBe(1);
  expect(thumb.picture[1]).toBeNaN();
  expect(thumb.picture[3]).toBeNaN();
});

test("a paged tensor has no thumbnail", () => {
  expect(
    planeThumbnail({
      shape: [50, 50],
      dtype: "float32",
      numel: 2500,
      values: [],
    }),
  ).toBeNull();
});
