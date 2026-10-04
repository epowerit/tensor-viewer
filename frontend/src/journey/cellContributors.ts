import type { Operation, Run, Tensor } from "../api/client";
import {
  locatePart,
  locateWhole,
  tensorAssembly,
} from "../operations/assembly";
import {
  convolutionSelection,
  tensorConvolution,
} from "../operations/convolution";
import {
  layerNormalization,
  layerNormSelection,
} from "../operations/layerNormalization";
import { layoutTransition } from "../operations/layoutTransition";
import { linearProjection, linearSelection } from "../operations/linear";
import {
  poolingRegion,
  poolingSample,
  tensorPooling,
} from "../operations/pooling";
import {
  broadcastIndex,
  einsumTerms,
  reductionGroup,
  relationSource,
  tensorRelation,
  triangleKeeps,
} from "../operations/relations";
import {
  dotContributors,
  normalizationGroup,
  product,
  ravel,
  unravel,
} from "../tensors/coordinates";
import { sourceIndex } from "../tensors/relationships";
import { producedTensorIds } from "../tensors/provenance";

export const CELL_CONTRIBUTOR_LIMIT = 64;
export type CellContributor = { tensorId: string; index: number; role: string };
export type CellContributors = {
  status: "mapped" | "unsupported" | "invalid";
  sources: CellContributor[];
  /** Operand uses, not distinct cells: x + x reads the same cell twice. */
  total: number;
  truncated: boolean;
  summary: string;
};
const empty = (
  status: "unsupported" | "invalid",
  summary: string,
): CellContributors => ({
  status,
  sources: [],
  total: 0,
  truncated: false,
  summary,
});
const unsupported = (
  summary = "An exact cell relationship is not recorded for this operation.",
) => empty("unsupported", summary);
const same = (a: number[], b: number[]) =>
  a.length === b.length && a.every((n, i) => n === b[i]);
const integer = (n: unknown, minimum = 0): n is number =>
  typeof n === "number" && Number.isSafeInteger(n) && n >= minimum;
const validTensor = (t: Tensor | undefined): t is Tensor =>
  !!t &&
  t.shape.length <= 64 &&
  t.shape.every((n) => integer(n)) &&
  integer(t.numel) &&
  product(t.shape) === t.numel;
const source = (
  tensor: Tensor,
  index: number,
  role: string,
): CellContributor => ({ tensorId: tensor.id, index, role });

/**
 * Trace one output cell through one recorded operation. Work and allocations
 * depend on rank/operand metadata and at most 64 source uses, never tensor size.
 * This is a structural dependency map, not a derivative or a recomputation.
 */
export function traceCellContributors(
  run: Run,
  op: Operation,
  tensorId: string,
  index: number,
): CellContributors {
  const output = run.trace.tensors[tensorId];
  if (
    !producedTensorIds(op).includes(tensorId) ||
    !validTensor(output) ||
    !integer(index) ||
    index >= output.numel
  )
    return empty(
      "invalid",
      "Select a valid cell in an output of this operation.",
    );
  if (op.status !== "ok")
    return unsupported("This operation did not complete.");
  if (op.mutations?.length || op.arguments.out != null)
    return unsupported(
      "A recorded write needs a mutation-specific relationship; no cell origins are inferred.",
    );
  const inputs = op.inputs.map((id) => run.trace.tensors[id]);
  if (
    !inputs.every(validTensor) ||
    !op.outputs.every((id) => validTensor(run.trace.tensors[id]))
  )
    return empty(
      "invalid",
      "The recorded tensor metadata is incomplete or inconsistent.",
    );
  const mapped = (
    sources: CellContributor[],
    total: number,
    summary: string,
  ): CellContributors => {
    if (
      !integer(total) ||
      sources.length > total ||
      sources.some(
        (s) =>
          !integer(s.index) ||
          !run.trace.tensors[s.tensorId] ||
          s.index >= run.trace.tensors[s.tensorId].numel,
      )
    )
      return unsupported(
        "The recorded relationship cannot be represented with exact cell indices.",
      );
    const visible = sources.slice(0, CELL_CONTRIBUTOR_LIMIT);
    return {
      status: "mapped",
      sources: visible,
      total,
      truncated: total > visible.length,
      summary:
        total > visible.length
          ? `${summary} Tracking the first ${visible.length} of ${total.toLocaleString()} source uses; sampled cells are highlighted.`
          : summary,
    };
  };
  const assembly = tensorAssembly(run, op);
  if (assembly) {
    if (assembly.joining) {
      const part = locatePart(assembly, index);
      return mapped(
        [
          source(
            assembly.parts[part.part],
            part.index,
            `operand ${part.part + 1}`,
          ),
        ],
        1,
        "This cell comes from its exact position in one joined operand.",
      );
    }
    const part = assembly.parts.findIndex((t) => t.id === tensorId);
    return mapped(
      [
        source(
          assembly.whole,
          locateWhole(assembly, part, index).index,
          "input",
        ),
      ],
      1,
      "This cell retains its position within the original tensor.",
    );
  }
  const conv = tensorConvolution(run, op);
  if (conv) {
    const target = convolutionSelection(conv, index, 0);
    const validCounts = conv.kernel.map((size, axis) => {
      const start =
        target.coordinates[conv.channelAxis + 1 + axis] * conv.stride[axis] -
        conv.before[axis];
      return Math.max(
        0,
        Math.min(
          size - 1,
          Math.floor(
            (conv.input.shape[conv.channelAxis + 1 + axis] - 1 - start) /
              conv.dilation[axis],
          ),
        ) -
          Math.max(0, Math.ceil(-start / conv.dilation[axis])) +
          1,
      );
    });
    const sources: CellContributor[] = [];
    for (
      let term = 0;
      term < conv.terms && sources.length < CELL_CONTRIBUTOR_LIMIT;
      term++
    ) {
      const pair = convolutionSelection(conv, index, term);
      if (pair.input !== null)
        sources.push(source(conv.input, pair.input, "input"));
      sources.push(source(conv.weight, pair.weight, "kernel weight"));
    }
    if (conv.bias && sources.length < CELL_CONTRIBUTOR_LIMIT)
      sources.push(source(conv.bias, target.feature, "bias"));
    return mapped(
      sources,
      conv.terms + product(validCounts) * conv.channels + Number(!!conv.bias),
      "The receptive field is paired with this output channel’s kernel; padding is a constant, with no input cell.",
    );
  }
  const pool = tensorPooling(run, op);
  if (pool) {
    const region = poolingRegion(pool, index);
    const starts = region.start.map((start, axis) =>
      Math.max(0, Math.ceil(-start / pool.dilation[axis])),
    );
    const counts = starts.map((start, axis) =>
      Math.max(
        0,
        Math.min(
          region.shape[axis] - 1,
          Math.floor(
            (pool.input.shape.at(-2 + axis)! - 1 - region.start[axis]) /
              pool.dilation[axis],
          ),
        ) -
          start +
          1,
      ),
    );
    const sources = Array.from(
      { length: Math.min(region.real, CELL_CONTRIBUTOR_LIMIT) },
      (_, i) => {
        const local = unravel(i, counts).map((n, axis) => n + starts[axis]);
        return source(
          pool.input,
          poolingSample(pool, region, ravel(local, region.shape)).input!,
          pool.mode === "max" ? "compared input" : "averaged input",
        );
      },
    );
    return mapped(
      sources,
      region.real,
      pool.mode === "max"
        ? "These input cells are compared to select the maximum and its position; a winner is not inferred from missing values."
        : "These real input cells form the averaging window; padding contributes no input cell.",
    );
  }
  // Remaining rules describe one output, so do not apply one to a tuple member.
  if (op.outputs.length !== 1) return unsupported();
  const linear = linearProjection(run, op);
  if (linear) {
    const sources: CellContributor[] = [];
    for (
      let term = 0;
      term < Math.min(linear.features, CELL_CONTRIBUTOR_LIMIT / 2);
      term++
    ) {
      const pair = linearSelection(linear, index, term);
      sources.push(
        source(linear.input, pair.input, "input"),
        source(linear.weight, pair.weight, "weight"),
      );
    }
    if (linear.bias && sources.length < CELL_CONTRIBUTOR_LIMIT)
      sources.push(source(linear.bias, index % linear.outputs, "bias"));
    return mapped(
      sources,
      linear.features * 2 + Number(!!linear.bias),
      "This feature is a dot product of the input vector and one weight row, plus its bias when present.",
    );
  }
  if (["matmul", "mm", "bmm"].includes(op.kind) && inputs.length === 2) {
    const [a, b] = inputs;
    if (
      !a.shape.length ||
      !b.shape.length ||
      (op.kind === "mm" && (a.shape.length !== 2 || b.shape.length !== 2)) ||
      (op.kind === "bmm" &&
        (a.shape.length !== 3 ||
          b.shape.length !== 3 ||
          a.shape[0] !== b.shape[0]))
    )
      return unsupported();
    const left = a.shape.length === 1 ? [1, ...a.shape] : a.shape;
    const right = b.shape.length === 1 ? [...b.shape, 1] : b.shape;
    const batchRank = Math.max(left.length, right.length) - 2;
    const batch: number[] = [];
    for (let axis = 0; axis < batchRank; axis++) {
      const m = left[axis - batchRank + left.length - 2] ?? 1;
      const n = right[axis - batchRank + right.length - 2] ?? 1;
      if (m !== n && m !== 1 && n !== 1) return unsupported();
      batch.push(m === 1 ? n : m);
    }
    const promoted = [...batch, left.at(-2)!, right.at(-1)!];
    const expected = [
      ...batch,
      ...(a.shape.length === 1 ? [] : [left.at(-2)!]),
      ...(b.shape.length === 1 ? [] : [right.at(-1)!]),
    ];
    if (left.at(-1) !== right.at(-2) || !same(expected, output.shape))
      return unsupported();
    const pairs = dotContributors(
      left,
      right,
      promoted,
      index,
      CELL_CONTRIBUTOR_LIMIT / 2,
    );
    return mapped(
      pairs.flatMap((pair) => [
        source(a, pair.left, "left input"),
        source(b, pair.right, "right input"),
      ]),
      left.at(-1)! * 2,
      "Matching entries in this row and column are multiplied and summed; batch axes follow broadcasting.",
    );
  }
  const norm = layerNormalization(run, op);
  if (norm) {
    const selected = layerNormSelection(norm, index);
    const sources = Array.from(
      { length: Math.min(norm.size, CELL_CONTRIBUTOR_LIMIT) },
      (_, i) => source(norm.input, selected.start + i, "normalization group"),
    );
    if (norm.weight && sources.length < CELL_CONTRIBUTOR_LIMIT)
      sources.push(source(norm.weight, selected.feature, "scale"));
    if (norm.bias && sources.length < CELL_CONTRIBUTOR_LIMIT)
      sources.push(source(norm.bias, selected.feature, "bias"));
    return mapped(
      sources,
      norm.size + Number(!!norm.weight) + Number(!!norm.bias),
      "The entire normalization group determines its mean and variance; this position uses its own scale and bias.",
    );
  }
  if (
    ["softmax", "log_softmax"].includes(op.kind) &&
    inputs.length === 1 &&
    same(inputs[0].shape, output.shape)
  ) {
    const raw = op.arguments.dim;
    if (
      !integer(Math.abs(Number(raw))) ||
      typeof raw !== "number" ||
      raw < -output.shape.length ||
      raw >= output.shape.length
    )
      return unsupported();
    const axis = (raw + output.shape.length) % output.shape.length;
    return mapped(
      normalizationGroup(output.shape, index, axis, CELL_CONTRIBUTOR_LIMIT).map(
        (i) => source(inputs[0], i, "normalization group"),
      ),
      output.shape[axis],
      "All scores along this axis share the normalization denominator.",
    );
  }
  // Table rules need only the selected recorded entry. tensorRelation validates
  // every table entry, which is unnecessary work for a single-cell inspection.
  const table = op.lesson.relation;
  if (op.lesson.interaction === "relation" && table?.rule === "table") {
    const mapping = op.lesson.mapping;
    const operand = table.operand;
    if (
      !integer(operand) ||
      !inputs[operand] ||
      !mapping ||
      mapping.length !== output.numel
    )
      return unsupported();
    const input = inputs[operand];
    const origin = mapping[index];
    if (
      !integer(origin) ||
      origin >= input.numel ||
      input.dtype !== output.dtype
    )
      return unsupported();
    if (op.kind === "embedding" && inputs.length === 2 && operand === 1) {
      const ids = inputs[0];
      const width = input.shape[1];
      if (
        input.shape.length !== 2 ||
        !["int32", "int64"].includes(ids.dtype) ||
        !same(output.shape, [...ids.shape, width]) ||
        origin % width !== index % width
      )
        return unsupported();
      const position = Math.floor(index / width);
      const recorded = ids.values?.[position];
      if (recorded !== undefined && recorded !== Math.floor(origin / width))
        return unsupported();
      return mapped(
        [
          source(ids, position, "row index"),
          source(input, origin, "table value"),
        ],
        2,
        "The recorded index selects a table row, and this feature reads its exact table cell.",
      );
    }
    if (
      ["gather", "index_select"].includes(op.kind) &&
      inputs.length === 2 &&
      operand === 0
    ) {
      const ids = inputs[1];
      const raw = op.arguments.dim;
      const rank = input.shape.length;
      if (
        !rank ||
        typeof raw !== "number" ||
        !Number.isSafeInteger(raw) ||
        raw < -rank ||
        raw >= rank ||
        !["int32", "int64"].includes(ids.dtype) ||
        (op.kind === "gather" && ids.dtype !== "int64")
      )
        return unsupported();
      const axis = (raw + rank) % rank;
      const coords = unravel(index, output.shape);
      const from = unravel(origin, input.shape);
      let position: number;
      if (op.kind === "gather") {
        if (
          ids.shape.length !== rank ||
          !same(ids.shape, output.shape) ||
          ids.shape.some((n, i) => i !== axis && n > input.shape[i])
        )
          return unsupported();
        position = index;
      } else {
        const expected = input.shape.map((n, i) =>
          i === axis ? ids.numel : n,
        );
        if (ids.shape.length !== 1 || !same(expected, output.shape))
          return unsupported();
        position = coords[axis];
      }
      if (from.some((n, i) => i !== axis && n !== coords[i]))
        return unsupported();
      const recorded = ids.values?.[position];
      if (recorded !== undefined && recorded !== from[axis])
        return unsupported();
      return mapped(
        [
          source(ids, position, "selection index"),
          source(input, origin, "selected value"),
        ],
        2,
        "The index cell chooses a position along the selected axis; the value comes from that exact input cell.",
      );
    }
    return unsupported(
      "The value’s recorded origin is known, but the complete inputs selecting it are not yet mapped.",
    );
  }
  if (
    op.lesson.interaction === "broadcast_add" &&
    inputs.length &&
    inputs.every(
      (t) =>
        t.shape.length <= output.shape.length &&
        t.shape.every((n, axis) => {
          const size =
            output.shape[output.shape.length - t.shape.length + axis];
          return n === size || n === 1;
        }),
    )
  )
    return mapped(
      inputs.map((t, operand) =>
        source(
          t,
          broadcastIndex(t.shape, output.shape, index),
          operand === 0 ? "input" : "other",
        ),
      ),
      inputs.length,
      "This cell combines one cell of each operand, at its broadcast position.",
    );
  const relation = tensorRelation(op, inputs, output);
  if (relation) {
    const input = inputs[relation.operand];
    if (["index", "tile", "pad"].includes(relation.rule)) {
      const origin = relationSource(relation, input, output, index);
      if (origin === undefined && relation.rule !== "pad") return unsupported();
      return mapped(
        origin === undefined ? [] : [source(input, origin, "input")],
        Number(origin !== undefined),
        origin === undefined
          ? "This border cell is a constant fill; it has no input cell."
          : "This cell reads the exact input position given by the recorded selection rule.",
      );
    }
    if (relation.rule === "reduce") {
      const group = reductionGroup(
        relation,
        input,
        output,
        index,
        CELL_CONTRIBUTOR_LIMIT,
      );
      return mapped(
        group.indices.map((i) => source(input, i, "reduced input")),
        group.size,
        "These input cells form the group reduced into this output cell.",
      );
    }
    if (relation.rule === "prefix") {
      const coords = unravel(index, input.shape);
      const total = coords[relation.axis] + 1;
      const sources = Array.from(
        { length: Math.min(total, CELL_CONTRIBUTOR_LIMIT) },
        (_, i) =>
          source(
            input,
            ravel(
              coords.map((n, axis) => (axis === relation.axis ? i : n)),
              input.shape,
            ),
            "prefix input",
          ),
      );
      return mapped(
        sources,
        total,
        "The prefix includes every input from the start of this axis through the selected position.",
      );
    }
    if (relation.rule === "einsum") {
      const terms = einsumTerms(
        relation,
        inputs.map((t) => t.shape),
        output.shape,
        index,
        Math.ceil(CELL_CONTRIBUTOR_LIMIT / inputs.length),
      );
      return mapped(
        terms.terms.flatMap((term) =>
          term.map((i, operand) =>
            source(inputs[operand], i, `operand ${operand + 1}`),
          ),
        ),
        terms.size * inputs.length,
        "The output coordinates fix the retained indices; the remaining index combinations form summed products.",
      );
    }
    if (relation.rule === "one_hot")
      return mapped(
        [
          source(
            input,
            Math.floor(index / output.shape.at(-1)!),
            "class index",
          ),
        ],
        1,
        "This input index decides whether the selected class position is one or zero.",
      );
    if (relation.rule === "channel_affine" && relation.operand === 0) {
      const channel = unravel(index, output.shape)[relation.axis];
      return mapped(
        inputs.map((t, operand) =>
          source(t, operand === 0 ? index : channel, relation.roles[operand]),
        ),
        inputs.length,
        "This cell uses its channel’s recorded statistics and affine parameters.",
      );
    }
    if (relation.rule === "elementwise") {
      if (!relation.roles.every((role) => typeof role === "string"))
        return unsupported();
      const origins = inputs.map((t, operand) =>
        source(
          t,
          broadcastIndex(t.shape, output.shape, index),
          relation.roles[operand],
        ),
      );
      if (relation.diagonal !== undefined) {
        if (!["tril", "triu"].includes(op.kind)) return unsupported();
        const kept = triangleKeeps(
          op.kind,
          relation.diagonal,
          unravel(index, output.shape),
        );
        return mapped(
          kept ? origins : [],
          kept ? origins.length : 0,
          kept
            ? "This position is inside the retained triangle."
            : "This position is outside the triangle and is filled with zero.",
        );
      }
      if (["where", "masked_fill"].includes(op.kind)) {
        const conditionRole = op.kind === "where" ? "condition" : "mask";
        const condition = relation.roles.indexOf(conditionRole);
        const value = inputs[condition]?.values?.[origins[condition]?.index];
        if (typeof value !== "number" || !Number.isFinite(value))
          return unsupported(
            "The recorded condition value is needed to identify the selected branch.",
          );
        const role =
          op.kind === "where"
            ? value !== 0
              ? "input"
              : "other"
            : value !== 0
              ? "value"
              : "input";
        const branch = relation.roles.indexOf(role);
        if (branch < 0 && !Object.hasOwn(op.arguments, role))
          return unsupported();
        const selected = [
          origins[condition],
          ...(branch < 0 ? [] : [origins[branch]]),
        ];
        return mapped(
          selected,
          selected.length,
          "The recorded condition and the selected branch determine this cell; the other branch is not included.",
        );
      }
      return mapped(
        origins,
        origins.length,
        "Each operand contributes its matching cell, with singleton axes reused by broadcasting.",
      );
    }
    return unsupported();
  }
  if (inputs[0]) {
    const transition = layoutTransition(op, inputs[0], output, index);
    if (transition)
      return mapped(
        [source(inputs[0], transition.source, "input")],
        1,
        "The value is preserved while its logical coordinates are rearranged.",
      );
    if (
      op.lesson.interaction === "mapping" &&
      op.lesson.mapping &&
      op.lesson.mapping.length === output.numel
    )
      return mapped(
        [source(inputs[0], op.lesson.mapping[index], "input")],
        1,
        "The recorded index map identifies this cell’s exact source.",
      );
    if (op.kind === "unfold" && op.lesson.mapping_rule === "unfold") {
      const rank = inputs[0].shape.length;
      const { dimension, size, step } = op.arguments;
      if (
        typeof dimension !== "number" ||
        !Number.isSafeInteger(dimension) ||
        dimension < -rank ||
        dimension >= rank ||
        !integer(size, 1) ||
        !integer(step, 1)
      )
        return unsupported();
      const axis = (dimension + rank) % rank;
      const expected = [...inputs[0].shape, size];
      expected[axis] = Math.floor((expected[axis] - size) / step) + 1;
      if (!same(expected, output.shape)) return unsupported();
      return mapped(
        [
          source(
            inputs[0],
            sourceIndex(op, inputs[0], output, index)!,
            "input",
          ),
        ],
        1,
        "This window position reads its exact input cell; overlapping windows can read the same cell.",
      );
    }
    if (
      op.kind === "roll" &&
      op.lesson.mapping_rule === "roll" &&
      same(inputs[0].shape, output.shape)
    ) {
      const shifts = Array.isArray(op.arguments.shifts)
        ? op.arguments.shifts
        : [op.arguments.shifts];
      const dims =
        op.arguments.dims == null
          ? []
          : Array.isArray(op.arguments.dims)
            ? op.arguments.dims
            : [op.arguments.dims];
      const rank = inputs[0].shape.length;
      if (
        !shifts.length ||
        shifts.some((n) => typeof n !== "number" || !Number.isSafeInteger(n)) ||
        (dims.length ? dims.length !== shifts.length : shifts.length !== 1) ||
        dims.some(
          (n) =>
            typeof n !== "number" ||
            !Number.isSafeInteger(n) ||
            n < -rank ||
            n >= rank,
        )
      )
        return unsupported();
      // Avoid inexact addition even for valid, extremely large shift metadata.
      const coords = unravel(index, output.shape);
      const mod = (n: bigint, size: number) =>
        Number(((n % BigInt(size)) + BigInt(size)) % BigInt(size));
      let origin: number;
      if (!dims.length)
        origin = mod(BigInt(index) - BigInt(shifts[0]), inputs[0].numel);
      else {
        dims.forEach((dim, i) => {
          const axis = (Number(dim) + rank) % rank;
          coords[axis] = mod(
            BigInt(coords[axis]) - BigInt(shifts[i]),
            inputs[0].shape[axis],
          );
        });
        origin = ravel(coords, inputs[0].shape);
      }
      return mapped(
        [source(inputs[0], origin, "input")],
        1,
        "This cell reads the shifted input coordinate, wrapping around the rolled axis.",
      );
    }
  }
  return unsupported();
}
