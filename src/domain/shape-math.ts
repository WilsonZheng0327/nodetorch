// Shared spatial output-size math for conv/pool layers (Layer 5 domain helper).
// One source of truth for the formula that was hand-written across every
// conv/pool node's shape executor.

/**
 * Output size of a conv/pool along one spatial dimension:
 * `floor((dim + 2·padding − kernel) / stride) + 1`.
 */
export function convOutputSize(
  dim: number,
  kernel: number,
  padding: number,
  stride: number,
): number {
  return Math.floor((dim + 2 * padding - kernel) / stride) + 1;
}

/**
 * Output size of a transposed conv (upsampling) along one spatial dimension:
 * `(dim − 1)·stride − 2·padding + kernel + outputPadding`.
 */
export function convTransposeOutputSize(
  dim: number,
  kernel: number,
  padding: number,
  stride: number,
  outputPadding = 0,
): number {
  return (dim - 1) * stride - 2 * padding + kernel + outputPadding;
}
