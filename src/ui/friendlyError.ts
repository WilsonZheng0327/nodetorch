// Translate common PyTorch error strings into student-friendly, actionable
// messages. Kept as a standalone pure function (exported) so it can be unit
// tested against the raw backend error copy.

/** Translate a common PyTorch error into a student-friendly message, or return
 *  the original string unchanged if none matches. */
export function friendlyError(msg: string): string {
  if (msg.includes('mat1 and mat2 shapes cannot be multiplied')) {
    const match = msg.match(/(\d+x\d+).*?(\d+x\d+)/);
    if (match)
      return `Shape mismatch in Linear layer: input is ${match[1]} but weights expect ${match[2]}. Check upstream layer output size.`;
  }
  if (msg.includes('Expected 4-dimensional input'))
    return 'This layer expects a 4D tensor [B,C,H,W]. Add a Reshape node or check connections.';
  if (msg.includes('Expected 3-dimensional input'))
    return 'This layer expects a 3D tensor [B,seq,features]. Check input dimensions.';
  if (msg.includes('Expected 2-dimensional input'))
    return 'This layer expects a 2D tensor [B,features]. Did you forget a Flatten layer?';
  if (msg.includes('size mismatch'))
    return `Tensor size mismatch — shapes don't align. Check that connected layers have compatible dimensions.`;
  if (msg.includes('CUDA out of memory'))
    return 'GPU out of memory. Try reducing batch size or using CPU.';
  if (msg.includes('is not a valid device'))
    return 'Selected device not available. Switch to CPU in the dashboard System tab.';
  if (msg.includes('negative dimension'))
    return 'Layer produced a negative dimension — kernel/stride/padding combination is too large for the input size.';
  return msg;
}
