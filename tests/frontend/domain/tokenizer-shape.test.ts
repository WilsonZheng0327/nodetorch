// Unit tests for the Tokenizer nodes' shape executors.
// The tokenizer was split from one node with a `mode` property into three
// distinct node types (char / word / bpe); each bakes the mode into its type
// and reports it as a fixed "Mode" metadata label. All three share the same
// shape behavior: [B, L] in → [B, maxLen] out.

import { describe, it, expect } from 'vitest';

import { tokenizerCharNode } from '../../../src/domain/nodes/preprocessing/tokenizer-char';
import { tokenizerWordNode } from '../../../src/domain/nodes/preprocessing/tokenizer-word';
import { tokenizerBpeNode } from '../../../src/domain/nodes/preprocessing/tokenizer-bpe';

/** Shorthand to grab the shape executor. */
function shapeExecutor(node: { executors: Record<string, { execute: Function }> }) {
  return node.executors.shape;
}

const TOKENIZERS = [
  { node: tokenizerCharNode, label: 'Character-level' },
  { node: tokenizerWordNode, label: 'Word-level' },
  { node: tokenizerBpeNode, label: 'Byte-Pair Encoding' },
];

describe.each(TOKENIZERS)('Tokenizer shape executor — $label', ({ node, label }) => {
  const executor = shapeExecutor(node);

  it('outputs [B, maxLen] from [B, L] input', async () => {
    const result = await executor.execute({
      inputs: { in: [32, 256] },
      properties: { vocabSize: 10000, maxLen: 128 },
    });
    expect(result.outputs.out).toEqual([32, 128]);
  });

  it('preserves the batch dimension', async () => {
    const result = await executor.execute({
      inputs: { in: [64, 512] },
      properties: { vocabSize: 100, maxLen: 256 },
    });
    expect(result.outputs.out).toEqual([64, 256]);
  });

  it('returns empty outputs when there is no input', async () => {
    const result = await executor.execute({
      inputs: {},
      properties: { vocabSize: 10000, maxLen: 128 },
    });
    expect(result.outputs).toEqual({});
  });

  it('reports its mode and output shape in metadata', async () => {
    const result = await executor.execute({
      inputs: { in: [32, 256] },
      properties: { vocabSize: 50000, maxLen: 512 },
    });
    expect(result.metadata.outputShape).toEqual([32, 512]);
    expect(result.metadata.shapes).toEqual(
      expect.arrayContaining([expect.objectContaining({ label: 'Mode', value: label })]),
    );
  });

  it('works with maxLen=1 (minimum)', async () => {
    const result = await executor.execute({
      inputs: { in: [8, 1000] },
      properties: { vocabSize: 5000, maxLen: 1 },
    });
    expect(result.outputs.out).toEqual([8, 1]);
  });
});
