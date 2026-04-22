// Unit tests for the Tokenizer node's shape executor.
// Verifies output shape computation, metadata, and edge cases.

import { describe, it, expect } from 'vitest';

import { tokenizerNode } from '../../../src/domain/nodes/preprocessing/tokenizer';

/** Shorthand to grab the shape executor. */
function shapeExecutor(node: { executors: Record<string, { execute: Function }> }) {
  return node.executors.shape;
}

// ---------------------------------------------------------------------------
// Tokenizer shape executor
// ---------------------------------------------------------------------------

describe('Tokenizer shape executor', () => {
  const executor = shapeExecutor(tokenizerNode);

  it('outputs [B, maxLen] from [B, L] input', async () => {
    const result = await executor.execute({
      inputs: { in: [32, 256] },
      properties: { mode: 'word', vocabSize: 10000, maxLen: 128 },
    });
    expect(result.outputs.out).toEqual([32, 128]);
  });

  it('preserves batch dimension', async () => {
    const result = await executor.execute({
      inputs: { in: [64, 512] },
      properties: { mode: 'character', vocabSize: 100, maxLen: 256 },
    });
    expect(result.outputs.out).toEqual([64, 256]);
  });

  it('returns empty outputs when no input', async () => {
    const result = await executor.execute({
      inputs: {},
      properties: { mode: 'word', vocabSize: 10000, maxLen: 128 },
    });
    expect(result.outputs).toEqual({});
  });

  it('includes metadata with mode and vocab info', async () => {
    const result = await executor.execute({
      inputs: { in: [32, 256] },
      properties: { mode: 'bpe', vocabSize: 50000, maxLen: 512 },
    });
    expect(result.metadata).toBeDefined();
    expect(result.metadata.outputShape).toEqual([32, 512]);
    expect(result.metadata.shapes).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ label: 'Mode', value: 'Byte-Pair Encoding' }),
      ]),
    );
  });

  it('maps mode names to readable labels', async () => {
    const modes = [
      { mode: 'character', expected: 'Character-level' },
      { mode: 'word', expected: 'Word-level' },
      { mode: 'bpe', expected: 'Byte-Pair Encoding' },
    ];
    for (const { mode, expected } of modes) {
      const result = await executor.execute({
        inputs: { in: [1, 10] },
        properties: { mode, vocabSize: 100, maxLen: 10 },
      });
      const modeShape = result.metadata.shapes.find(
        (s: { label: string }) => s.label === 'Mode',
      );
      expect(modeShape?.value).toBe(expected);
    }
  });

  it('works with maxLen=1 (minimum)', async () => {
    const result = await executor.execute({
      inputs: { in: [8, 1000] },
      properties: { mode: 'word', vocabSize: 5000, maxLen: 1 },
    });
    expect(result.outputs.out).toEqual([8, 1]);
  });
});
