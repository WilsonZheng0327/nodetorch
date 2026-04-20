// Unit tests for shape executors on node definitions.
// Each test calls the node's executors.shape.execute() directly with
// a minimal context and checks the computed output shapes.

import { describe, it, expect } from 'vitest';

import { conv2dNode } from '../../../src/domain/nodes/layers/conv2d';
import { linearNode } from '../../../src/domain/nodes/layers/linear';
import { flattenNode } from '../../../src/domain/nodes/layers/flatten';
import { reluNode } from '../../../src/domain/nodes/activations/relu';
import { maxPool2dNode } from '../../../src/domain/nodes/layers/maxpool2d';
import { batchNorm2dNode } from '../../../src/domain/nodes/layers/batchnorm2d';
import { dropoutNode } from '../../../src/domain/nodes/layers/dropout';
import { embeddingNode } from '../../../src/domain/nodes/layers/embedding';
import { lstmNode } from '../../../src/domain/nodes/layers/lstm';
import { reshapeNode } from '../../../src/domain/nodes/structural/reshape';
import { convTranspose2dNode } from '../../../src/domain/nodes/layers/conv-transpose2d';
import { softmaxNode } from '../../../src/domain/nodes/activations/softmax';
import { addNode } from '../../../src/domain/nodes/structural/add';
import { crossEntropyLossNode } from '../../../src/domain/nodes/loss/cross-entropy-loss';
import { mseLossNode } from '../../../src/domain/nodes/loss/mse-loss';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Shorthand to grab the shape executor from a node definition. */
function shapeExecutor(node: { executors: Record<string, { execute: Function }> }) {
  return node.executors.shape;
}

// ---------------------------------------------------------------------------
// Conv2d
// ---------------------------------------------------------------------------

describe('Conv2d shape executor', () => {
  const executor = shapeExecutor(conv2dNode);

  it('computes output shape with padding=1 (same spatial size)', async () => {
    const result = await executor.execute({
      inputs: { in: [1, 3, 32, 32] },
      properties: { outChannels: 16, kernelSize: 3, stride: 1, padding: 1 },
    });
    expect(result.outputs.out).toEqual([1, 16, 32, 32]);
  });

  it('computes output shape with stride=2 and padding=0 (shrinks)', async () => {
    const result = await executor.execute({
      inputs: { in: [1, 3, 32, 32] },
      properties: { outChannels: 16, kernelSize: 3, stride: 2, padding: 0 },
    });
    // floor((32 + 0 - 3) / 2) + 1 = floor(29/2) + 1 = 14 + 1 = 15
    expect(result.outputs.out).toEqual([1, 16, 15, 15]);
  });

  it('returns empty outputs when no input is provided', async () => {
    const result = await executor.execute({
      inputs: {},
      properties: { outChannels: 16, kernelSize: 3, stride: 1, padding: 0 },
    });
    expect(result.outputs).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// Linear
// ---------------------------------------------------------------------------

describe('Linear shape executor', () => {
  const executor = shapeExecutor(linearNode);

  it('maps last dimension to outFeatures', async () => {
    const result = await executor.execute({
      inputs: { in: [1, 512] },
      properties: { outFeatures: 10 },
    });
    expect(result.outputs.out).toEqual([1, 10]);
  });
});

// ---------------------------------------------------------------------------
// Flatten
// ---------------------------------------------------------------------------

describe('Flatten shape executor', () => {
  const executor = shapeExecutor(flattenNode);

  it('flattens spatial dims after batch', async () => {
    const result = await executor.execute({
      inputs: { in: [1, 64, 7, 7] },
      properties: {},
    });
    // 64 * 7 * 7 = 3136
    expect(result.outputs.out).toEqual([1, 3136]);
  });
});

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

describe('ReLU shape executor', () => {
  const executor = shapeExecutor(reluNode);

  it('passes through shape unchanged', async () => {
    const result = await executor.execute({
      inputs: { in: [1, 64, 32, 32] },
      properties: {},
    });
    expect(result.outputs.out).toEqual([1, 64, 32, 32]);
  });
});

// ---------------------------------------------------------------------------
// MaxPool2d
// ---------------------------------------------------------------------------

describe('MaxPool2d shape executor', () => {
  const executor = shapeExecutor(maxPool2dNode);

  it('halves spatial dimensions with kernel=2, stride=2', async () => {
    const result = await executor.execute({
      inputs: { in: [1, 64, 32, 32] },
      properties: { kernelSize: 2, stride: 2, padding: 0 },
    });
    expect(result.outputs.out).toEqual([1, 64, 16, 16]);
  });
});

// ---------------------------------------------------------------------------
// BatchNorm2d
// ---------------------------------------------------------------------------

describe('BatchNorm2d shape executor', () => {
  const executor = shapeExecutor(batchNorm2dNode);

  it('passes through shape unchanged', async () => {
    const result = await executor.execute({
      inputs: { in: [1, 64, 32, 32] },
      properties: {},
    });
    expect(result.outputs.out).toEqual([1, 64, 32, 32]);
  });
});

// ---------------------------------------------------------------------------
// Dropout
// ---------------------------------------------------------------------------

describe('Dropout shape executor', () => {
  const executor = shapeExecutor(dropoutNode);

  it('passes through shape unchanged', async () => {
    const result = await executor.execute({
      inputs: { in: [1, 512] },
      properties: { p: 0.5 },
    });
    expect(result.outputs.out).toEqual([1, 512]);
  });
});

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

describe('Embedding shape executor', () => {
  const executor = shapeExecutor(embeddingNode);

  it('appends embeddingDim to input shape', async () => {
    const result = await executor.execute({
      inputs: { in: [1, 128] },
      properties: { numEmbeddings: 10000, embeddingDim: 64 },
    });
    // [1, 128] → [1, 128, 64]
    expect(result.outputs.out).toEqual([1, 128, 64]);
  });
});

// ---------------------------------------------------------------------------
// LSTM
// ---------------------------------------------------------------------------

describe('LSTM shape executor', () => {
  const executor = shapeExecutor(lstmNode);

  it('computes output and hidden/cell shapes', async () => {
    const result = await executor.execute({
      inputs: { in: [1, 128, 64] },
      properties: { hiddenSize: 128, numLayers: 1, bidirectional: false, dropout: 0 },
    });
    expect(result.outputs.out).toEqual([1, 128, 128]);
    expect(result.outputs.hidden).toEqual([1, 1, 128]);
    expect(result.outputs.cell).toEqual([1, 1, 128]);
  });
});

// ---------------------------------------------------------------------------
// Reshape
// ---------------------------------------------------------------------------

describe('Reshape shape executor', () => {
  const executor = shapeExecutor(reshapeNode);

  it('reshapes flat tensor to spatial dims with -1 for batch', async () => {
    const result = await executor.execute({
      inputs: { in: [1, 392] },
      properties: { targetShape: '-1, 8, 7, 7' },
    });
    // total = 1*392 = 392; known = 8*7*7 = 392; -1 → 392/392 = 1
    expect(result.outputs.out).toEqual([1, 8, 7, 7]);
  });
});

// ---------------------------------------------------------------------------
// ConvTranspose2d
// ---------------------------------------------------------------------------

describe('ConvTranspose2d shape executor', () => {
  const executor = shapeExecutor(convTranspose2dNode);

  it('doubles spatial dims with kernel=4, stride=2, padding=1', async () => {
    const result = await executor.execute({
      inputs: { in: [1, 8, 7, 7] },
      properties: { outChannels: 16, kernelSize: 4, stride: 2, padding: 1, outputPadding: 0 },
    });
    // outH = (7-1)*2 - 2*1 + 4 + 0 = 12 - 2 + 4 = 14
    expect(result.outputs.out).toEqual([1, 16, 14, 14]);
  });
});

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

describe('Softmax shape executor', () => {
  const executor = shapeExecutor(softmaxNode);

  it('passes through shape unchanged', async () => {
    const result = await executor.execute({
      inputs: { in: [1, 10] },
      properties: { dim: -1 },
    });
    expect(result.outputs.out).toEqual([1, 10]);
  });
});

// ---------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------

describe('Add shape executor', () => {
  const executor = shapeExecutor(addNode);

  it('outputs same shape for matching inputs', async () => {
    const result = await executor.execute({
      inputs: { a: [1, 64, 32, 32], b: [1, 64, 32, 32] },
      properties: {},
    });
    expect(result.outputs.out).toEqual([1, 64, 32, 32]);
  });
});

// ---------------------------------------------------------------------------
// CrossEntropyLoss
// ---------------------------------------------------------------------------

describe('CrossEntropyLoss shape executor', () => {
  const executor = shapeExecutor(crossEntropyLossNode);

  it('outputs scalar from predictions [B,C] and labels [B]', async () => {
    const result = await executor.execute({
      inputs: { predictions: [1, 10], labels: [1] },
      properties: {},
    });
    // scalar output → empty array
    expect(result.outputs.out).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// MSELoss
// ---------------------------------------------------------------------------

describe('MSELoss shape executor', () => {
  const executor = shapeExecutor(mseLossNode);

  it('outputs scalar when predictions and labels shapes match', async () => {
    const result = await executor.execute({
      inputs: { predictions: [1, 10], labels: [1, 10] },
      properties: {},
    });
    // scalar output → empty array
    expect(result.outputs.out).toEqual([]);
  });
});
