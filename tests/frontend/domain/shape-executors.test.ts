// Unit tests for shape executors on node definitions.
// Each test calls the node's executors.shape.execute() directly with
// a minimal context and checks the computed output shapes.

import { describe, it, expect } from 'vitest';

import { conv2dNode } from '../../../src/domain/nodes/layers/conv2d';
import { conv1dNode } from '../../../src/domain/nodes/layers/conv1d';
import { linearNode } from '../../../src/domain/nodes/layers/linear';
import { flattenNode } from '../../../src/domain/nodes/layers/flatten';
import { reluNode } from '../../../src/domain/nodes/activations/relu';
import { sigmoidNode } from '../../../src/domain/nodes/activations/sigmoid';
import { geluNode } from '../../../src/domain/nodes/activations/gelu';
import { tanhNode } from '../../../src/domain/nodes/activations/tanh';
import { leakyReluNode } from '../../../src/domain/nodes/activations/leaky-relu';
import { maxPool2dNode } from '../../../src/domain/nodes/layers/maxpool2d';
import { maxPool1dNode } from '../../../src/domain/nodes/layers/maxpool1d';
import { avgPool2dNode } from '../../../src/domain/nodes/layers/avgpool2d';
import { adaptiveAvgPool2dNode } from '../../../src/domain/nodes/layers/adaptive-avgpool2d';
import { batchNorm2dNode } from '../../../src/domain/nodes/layers/batchnorm2d';
import { batchNorm1dNode } from '../../../src/domain/nodes/layers/batchnorm1d';
import { groupNormNode } from '../../../src/domain/nodes/layers/groupnorm';
import { instanceNorm2dNode } from '../../../src/domain/nodes/layers/instancenorm2d';
import { layerNormNode } from '../../../src/domain/nodes/layers/layernorm';
import { dropoutNode } from '../../../src/domain/nodes/layers/dropout';
import { dropout2dNode } from '../../../src/domain/nodes/layers/dropout2d';
import { embeddingNode } from '../../../src/domain/nodes/layers/embedding';
import { positionalEncodingNode } from '../../../src/domain/nodes/layers/positional-encoding';
import { multiHeadAttentionNode } from '../../../src/domain/nodes/layers/multihead-attention';
import { attentionNode } from '../../../src/domain/nodes/layers/attention';
import { lstmNode } from '../../../src/domain/nodes/layers/lstm';
import { gruNode } from '../../../src/domain/nodes/layers/gru';
import { rnnNode } from '../../../src/domain/nodes/layers/rnn';
import { upsampleNode } from '../../../src/domain/nodes/layers/upsample';
import { reshapeNode } from '../../../src/domain/nodes/structural/reshape';
import { convTranspose2dNode } from '../../../src/domain/nodes/layers/conv-transpose2d';
import { softmaxNode } from '../../../src/domain/nodes/activations/softmax';
import { addNode } from '../../../src/domain/nodes/structural/add';
import { concatNode } from '../../../src/domain/nodes/structural/concat';
import { permuteNode } from '../../../src/domain/nodes/structural/permute';
import { sequencePoolNode } from '../../../src/domain/nodes/structural/sequence-pool';
import { reparameterizeNode } from '../../../src/domain/nodes/structural/reparameterize';
import { crossEntropyLossNode } from '../../../src/domain/nodes/loss/cross-entropy-loss';
import { mseLossNode } from '../../../src/domain/nodes/loss/mse-loss';
import { vaeLossNode } from '../../../src/domain/nodes/loss/vae-loss';
import { ganLossNode } from '../../../src/domain/nodes/loss/gan-loss';

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

// ---------------------------------------------------------------------------
// Conv1d
// ---------------------------------------------------------------------------

describe('Conv1d shape executor', () => {
  const executor = shapeExecutor(conv1dNode);

  it('computes output shape with padding=1', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 3, 32] },
      properties: { outChannels: 16, kernelSize: 3, stride: 1, padding: 1 },
    });
    expect(result.outputs.out).toEqual([2, 16, 32]);
  });

  it('computes output shape with stride=2', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 3, 32] },
      properties: { outChannels: 16, kernelSize: 3, stride: 2, padding: 0 },
    });
    expect(result.outputs.out).toEqual([2, 16, 15]);
  });
});

// ---------------------------------------------------------------------------
// MaxPool1d
// ---------------------------------------------------------------------------

describe('MaxPool1d shape executor', () => {
  const executor = shapeExecutor(maxPool1dNode);

  it('halves length with kernel=2, stride=2', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 4, 32] },
      properties: { kernelSize: 2, stride: 2, padding: 0 },
    });
    expect(result.outputs.out).toEqual([2, 4, 16]);
  });
});

// ---------------------------------------------------------------------------
// AvgPool2d
// ---------------------------------------------------------------------------

describe('AvgPool2d shape executor', () => {
  const executor = shapeExecutor(avgPool2dNode);

  it('halves spatial dims with kernel=2, stride=2', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 3, 64, 64] },
      properties: { kernelSize: 2, stride: 2, padding: 0 },
    });
    expect(result.outputs.out).toEqual([2, 3, 32, 32]);
  });
});

// ---------------------------------------------------------------------------
// AdaptiveAvgPool2d
// ---------------------------------------------------------------------------

describe('AdaptiveAvgPool2d shape executor', () => {
  const executor = shapeExecutor(adaptiveAvgPool2dNode);

  it('outputs target spatial size', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 512, 7, 7] },
      properties: { outputH: 1, outputW: 1 },
    });
    expect(result.outputs.out).toEqual([2, 512, 1, 1]);
  });
});

// ---------------------------------------------------------------------------
// BatchNorm1d
// ---------------------------------------------------------------------------

describe('BatchNorm1d shape executor', () => {
  const executor = shapeExecutor(batchNorm1dNode);

  it('passes through shape unchanged', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 16, 32] },
      properties: {},
    });
    expect(result.outputs.out).toEqual([2, 16, 32]);
  });
});

// ---------------------------------------------------------------------------
// GroupNorm
// ---------------------------------------------------------------------------

describe('GroupNorm shape executor', () => {
  const executor = shapeExecutor(groupNormNode);

  it('passes through shape unchanged', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 32, 16, 16] },
      properties: { numGroups: 8 },
    });
    expect(result.outputs.out).toEqual([2, 32, 16, 16]);
  });
});

// ---------------------------------------------------------------------------
// InstanceNorm2d
// ---------------------------------------------------------------------------

describe('InstanceNorm2d shape executor', () => {
  const executor = shapeExecutor(instanceNorm2dNode);

  it('passes through shape unchanged', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 3, 64, 64] },
      properties: {},
    });
    expect(result.outputs.out).toEqual([2, 3, 64, 64]);
  });
});

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

describe('LayerNorm shape executor', () => {
  const executor = shapeExecutor(layerNormNode);

  it('passes through shape unchanged', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 10, 256] },
      properties: { numLastDims: 1 },
    });
    expect(result.outputs.out).toEqual([2, 10, 256]);
  });
});

// ---------------------------------------------------------------------------
// Dropout2d
// ---------------------------------------------------------------------------

describe('Dropout2d shape executor', () => {
  const executor = shapeExecutor(dropout2dNode);

  it('passes through shape unchanged', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 3, 32, 32] },
      properties: { p: 0.5 },
    });
    expect(result.outputs.out).toEqual([2, 3, 32, 32]);
  });
});

// ---------------------------------------------------------------------------
// PositionalEncoding
// ---------------------------------------------------------------------------

describe('PositionalEncoding shape executor', () => {
  const executor = shapeExecutor(positionalEncodingNode);

  it('passes through shape unchanged', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 10, 64] },
      properties: { maxLen: 512, encodingType: 'learned' },
    });
    expect(result.outputs.out).toEqual([2, 10, 64]);
  });

  it('returns empty outputs when no input', async () => {
    const result = await executor.execute({
      inputs: {},
      properties: { maxLen: 512, encodingType: 'learned' },
    });
    expect(result.outputs).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// MultiHeadAttention
// ---------------------------------------------------------------------------

describe('MultiHeadAttention shape executor', () => {
  const executor = shapeExecutor(multiHeadAttentionNode);

  it('outputs [B, seq_q, embed_dim]', async () => {
    const result = await executor.execute({
      inputs: { query: [2, 10, 256], key: [2, 10, 256], value: [2, 10, 256] },
      properties: { embedDim: 256, numHeads: 8, dropout: 0, causalMask: false },
    });
    expect(result.outputs.out).toEqual([2, 10, 256]);
  });

  it('errors when embedDim not divisible by numHeads', async () => {
    const result = await executor.execute({
      inputs: { query: [2, 10, 256], key: [2, 10, 256], value: [2, 10, 256] },
      properties: { embedDim: 256, numHeads: 3, dropout: 0, causalMask: false },
    });
    expect(result.metadata.error).toBeDefined();
  });
});

// ---------------------------------------------------------------------------
// Attention (scaled dot-product)
// ---------------------------------------------------------------------------

describe('Attention shape executor', () => {
  const executor = shapeExecutor(attentionNode);

  it('outputs same shape as query', async () => {
    const result = await executor.execute({
      inputs: { query: [2, 10, 64], key: [2, 10, 64], value: [2, 10, 64] },
      properties: { dropout: 0, causalMask: false },
    });
    expect(result.outputs.out).toEqual([2, 10, 64]);
  });
});

// ---------------------------------------------------------------------------
// GRU
// ---------------------------------------------------------------------------

describe('GRU shape executor', () => {
  const executor = shapeExecutor(gruNode);

  it('computes output and hidden shapes', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 10, 32] },
      properties: { hiddenSize: 64, numLayers: 1, bidirectional: false, dropout: 0 },
    });
    expect(result.outputs.out).toEqual([2, 10, 64]);
    expect(result.outputs.hidden).toEqual([1, 2, 64]);
  });
});

// ---------------------------------------------------------------------------
// RNN
// ---------------------------------------------------------------------------

describe('RNN shape executor', () => {
  const executor = shapeExecutor(rnnNode);

  it('computes output and hidden shapes', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 10, 32] },
      properties: { hiddenSize: 64, numLayers: 1, bidirectional: false, nonlinearity: 'tanh' },
    });
    expect(result.outputs.out).toEqual([2, 10, 64]);
    expect(result.outputs.hidden).toEqual([1, 2, 64]);
  });
});

// ---------------------------------------------------------------------------
// Upsample
// ---------------------------------------------------------------------------

describe('Upsample shape executor', () => {
  const executor = shapeExecutor(upsampleNode);

  it('doubles spatial dims with scaleFactor=2', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 3, 32, 32] },
      properties: { scaleFactor: 2, mode: 'nearest' },
    });
    expect(result.outputs.out).toEqual([2, 3, 64, 64]);
  });
});

// ---------------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------------

describe('Sigmoid shape executor', () => {
  const executor = shapeExecutor(sigmoidNode);

  it('passes through shape unchanged', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 10] },
      properties: {},
    });
    expect(result.outputs.out).toEqual([2, 10]);
  });
});

// ---------------------------------------------------------------------------
// GELU
// ---------------------------------------------------------------------------

describe('GELU shape executor', () => {
  const executor = shapeExecutor(geluNode);

  it('passes through shape unchanged', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 256] },
      properties: {},
    });
    expect(result.outputs.out).toEqual([2, 256]);
  });
});

// ---------------------------------------------------------------------------
// Tanh
// ---------------------------------------------------------------------------

describe('Tanh shape executor', () => {
  const executor = shapeExecutor(tanhNode);

  it('passes through shape unchanged', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 10] },
      properties: {},
    });
    expect(result.outputs.out).toEqual([2, 10]);
  });
});

// ---------------------------------------------------------------------------
// LeakyReLU
// ---------------------------------------------------------------------------

describe('LeakyReLU shape executor', () => {
  const executor = shapeExecutor(leakyReluNode);

  it('passes through shape unchanged', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 64] },
      properties: { negativeSlope: 0.01 },
    });
    expect(result.outputs.out).toEqual([2, 64]);
  });
});

// ---------------------------------------------------------------------------
// Concat
// ---------------------------------------------------------------------------

describe('Concat shape executor', () => {
  const executor = shapeExecutor(concatNode);

  it('concatenates along dim=1', async () => {
    const result = await executor.execute({
      inputs: { in_0: [2, 3, 4], in_1: [2, 5, 4] },
      properties: { numInputs: 2, dim: 1 },
    });
    expect(result.outputs.out).toEqual([2, 8, 4]);
  });
});

// ---------------------------------------------------------------------------
// Permute
// ---------------------------------------------------------------------------

describe('Permute shape executor', () => {
  const executor = shapeExecutor(permuteNode);

  it('reorders dimensions', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 10, 5] },
      properties: { dims: '0, 2, 1' },
    });
    expect(result.outputs.out).toEqual([2, 5, 10]);
  });
});

// ---------------------------------------------------------------------------
// SequencePool
// ---------------------------------------------------------------------------

describe('SequencePool shape executor', () => {
  const executor = shapeExecutor(sequencePoolNode);

  it('removes sequence dimension (last mode)', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 10, 256] },
      properties: { mode: 'last' },
    });
    expect(result.outputs.out).toEqual([2, 256]);
  });

  it('removes sequence dimension (mean mode)', async () => {
    const result = await executor.execute({
      inputs: { in: [2, 10, 256] },
      properties: { mode: 'mean' },
    });
    expect(result.outputs.out).toEqual([2, 256]);
  });
});

// ---------------------------------------------------------------------------
// Reparameterize
// ---------------------------------------------------------------------------

describe('Reparameterize shape executor', () => {
  const executor = shapeExecutor(reparameterizeNode);

  it('outputs same shape as mean', async () => {
    const result = await executor.execute({
      inputs: { mean: [2, 32], logvar: [2, 32] },
      properties: {},
    });
    expect(result.outputs.out).toEqual([2, 32]);
  });
});

// ---------------------------------------------------------------------------
// VAE Loss
// ---------------------------------------------------------------------------

describe('VAELoss shape executor', () => {
  const executor = shapeExecutor(vaeLossNode);

  it('outputs scalar', async () => {
    const result = await executor.execute({
      inputs: {
        reconstruction: [2, 784],
        original: [2, 784],
        mean: [2, 32],
        logvar: [2, 32],
      },
      properties: { beta: 1.0 },
    });
    expect(result.outputs.out).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// GAN Loss
// ---------------------------------------------------------------------------

describe('GAN Loss shape executor', () => {
  const executor = shapeExecutor(ganLossNode);

  it('outputs scalar', async () => {
    const result = await executor.execute({
      inputs: { real_scores: [2, 1], fake_scores: [2, 1] },
      properties: { labelSmoothing: 0.1 },
    });
    expect(result.outputs.out).toEqual([]);
  });
});
