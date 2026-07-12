// Tests for the data.custom node: its props-only shape executor across
// modalities, and that it's registered in the data-node list.

import { describe, it, expect } from 'vitest';

import { customNode } from '../../../src/domain/nodes/data/custom';
import { dataNodes } from '../../../src/domain/nodes/data';

// Full properties = every default, then per-case overrides (mirrors how a real
// node instance is initialized from its property defaults).
const defaults = Object.fromEntries(customNode.getProperties().map((p) => [p.id, p.defaultValue]));
const run = (over: Record<string, unknown>) =>
  customNode.executors.shape.execute({ inputs: {}, properties: { ...defaults, ...over } });

describe('data.custom shape executor', () => {
  it('image + classification → [B, C, H, W] and scalar labels', async () => {
    const r = await run({
      inputKind: 'image',
      task: 'classification',
      channels: 1,
      imageSize: 28,
      batchSize: 8,
    });
    expect(r.outputs.out).toEqual([8, 1, 28, 28]);
    expect(r.outputs.labels).toEqual([8]);
    expect(r.metadata?.datasetType).toBe('data.custom');
  });

  it('image respects RGB channels', async () => {
    const r = await run({ inputKind: 'image', channels: 3, imageSize: 16, batchSize: 4 });
    expect(r.outputs.out).toEqual([4, 3, 16, 16]);
  });

  it('text + classification → [B, maxLen] and scalar labels', async () => {
    const r = await run({ inputKind: 'text', task: 'classification', maxLen: 64, batchSize: 8 });
    expect(r.outputs.out).toEqual([8, 64]);
    expect(r.outputs.labels).toEqual([8]);
  });

  it('text + language_model → [B, seqLen] input and target', async () => {
    const r = await run({ inputKind: 'text', task: 'language_model', seqLen: 32, batchSize: 8 });
    expect(r.outputs.out).toEqual([8, 32]);
    expect(r.outputs.labels).toEqual([8, 32]);
  });
});

describe('data.custom registration & conditional props', () => {
  it('is registered in the data-node list', () => {
    expect(dataNodes.some((n) => n.type === 'data.custom')).toBe(true);
  });

  it('shows image params only for image input', () => {
    const imageSize = customNode.getProperties().find((p) => p.id === 'imageSize')!;
    expect(imageSize.visible?.({ inputKind: 'image' })).toBe(true);
    expect(imageSize.visible?.({ inputKind: 'text' })).toBe(false);
  });

  it('shows seqLen for LM and maxLen otherwise', () => {
    const props = customNode.getProperties();
    const seqLen = props.find((p) => p.id === 'seqLen')!;
    const maxLen = props.find((p) => p.id === 'maxLen')!;
    expect(seqLen.visible?.({ inputKind: 'text', task: 'language_model' })).toBe(true);
    expect(maxLen.visible?.({ inputKind: 'text', task: 'language_model' })).toBe(false);
    expect(maxLen.visible?.({ inputKind: 'text', task: 'classification' })).toBe(true);
  });

  it('hides the label column for language models', () => {
    const labelColumn = customNode.getProperties().find((p) => p.id === 'labelColumn')!;
    expect(labelColumn.visible?.({ task: 'classification' })).toBe(true);
    expect(labelColumn.visible?.({ task: 'language_model' })).toBe(false);
  });
});
