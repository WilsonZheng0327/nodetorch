// Tests for the agent graph-tool executor (src/ui/chat/graphTools.ts).
// Uses a fake GraphToolApi backed by a real Graph so we exercise the tool
// dispatch, validation, and result strings without the full useGraph hook.

import { describe, it, expect, vi } from 'vitest';

import { initDomain } from '../../../src/domain';
import { createGraph, createNode, addNode, createEdge, addEdge } from '../../../src/core/graph';
import { executeGraphTool, type GraphToolApi } from '../../../src/ui/chat/graphTools';
import { callBackend } from '../../../src/api/client';

// get_dataset_info calls the backend; stub it so tests stay offline.
vi.mock('../../../src/api/client', () => ({ callBackend: vi.fn() }));

const domain = initDomain();

function makeApi() {
  const g = createGraph('main', 'Main');
  addNode(g, createNode('c1', 'ml.layers.conv2d', { x: 0, y: 0 }, { outChannels: 32 }));
  const flags = { organized: false, blockAdded: '', entered: '', exited: 0, saved: '' };

  const api: GraphToolApi = {
    getCurrentGraph: () => g,
    updateProperty: async (id, k, v) => {
      const n = g.nodes.get(id);
      if (n) (n.properties as Record<string, unknown>)[k] = v;
    },
    addNode: async (type, pos, requestedId) => {
      const id = requestedId && !g.nodes.has(requestedId) ? requestedId : `${type}-new`;
      addNode(g, createNode(id, type, pos, {}));
      return id;
    },
    connect: async () => {},
    removeNode: async (id) => {
      g.nodes.delete(id);
    },
    removeEdge: async (s, sp, t, tp) => {
      const i = g.edges.findIndex(
        (e) =>
          e.source.nodeId === s &&
          e.source.portId === sp &&
          e.target.nodeId === t &&
          e.target.portId === tp,
      );
      if (i < 0) return false;
      g.edges.splice(i, 1);
      return true;
    },
    clearGraph: () => {
      g.nodes.clear();
      g.edges = [];
    },
    organizeGraph: () => {
      flags.organized = true;
    },
    addBlockFromTemplate: async (filename) => {
      flags.blockAdded = filename;
    },
    enterSubgraph: (id) => {
      flags.entered = id;
    },
    exitSubgraph: () => {
      flags.exited += 1;
    },
    saveBlock: async (id) => {
      flags.saved = id;
    },
    // fake validation: reject only the sentinel 'bad' target port
    isValidConnection: (c) => c.targetHandle !== 'bad',
    trainingProgress: [],
    modelTrained: false,
    modelStale: false,
  };
  return { api, g, flags };
}

describe('executeGraphTool', () => {
  it('set_node_property applies, and errors on missing node / invalid key', async () => {
    const { api, g } = makeApi();
    expect(
      await executeGraphTool(api, domain, 'set_node_property', {
        nodeId: 'c1',
        key: 'outChannels',
        value: 64,
      }),
    ).toMatch(/^ok/);
    expect(g.nodes.get('c1')!.properties.outChannels).toBe(64);
    expect(
      await executeGraphTool(api, domain, 'set_node_property', {
        nodeId: 'nope',
        key: 'x',
        value: 1,
      }),
    ).toMatch(/^error/);
    expect(
      await executeGraphTool(api, domain, 'set_node_property', {
        nodeId: 'c1',
        key: 'bogus',
        value: 1,
      }),
    ).toMatch(/^error/);
  });

  it('add_node ok for a valid type, error for unknown, honors a requested id', async () => {
    const { api, g } = makeApi();
    expect(
      await executeGraphTool(api, domain, 'add_node', { type: 'ml.activations.relu' }),
    ).toMatch(/^ok: added ml\.activations\.relu/);
    expect([...g.nodes.values()].some((n) => n.type === 'ml.activations.relu')).toBe(true);
    expect(await executeGraphTool(api, domain, 'add_node', { type: 'not.a.type' })).toMatch(
      /^error/,
    );
    expect(
      await executeGraphTool(api, domain, 'add_node', { type: 'ml.activations.relu', id: 'relu1' }),
    ).toMatch(/as "relu1"/);
    expect(g.nodes.has('relu1')).toBe(true);
  });

  it('add_node applies valid initial properties and reports unknown keys', async () => {
    const { api, g } = makeApi();
    const msg = await executeGraphTool(api, domain, 'add_node', {
      type: 'ml.layers.conv2d',
      id: 'c9',
      properties: { outChannels: 8, bogus: 1 },
    });
    expect(msg).toMatch(/^ok: added ml\.layers\.conv2d as "c9"/);
    expect(msg).toMatch(/ignored unknown property bogus/);
    expect(msg).toMatch(/valid keys: .*outChannels/);
    expect(g.nodes.get('c9')!.properties.outChannels).toBe(8);
    expect(g.nodes.get('c9')!.properties.bogus).toBeUndefined();
  });

  it('connect validates and errors on missing node / invalid port', async () => {
    const { api, g } = makeApi();
    addNode(g, createNode('r1', 'ml.activations.relu', { x: 1, y: 0 }, {}));
    expect(
      await executeGraphTool(api, domain, 'connect', {
        sourceId: 'c1',
        sourcePort: 'out',
        targetId: 'r1',
        targetPort: 'in',
      }),
    ).toMatch(/^ok/);
    expect(
      await executeGraphTool(api, domain, 'connect', {
        sourceId: 'c1',
        sourcePort: 'out',
        targetId: 'nope',
        targetPort: 'in',
      }),
    ).toMatch(/^error/);
    expect(
      await executeGraphTool(api, domain, 'connect', {
        sourceId: 'c1',
        sourcePort: 'out',
        targetId: 'r1',
        targetPort: 'bad',
      }),
    ).toMatch(/invalid connection/);
  });

  it('remove_node / remove_edge', async () => {
    const { api, g } = makeApi();
    addNode(g, createNode('r1', 'ml.activations.relu', { x: 1, y: 0 }, {}));
    addEdge(g, createEdge('e1', 'c1', 'out', 'r1', 'in'));
    expect(
      await executeGraphTool(api, domain, 'remove_edge', {
        sourceId: 'c1',
        sourcePort: 'out',
        targetId: 'r1',
        targetPort: 'in',
      }),
    ).toMatch(/^ok/);
    expect(g.edges.length).toBe(0);
    expect(
      await executeGraphTool(api, domain, 'remove_edge', {
        sourceId: 'c1',
        sourcePort: 'out',
        targetId: 'r1',
        targetPort: 'in',
      }),
    ).toMatch(/^error/);
    expect(await executeGraphTool(api, domain, 'remove_node', { nodeId: 'c1' })).toMatch(/^ok/);
    expect(g.nodes.has('c1')).toBe(false);
  });

  it('clear_graph and organize_layout', async () => {
    const { api, g, flags } = makeApi();
    expect(await executeGraphTool(api, domain, 'clear_graph', {})).toMatch(/^ok/);
    expect(g.nodes.size).toBe(0);
    expect(await executeGraphTool(api, domain, 'organize_layout', {})).toMatch(/^ok/);
    expect(flags.organized).toBe(true);
  });

  it('add_block passes the filename through', async () => {
    const { api, flags } = makeApi();
    expect(await executeGraphTool(api, domain, 'add_block', { filename: 'resblock.json' })).toMatch(
      /^ok/,
    );
    expect(flags.blockAdded).toBe('resblock.json');
  });

  it('enter_block / exit_block / save_block on a subgraph.block node', async () => {
    const { api, g, flags } = makeApi();
    const block = createNode('blk1', 'subgraph.block', { x: 2, y: 0 }, { blockName: 'MyBlock' });
    block.subgraph = createGraph('blk1-inner', 'MyBlock');
    addNode(g, block);

    // enter/save require a real subgraph.block; a plain node is rejected.
    expect(await executeGraphTool(api, domain, 'enter_block', { nodeId: 'c1' })).toMatch(/^error/);
    expect(await executeGraphTool(api, domain, 'enter_block', { nodeId: 'blk1' })).toMatch(/^ok/);
    expect(flags.entered).toBe('blk1');

    expect(await executeGraphTool(api, domain, 'exit_block', {})).toMatch(/^ok/);
    expect(flags.exited).toBe(1);

    expect(
      await executeGraphTool(api, domain, 'save_block', { nodeId: 'blk1', name: 'Renamed' }),
    ).toMatch(/^ok/);
    expect(flags.saved).toBe('blk1');
    expect(g.nodes.get('blk1')!.properties.blockName).toBe('Renamed');
    expect(await executeGraphTool(api, domain, 'save_block', { nodeId: 'c1' })).toMatch(/^error/);
  });

  it('get_graph and get_node report state', async () => {
    const { api } = makeApi();
    const summary = await executeGraphTool(api, domain, 'get_graph', {});
    expect(summary).toMatch(/c1 \(ml\.layers\.conv2d\)/);
    const node = await executeGraphTool(api, domain, 'get_node', { nodeId: 'c1' });
    expect(node).toMatch(/c1 \(ml\.layers\.conv2d\)/);
    expect(node).toMatch(/outChannels/);
    expect(await executeGraphTool(api, domain, 'get_node', { nodeId: 'nope' })).toMatch(/^error/);
  });

  it('get_dataset_info summarizes the backend detail payload (no pixel blobs)', async () => {
    const { api, g } = makeApi();
    addNode(g, createNode('ds1', 'data.custom', { x: 0, y: 1 }, { hfId: 'beans' }));

    vi.mocked(callBackend).mockResolvedValueOnce({
      ok: true,
      data: {
        detail: {
          name: 'beans',
          description: 'Custom HuggingFace dataset: beans',
          trainSamples: 1034,
          testSamples: 128,
          diskSize: '—',
          labels: ['angular_leaf_spot', 'bean_rust', 'healthy'],
          channels: 3,
          imageSize: [32, 32],
          sampleImages: { healthy: [] }, // must NOT appear in the summary
        },
      },
    });
    const summary = await executeGraphTool(api, domain, 'get_dataset_info', { nodeId: 'ds1' });
    expect(callBackend).toHaveBeenCalledWith('/dataset-detail', {
      type: 'data.custom',
      id: 'ds1',
      properties: { hfId: 'beans' },
    });
    expect(summary).toMatch(/train samples: 1034, test samples: 128/);
    expect(summary).toMatch(/images: 3×32×32/);
    expect(summary).toMatch(/classes \(3\): angular_leaf_spot, bean_rust, healthy/);
    expect(summary).not.toMatch(/sampleImages/);

    // Backend failure (e.g. bad hfId) surfaces as a tool error the model can act on.
    vi.mocked(callBackend).mockResolvedValueOnce({ ok: false, error: 'Dataset not found' });
    expect(await executeGraphTool(api, domain, 'get_dataset_info', { nodeId: 'ds1' })).toBe(
      'error: Dataset not found',
    );

    // Non-data nodes and unknown ids are rejected without hitting the backend.
    expect(await executeGraphTool(api, domain, 'get_dataset_info', { nodeId: 'c1' })).toMatch(
      /not a data node/,
    );
    expect(await executeGraphTool(api, domain, 'get_dataset_info', { nodeId: 'nope' })).toMatch(
      /^error: no node/,
    );
  });

  it('get_dataset_info reports text/LM datasets with vocab and sample snippets', async () => {
    const { api, g } = makeApi();
    addNode(g, createNode('ds2', 'data.tiny_shakespeare', { x: 0, y: 2 }, {}));
    vi.mocked(callBackend).mockResolvedValueOnce({
      ok: true,
      data: {
        detail: {
          name: 'Tiny Shakespeare',
          description: 'Character-level corpus',
          trainSamples: 1000,
          diskSize: '—',
          labels: [],
          isText: true,
          isLanguageModel: true,
          vocabSize: 65,
          sampleTexts: { Sample: ['First Citizen: Before we proceed any further'] },
        },
      },
    });
    const summary = await executeGraphTool(api, domain, 'get_dataset_info', { nodeId: 'ds2' });
    expect(summary).toMatch(/language-model corpus, vocab size: 65/);
    expect(summary).toMatch(/sample \(Sample\): "First Citizen/);
  });

  it('validate returns a forward/training report', async () => {
    const { api } = makeApi();
    expect(await executeGraphTool(api, domain, 'validate', { mode: 'forward' })).toMatch(/forward/);
    expect(await executeGraphTool(api, domain, 'validate', { mode: 'training' })).toMatch(
      /training/,
    );
  });

  it('validate surfaces per-node shape errors, not just structural ones', async () => {
    const { api } = makeApi();
    // A structurally-fine node carrying a shape error in its lastResult — the
    // kind validateForward/Training are blind to (e.g. 4-D into CrossEntropyLoss).
    const g = api.getCurrentGraph();
    const c1 = g.nodes.get('c1')!;
    c1.lastResult = { outputs: {}, metadata: { error: 'Predictions should be [B, C]' } };
    const report = await executeGraphTool(api, domain, 'validate', { mode: 'forward' });
    expect(report).toMatch(/c1: Predictions should be/);
    expect(report).not.toMatch(/^ok/);
  });

  it('get_training_results reports state and formats the epoch history', async () => {
    const { api } = makeApi();
    // Untrained → says so.
    expect(await executeGraphTool(api, domain, 'get_training_results', {})).toMatch(
      /not been trained/,
    );

    // With history → a per-epoch table picking columns from the metrics present.
    api.trainingProgress = [
      { epoch: 1, loss: 0.9, accuracy: 0.6, valLoss: 1.0, valAccuracy: 0.55 },
      { epoch: 2, loss: 0.4, accuracy: 0.85, valLoss: 0.7, valAccuracy: 0.8 },
    ];
    api.modelTrained = true;
    const report = await executeGraphTool(api, domain, 'get_training_results', {});
    expect(report).toMatch(/2 epoch\(s\)/);
    expect(report).toMatch(/epoch 2: loss=0\.4000, acc=85\.0%, valLoss=0\.7000, valAcc=80\.0%/);

    // Stale flag surfaces a warning.
    api.modelStale = true;
    expect(await executeGraphTool(api, domain, 'get_training_results', {})).toMatch(
      /stale|no longer reflect/i,
    );
  });

  it('reports unknown tools', async () => {
    const { api } = makeApi();
    expect(await executeGraphTool(api, domain, 'frobnicate', {})).toMatch(/unknown tool/);
  });
});
