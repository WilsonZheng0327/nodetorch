// Tests for the agent graph-tool executor (src/ui/chat/graphTools.ts).
// Uses a fake GraphToolApi backed by a real Graph so we exercise the tool
// dispatch, validation, and result strings without the full useGraph hook.

import { describe, it, expect, vi } from 'vitest';

import { initDomain } from '../../../src/domain';
import { createGraph, createNode, addNode, createEdge, addEdge } from '../../../src/core/graph';
import { executeGraphTool, type GraphToolApi } from '../../../src/ui/chat/graphTools';
import { callBackend, getBackend } from '../../../src/api/client';

// get_dataset_info / get_saved_runs call the backend; stub it so tests stay offline.
vi.mock('../../../src/api/client', () => ({ callBackend: vi.fn(), getBackend: vi.fn() }));

const domain = initDomain();

function makeApi() {
  const g = createGraph('main', 'Main');
  addNode(g, createNode('c1', 'ml.layers.conv2d', { x: 0, y: 0 }, { outChannels: 32 }));
  const flags = {
    organized: false,
    blockAdded: '',
    entered: '',
    exited: 0,
    saved: '',
    trainStarted: 0,
    trainCancelled: 0,
    tested: 0,
  };

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
    // fake validation: reject any connection into the sentinel node 'r2'
    isValidConnection: (c) => c.target !== 'r2',
    trainingProgress: [],
    modelTrained: false,
    modelStale: false,
    trainingActive: false,
    batchProgress: null,
    testResult: null,
    testing: false,
    runTrain: async () => {
      flags.trainStarted += 1;
    },
    cancelTrain: () => {
      flags.trainCancelled += 1;
    },
    runTest: async () => {
      flags.tested += 1;
      return {
        result: {
          testLoss: 0.4,
          testAccuracy: 0.88,
          testSamples: 100,
          perClassAccuracy: [{ cls: 0, name: 'cat', accuracy: 0.9, count: 50 }],
        },
      };
    },
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
    // Nonexistent ports name the node's real ports.
    expect(
      await executeGraphTool(api, domain, 'connect', {
        sourceId: 'c1',
        sourcePort: 'out',
        targetId: 'r1',
        targetPort: 'bad',
      }),
    ).toMatch(/"r1" has no input port "bad" — its inputs: in/);
    expect(
      await executeGraphTool(api, domain, 'connect', {
        sourceId: 'c1',
        sourcePort: 'bogus',
        targetId: 'r1',
        targetPort: 'in',
      }),
    ).toMatch(/"c1" has no output port "bogus" — its outputs: out/);
    // Structurally fine but rejected by the validity rules (r2 is the sentinel).
    addNode(g, createNode('r2', 'ml.activations.relu', { x: 2, y: 0 }, {}));
    expect(
      await executeGraphTool(api, domain, 'connect', {
        sourceId: 'c1',
        sourcePort: 'out',
        targetId: 'r2',
        targetPort: 'in',
      }),
    ).toMatch(/invalid connection/);
  });

  it('connect splits "nodeId.portId" ids and flags occupied single-input ports', async () => {
    const { api, g } = makeApi();
    addNode(g, createNode('r1', 'ml.activations.relu', { x: 1, y: 0 }, {}));
    // The "c1.out" concatenation habit is auto-resolved, with a corrective note.
    expect(
      await executeGraphTool(api, domain, 'connect', {
        sourceId: 'c1.out',
        targetId: 'r1.in',
      }),
    ).toMatch(/^ok: connected c1\.out -> r1\.in \(note: .*separately/);

    // A single-input port that's taken points at the exact remove_edge call.
    addEdge(g, createEdge('e1', 'c1', 'out', 'r1', 'in'));
    expect(
      await executeGraphTool(api, domain, 'connect', {
        sourceId: 'c1',
        sourcePort: 'out',
        targetId: 'r1',
        targetPort: 'in',
      }),
    ).toMatch(/already connected \(from c1\.out\).*remove_edge\(c1, out, r1, in\)/);
  });

  it('connect warns when loss labels come from a computed tensor', async () => {
    const { api, g } = makeApi();
    addNode(g, createNode('ce', 'ml.loss.cross_entropy', { x: 2, y: 0 }, {}));
    const bad = await executeGraphTool(api, domain, 'connect', {
      sourceId: 'c1',
      sourcePort: 'out',
      targetId: 'ce',
      targetPort: 'labels',
    });
    expect(bad).toMatch(/^ok/);
    expect(bad).toMatch(/WARNING: loss labels normally come straight from a data node/);

    // Straight from a data node → no warning.
    addNode(g, createNode('d', 'data.mnist', { x: 0, y: 1 }, {}));
    const good = await executeGraphTool(api, domain, 'connect', {
      sourceId: 'd',
      sourcePort: 'labels',
      targetId: 'ce',
      targetPort: 'labels',
    });
    expect(good).toMatch(/^ok/);
    expect(good).not.toMatch(/WARNING/);
  });

  it('connect auto-orients a reversed optimizer edge (the sink has no outputs)', async () => {
    const { api, g } = makeApi();
    addNode(g, createNode('ce', 'ml.loss.cross_entropy', { x: 2, y: 0 }, {}));
    addNode(g, createNode('opt', 'ml.optimizers.adam', { x: 3, y: 0 }, {}));

    // The exact thrash from the field: model wires optimizer -> loss with a
    // guessed "in" port. The optimizer is a pure sink (no output ports), so the
    // direction is unambiguously backwards — flip it to loss.out -> opt.loss and
    // recover the single output port the model mis-named as "in".
    const res = await executeGraphTool(api, domain, 'connect', {
      sourceId: 'opt',
      sourcePort: 'loss',
      targetId: 'ce',
      targetPort: 'in',
    });
    expect(res).toMatch(/^ok: connected ce\.out -> opt\.loss/);
    expect(res).toMatch(/flipped direction/);

    // Same recovery when the model passes ids as "opt.loss" / "ce.out".
    const res2 = await executeGraphTool(api, domain, 'connect', {
      sourceId: 'opt.loss',
      targetId: 'ce.out',
    });
    expect(res2).toMatch(/^ok: connected ce\.out -> opt\.loss/);
  });

  it('connect does NOT flip a normal edge and still errors on a bad port', async () => {
    const { api, g } = makeApi();
    addNode(g, createNode('r1', 'ml.activations.relu', { x: 1, y: 0 }, {}));
    // relu has both in and out, so a wrong target port is a real mistake, not a
    // reversed edge — it must still error (no silent single-port "fix").
    const res = await executeGraphTool(api, domain, 'connect', {
      sourceId: 'c1',
      sourcePort: 'out',
      targetId: 'r1',
      targetPort: 'bad',
    });
    expect(res).toMatch(/"r1" has no input port "bad"/);
  });

  it('add_node reports when the requested id is already taken', async () => {
    const { api } = makeApi();
    const msg = await executeGraphTool(api, domain, 'add_node', {
      type: 'ml.activations.relu',
      id: 'c1', // taken by the conv in makeApi
    });
    expect(msg).toMatch(/requested id "c1" was already taken; use "ml\.activations\.relu-new"/);
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

  // Replace the lone conv with a structurally-valid training graph:
  // mnist → linear → cross_entropy ← labels, loss → adam.
  function buildTrainableGraph(g: ReturnType<typeof createGraph>) {
    g.nodes.delete('c1');
    addNode(g, createNode('d', 'data.mnist', { x: 0, y: 0 }, {}));
    addNode(g, createNode('lin', 'ml.layers.linear', { x: 1, y: 0 }, {}));
    addNode(g, createNode('ce', 'ml.loss.cross_entropy', { x: 2, y: 0 }, {}));
    addNode(g, createNode('opt', 'ml.optimizers.adam', { x: 3, y: 0 }, {}));
    addEdge(g, createEdge('t1', 'd', 'out', 'lin', 'in'));
    addEdge(g, createEdge('t2', 'lin', 'out', 'ce', 'predictions'));
    addEdge(g, createEdge('t3', 'd', 'labels', 'ce', 'labels'));
    addEdge(g, createEdge('t4', 'ce', 'out', 'opt', 'loss'));
  }

  it('start_training refuses an untrainable graph and reports the problems', async () => {
    const { api, flags } = makeApi(); // just a dangling conv — not trainable
    const msg = await executeGraphTool(api, domain, 'start_training', {});
    expect(msg).toMatch(/^error: the graph is not ready to train/);
    expect(flags.trainStarted).toBe(0);
  });

  it('start_training starts a valid graph, and refuses while one is running', async () => {
    const { api, g, flags } = makeApi();
    buildTrainableGraph(g);
    expect(await executeGraphTool(api, domain, 'start_training', {})).toMatch(
      /^ok: training started/,
    );
    expect(flags.trainStarted).toBe(1);

    api.trainingActive = true;
    expect(await executeGraphTool(api, domain, 'start_training', {})).toMatch(
      /already in progress/,
    );
    expect(flags.trainStarted).toBe(1);
  });

  it('stop_training cancels only when a run is active', async () => {
    const { api, flags } = makeApi();
    expect(await executeGraphTool(api, domain, 'stop_training', {})).toMatch(
      /^error: no training run/,
    );
    expect(flags.trainCancelled).toBe(0);

    api.trainingActive = true;
    expect(await executeGraphTool(api, domain, 'stop_training', {})).toMatch(/^ok: cancel/);
    expect(flags.trainCancelled).toBe(1);
  });

  it('get_training_status reports idle, running, and stale states', async () => {
    const { api } = makeApi();
    let s = await executeGraphTool(api, domain, 'get_training_status', {});
    expect(s).toMatch(/no training is running/);
    expect(s).toMatch(/no trained model in memory/);

    api.trainingActive = true;
    api.trainingProgress = [{ epoch: 1, totalEpochs: 5, loss: 0.8, accuracy: 0.7 }];
    api.batchProgress = { batch: 12, totalBatches: 400 };
    s = await executeGraphTool(api, domain, 'get_training_status', {});
    expect(s).toMatch(/RUNNING/);
    expect(s).toMatch(/epochs completed: 1 of 5/);
    expect(s).toMatch(/batch 12\/400/);
    expect(s).toMatch(/loss=0\.8000, acc=70\.0%/);

    api.trainingActive = false;
    api.modelTrained = true;
    api.modelStale = true;
    s = await executeGraphTool(api, domain, 'get_training_status', {});
    expect(s).toMatch(/STALE/);
    expect(s).toMatch(/last run ended at epoch 1/);
  });

  it('get_epoch_detail reports gradients, per-class accuracy, tracked samples, text', async () => {
    const { api } = makeApi();
    expect(await executeGraphTool(api, domain, 'get_epoch_detail', {})).toMatch(
      /^error: no training history/,
    );

    api.trainingProgress = [
      { epoch: 1, loss: 1.2, accuracy: 0.5 },
      {
        epoch: 2,
        totalEpochs: 2,
        loss: 0.4,
        accuracy: 0.9,
        time: 3.2,
        samples: 1000,
        gradientFlow: [
          { name: 'conv1', norm: 0.5 },
          { name: 'fc', norm: 0.001 },
        ],
        perClassAccuracy: [
          { cls: 0, accuracy: 0.95 },
          { cls: 1, accuracy: 0.6 },
        ],
        classNames: ['cat', 'dog'],
        trackedSamples: [
          { idx: 7, label: 1, predictedClass: 0, confidence: 0.55, loss: 1.9 },
        ],
        generatedText: 'To be or not to be',
      },
    ];
    // Defaults to the latest epoch.
    const detail = await executeGraphTool(api, domain, 'get_epoch_detail', {});
    expect(detail).toMatch(/Epoch 2 of 2: loss=0\.4000, acc=90\.0%/);
    expect(detail).toMatch(/- conv1: 0\.5000/);
    expect(detail).toMatch(/- fc: 0\.0010/);
    expect(detail).toMatch(/- dog: 60\.0%/); // class names resolved, worst first
    expect(detail).toMatch(/#7: true=dog pred=cat \(55\.0%\) loss=1\.9000 ✗/);
    expect(detail).toMatch(/To be or not to be/);

    expect(await executeGraphTool(api, domain, 'get_epoch_detail', { epoch: 1 })).toMatch(
      /Epoch 1/,
    );
    expect(await executeGraphTool(api, domain, 'get_epoch_detail', { epoch: 9 })).toMatch(
      /^error: no epoch 9/,
    );
  });

  it('get_test_results summarizes the test tab including top confusions', async () => {
    const { api } = makeApi();
    expect(await executeGraphTool(api, domain, 'get_test_results', {})).toMatch(
      /no test results/,
    );

    api.testResult = {
      testLoss: 0.31,
      testAccuracy: 0.91,
      testSamples: 10000,
      perClassAccuracy: [
        { cls: 0, name: 'cat', accuracy: 0.97, count: 5000 },
        { cls: 1, name: 'dog', accuracy: 0.85, count: 5000 },
      ],
      confusionMatrix: {
        size: 2,
        data: [
          [4850, 150],
          [750, 4250],
        ],
        classNames: ['cat', 'dog'],
      },
    };
    const s = await executeGraphTool(api, domain, 'get_test_results', {});
    expect(s).toMatch(/10000 samples — accuracy 91\.0%, loss 0\.3100/);
    expect(s).toMatch(/- dog: 85\.0% \(5000 samples\)/);
    expect(s).toMatch(/true dog predicted as cat: 750×/);
  });

  it('get_saved_runs lists runs and details one by id', async () => {
    const run = {
      id: 'run-1',
      timestamp: '2026-07-12 10:00',
      datasetType: 'data.mnist',
      epochs: 5,
      learningRate: 0.001,
      optimizer: 'adam',
      scheduler: 'none',
      finalLoss: 0.2,
      finalAccuracy: 0.93,
      bestValAccuracy: 0.92,
      duration: 42,
      totalParams: 101770,
      nodeCount: 5,
    };
    vi.mocked(getBackend).mockResolvedValueOnce({ ok: true, data: { runs: [run] } });
    const list = await executeGraphTool(makeApi().api, domain, 'get_saved_runs', {});
    expect(getBackend).toHaveBeenCalledWith('/runs');
    expect(list).toMatch(/run-1 .* 5 epochs, adam lr=0\.001/);
    expect(list).toMatch(/finalAcc=93\.0%/);

    vi.mocked(getBackend).mockResolvedValueOnce({
      ok: true,
      data: {
        run: {
          ...run,
          batchSize: 64,
          seed: 42,
          valSplit: 0.1,
          epochHistory: [
            { epoch: 1, loss: 0.9, accuracy: 0.6 },
            { epoch: 2, loss: 0.5, accuracy: 0.8 },
          ],
        },
      },
    });
    const detail = await executeGraphTool(makeApi().api, domain, 'get_saved_runs', {
      id: 'run-1',
    });
    expect(getBackend).toHaveBeenLastCalledWith('/runs/run-1');
    expect(detail).toMatch(/batchSize=64/);
    expect(detail).toMatch(/- epoch 2: loss=0\.5000, acc=80\.0%/);

    vi.mocked(getBackend).mockResolvedValueOnce({ ok: false, error: 'Run not found' });
    expect(
      await executeGraphTool(makeApi().api, domain, 'get_saved_runs', { id: 'nope' }),
    ).toBe('error: Run not found');
  });

  it('run_test guards on model state, then runs and reports the fresh results', async () => {
    const { api, flags } = makeApi();
    expect(await executeGraphTool(api, domain, 'run_test', {})).toMatch(
      /^error: no trained model/,
    );

    api.modelTrained = true;
    api.modelStale = true;
    expect(await executeGraphTool(api, domain, 'run_test', {})).toMatch(/STALE/);

    api.modelStale = false;
    api.trainingActive = true;
    expect(await executeGraphTool(api, domain, 'run_test', {})).toMatch(/training is in progress/);

    api.trainingActive = false;
    expect(flags.tested).toBe(0); // none of the refusals ran the evaluation
    const msg = await executeGraphTool(api, domain, 'run_test', {});
    expect(flags.tested).toBe(1);
    expect(msg).toMatch(/^ok: test evaluation complete/);
    expect(msg).toMatch(/100 samples — accuracy 88\.0%, loss 0\.4000/);

    // Backend failure surfaces as a tool error.
    api.runTest = async () => ({ error: 'boom' });
    expect(await executeGraphTool(api, domain, 'run_test', {})).toBe('error: boom');
  });

  it('get_system_info formats the raw (unenveloped) /system-info payload', async () => {
    const { api } = makeApi();
    const fetchMock = vi.fn(async () => ({
      json: async () => ({
        python: '3.11.9',
        pytorch: '2.4.0',
        cudaAvailable: true,
        gpuCount: 1,
        gpus: [{ name: 'RTX 3080', vram: 10, computeCapability: '8.6' }],
        currentDevice: 'cuda',
      }),
    }));
    vi.stubGlobal('fetch', fetchMock);
    try {
      const s = await executeGraphTool(api, domain, 'get_system_info', {});
      expect(fetchMock.mock.calls[0][0]).toMatch(/\/system-info$/);
      expect(s).toMatch(/Python 3\.11\.9, PyTorch 2\.4\.0/);
      expect(s).toMatch(/- RTX 3080: 10 GB VRAM, compute capability 8\.6/);
      expect(s).toMatch(/current training device: cuda/);

      fetchMock.mockRejectedValueOnce(new Error('down'));
      expect(await executeGraphTool(api, domain, 'get_system_info', {})).toMatch(
        /^error: Cannot connect/,
      );
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it('reports unknown tools', async () => {
    const { api } = makeApi();
    expect(await executeGraphTool(api, domain, 'frobnicate', {})).toMatch(/unknown tool/);
  });
});
