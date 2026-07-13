// Executes an agent tool call against the live graph via useGraph actions —
// the same functions the inspector/palette use, so edits are validated, render
// live, and are undoable. Returns a human-readable observation string the model
// reads back (errors included, so it can self-correct).

import type * as RF from '@xyflow/react';
import type { Graph, NodeInstance } from '../../core/graph';
import type { NodeRegistry } from '../../core/nodedef';
import { validateForward, validateTraining } from '../../core/validation';
import { callBackend, getBackend } from '../../api/client';
import { apiUrl } from '../../api/base';
import type { DatasetInfo } from '../inspector/DatasetDetail';
import type { EpochData, FullRun, SavedRun, SystemInfo, TestResult } from '../dashboard/types';

/** The subset of useGraph the tool executor needs. */
export interface GraphToolApi {
  /** Live current graph (root or active subgraph) — read fresh each call. */
  getCurrentGraph: () => Graph;
  /** Per-epoch training metrics from the most recent run this session. */
  trainingProgress: EpochData[];
  /** Whether a trained model is in memory, and whether the graph has since changed. */
  modelTrained: boolean;
  modelStale: boolean;
  /** True from the moment training starts until it finishes/cancels/errors. */
  trainingActive: boolean;
  /** Mid-epoch batch progress while training, else null. */
  batchProgress: { batch: number; totalBatches: number } | null;
  /** Held-out test-set evaluation (the dashboard's Test tab), if the user ran Test. */
  testResult: TestResult | null;
  /** True while a test evaluation is in flight. */
  testing: boolean;
  /** Start a training run (the Train-button path). Resolves when training ENDS. */
  runTrain: () => Promise<void>;
  /** Ask the in-flight training run to cancel (stops after the current epoch). */
  cancelTrain: () => void;
  /** Evaluate the trained model on the held-out test set (the Test-button path). */
  runTest: () => Promise<{ result: TestResult } | { error: string }>;
  updateProperty: (nodeId: string, key: string, value: unknown) => Promise<void>;
  addNode: (
    type: string,
    position: { x: number; y: number },
    requestedId?: string,
  ) => Promise<string | undefined>;
  connect: (connection: RF.Connection) => Promise<void>;
  removeNode: (nodeId: string) => Promise<void>;
  removeEdge: (
    sourceId: string,
    sourcePort: string,
    targetId: string,
    targetPort: string,
  ) => Promise<boolean>;
  clearGraph: () => void;
  organizeGraph: () => void;
  addBlockFromTemplate: (filename: string, position: { x: number; y: number }) => Promise<void>;
  /** Descend into a subgraph.block node so further edits target its inner graph. */
  enterSubgraph: (nodeId: string) => void;
  /** Pop up one level out of the current subgraph. */
  exitSubgraph: () => void;
  /** Save a subgraph.block node to the reusable-block library. */
  saveBlock: (nodeId: string) => Promise<void>;
  isValidConnection: (connection: RF.Connection) => boolean;
}

type Args = Record<string, unknown>;
type Domain = { nodeRegistry: NodeRegistry };

/** Place a new node to the right of the rightmost existing node. */
function nextPosition(g: Graph): { x: number; y: number } {
  let maxX = -Infinity;
  for (const n of g.nodes.values()) maxX = Math.max(maxX, n.position.x);
  return { x: maxX === -Infinity ? 120 : maxX + 220, y: 140 };
}

function summarizeGraph(g: Graph): string {
  const nodes = [...g.nodes.values()];
  if (!nodes.length) return '(the canvas is empty)';
  const lines: string[] = [`Nodes (${nodes.length}):`];
  for (const n of nodes) {
    const props = Object.entries(n.properties)
      .filter(([, v]) => v !== undefined && v !== '')
      .map(([k, v]) => `${k}=${v}`)
      .join(', ');
    lines.push(`- ${n.id} (${n.type})${props ? '  ' + props : ''}`);
  }
  if (g.edges.length) {
    lines.push(`Edges (${g.edges.length}):`);
    for (const e of g.edges)
      lines.push(
        `- ${e.source.nodeId}.${e.source.portId} -> ${e.target.nodeId}.${e.target.portId}`,
      );
  }
  return lines.join('\n');
}

function describeNode(n: NodeInstance): string {
  const md = n.lastResult?.metadata as Record<string, unknown> | undefined;
  const props = Object.entries(n.properties)
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(', ');
  const parts = [`${n.id} (${n.type})`, `props: ${props || '(none)'}`];
  if (md?.outputShape) parts.push(`outputShape: [${(md.outputShape as unknown[]).join(', ')}]`);
  if (md?.paramCount != null) parts.push(`params: ${md.paramCount}`);
  // Final training metrics land on the optimizer node after a run (see useGraph).
  if (typeof md?.finalLoss === 'number') parts.push(`finalLoss: ${md.finalLoss.toFixed(4)}`);
  if (typeof md?.finalAccuracy === 'number')
    parts.push(`finalAccuracy: ${(md.finalAccuracy * 100).toFixed(1)}%`);
  if (md?.error) parts.push(`ERROR: ${md.error}`);
  return parts.join('\n');
}

/** Format a metric: numbers to 4dp, accuracies as percentages, missing as "—". */
function fmtMetric(v: unknown, pct = false): string {
  if (typeof v !== 'number' || Number.isNaN(v)) return '—';
  return pct ? `${(v * 100).toFixed(1)}%` : v.toFixed(4);
}

/** The full per-epoch training history (loss/accuracy/val/perplexity curves).
 *  This lives in UI state, not the graph, so it's invisible to the other tools —
 *  this is the agent's only window onto how training actually went. */
function getTrainingResults(graph: GraphToolApi): string {
  const epochs = graph.trainingProgress;
  if (!epochs.length) {
    return graph.modelTrained
      ? 'a trained model is in memory, but no per-epoch history is available this session (it may have been loaded from a saved run)'
      : 'no training results yet — the model has not been trained this session';
  }

  // Pick columns from what the run actually produced (classification vs LM differ).
  const has = (k: keyof EpochData) => epochs.some((e) => typeof e[k] === 'number');
  const cols: { key: keyof EpochData; label: string; pct?: boolean }[] = [
    { key: 'loss', label: 'loss' },
  ];
  if (has('accuracy')) cols.push({ key: 'accuracy', label: 'acc', pct: true });
  if (has('valLoss')) cols.push({ key: 'valLoss', label: 'valLoss' });
  if (has('valAccuracy')) cols.push({ key: 'valAccuracy', label: 'valAcc', pct: true });
  if (has('perplexity')) cols.push({ key: 'perplexity', label: 'ppl' });
  if (has('valPerplexity')) cols.push({ key: 'valPerplexity', label: 'valPpl' });

  const row = (e: EpochData) =>
    `epoch ${e.epoch}: ` + cols.map((c) => `${c.label}=${fmtMetric(e[c.key], c.pct)}`).join(', ');

  // Cap output so a long run can't flood the context: show all if small, else
  // the first and last 15 epochs (enough to read the trend and the final state).
  const CAP = 30;
  let lines: string[];
  if (epochs.length <= CAP) {
    lines = epochs.map(row);
  } else {
    lines = [
      ...epochs.slice(0, 15).map(row),
      `… (${epochs.length - 30} epochs omitted) …`,
      ...epochs.slice(-15).map(row),
    ];
  }

  const stale = graph.modelStale
    ? '\n(NOTE: the graph was edited after training — these results no longer reflect the current graph)'
    : '';
  return `Training results — ${epochs.length} epoch(s):\n${lines.join('\n')}${stale}`;
}

/** One epoch's metrics on a single line, including whichever of the optional
 *  (val/LM/GAN) numbers this run actually produced. */
function epochHeadline(e: EpochData): string {
  const parts = [`loss=${fmtMetric(e.loss)}`];
  const optional: [string, number | null | undefined, boolean?][] = [
    ['acc', e.accuracy, true],
    ['valLoss', e.valLoss],
    ['valAcc', e.valAccuracy, true],
    ['ppl', e.perplexity],
    ['valPpl', e.valPerplexity],
    ['dLoss', e.dLoss],
    ['gLoss', e.gLoss],
    ['lr', e.learningRate],
  ];
  for (const [label, v, pct] of optional) {
    if (typeof v === 'number' && !Number.isNaN(v)) parts.push(`${label}=${fmtMetric(v, pct)}`);
  }
  return parts.join(', ');
}

/** Live view: is training running now, how far along, latest metrics. Reads UI
 *  state that updates per epoch/batch, so it works mid-run. */
function getTrainingStatus(graph: GraphToolApi): string {
  const epochs = graph.trainingProgress;
  const last = epochs[epochs.length - 1];
  if (graph.trainingActive) {
    const lines = ['training is RUNNING'];
    lines.push(`epochs completed: ${epochs.length}${last?.totalEpochs ? ` of ${last.totalEpochs}` : ''}`);
    if (graph.batchProgress) {
      lines.push(
        `current epoch: batch ${graph.batchProgress.batch}/${graph.batchProgress.totalBatches}`,
      );
    }
    if (last) lines.push(`latest (epoch ${last.epoch}): ${epochHeadline(last)}`);
    else lines.push('no epoch has completed yet');
    return lines.join('\n');
  }
  const lines = ['no training is running'];
  if (graph.modelTrained) {
    lines.push(
      graph.modelStale
        ? 'a trained model is in memory but STALE — the graph changed after training (retrain before inference/testing)'
        : 'a trained model is in memory and matches the current graph',
    );
  } else {
    lines.push('no trained model in memory yet');
  }
  if (last) lines.push(`last run ended at epoch ${last.epoch}: ${epochHeadline(last)}`);
  return lines.join('\n');
}

/** Deep-dive one epoch's record: gradient flow, per-class accuracy, tracked
 *  samples, generated text — the dashboard's drill-in data, as text. */
function getEpochDetail(graph: GraphToolApi, args: Args): string {
  const epochs = graph.trainingProgress;
  if (!epochs.length) return 'error: no training history this session — train first';
  const wanted = args.epoch == null ? undefined : Number(args.epoch);
  const e = wanted == null ? epochs[epochs.length - 1] : epochs.find((r) => r.epoch === wanted);
  if (!e) {
    return `error: no epoch ${wanted} in this run (have epochs ${epochs[0].epoch}–${epochs[epochs.length - 1].epoch})`;
  }

  const className = (c: number | null | undefined) =>
    c == null ? '?' : (e.classNames?.[c] ?? String(c));
  const lines = [
    `Epoch ${e.epoch}${e.totalEpochs ? ` of ${e.totalEpochs}` : ''}: ${epochHeadline(e)}`,
  ];
  if (typeof e.time === 'number') {
    lines.push(
      `time: ${e.time.toFixed(1)}s${e.samples ? `, ${e.samples} samples` : ''}${e.batches ? `, ${e.batches} batches` : ''}`,
    );
  }

  if (e.gradientFlow?.length) {
    const sorted = [...e.gradientFlow].sort((a, b) => b.norm - a.norm);
    const truncated = sorted.length > 20;
    const shown = truncated ? [...sorted.slice(0, 10), ...sorted.slice(-5)] : sorted;
    lines.push(
      `gradient norms by layer, largest first${truncated ? ' (10 largest + 5 smallest of ' + sorted.length + ')' : ''}:`,
    );
    for (const gf of shown) lines.push(`- ${gf.name}: ${fmtMetric(gf.norm)}`);
  }

  if (e.perClassAccuracy?.length) {
    const sorted = [...e.perClassAccuracy].sort((a, b) => a.accuracy - b.accuracy);
    const truncated = sorted.length > 20;
    const shown = truncated ? [...sorted.slice(0, 5), ...sorted.slice(-5)] : sorted;
    lines.push(`per-class accuracy, worst first${truncated ? ' (5 worst + 5 best)' : ''}:`);
    for (const c of shown) lines.push(`- ${className(c.cls)}: ${fmtMetric(c.accuracy, true)}`);
  }

  if (e.trackedSamples?.length) {
    lines.push(`tracked samples (${e.trackedSamples.length}):`);
    for (const t of e.trackedSamples.slice(0, 12)) {
      const verdict =
        t.predictedClass != null && t.label != null
          ? t.predictedClass === t.label
            ? ' ✓'
            : ' ✗'
          : '';
      const conf = typeof t.confidence === 'number' ? ` (${fmtMetric(t.confidence, true)})` : '';
      const loss = typeof t.loss === 'number' ? ` loss=${fmtMetric(t.loss)}` : '';
      lines.push(
        `- #${t.idx}: true=${className(t.label)} pred=${className(t.predictedClass)}${conf}${loss}${verdict}`,
      );
    }
  }

  if (e.generatedText) {
    const text = e.generatedText;
    lines.push(`generated text sample:\n"${text.slice(0, 300)}${text.length > 300 ? '…' : ''}"`);
  }
  if (e.generatedSamples?.length) {
    lines.push(
      `generated ${e.generatedSamples.length} sample image(s) — visible in the dashboard (pixels not shown here)`,
    );
  }
  return lines.join('\n');
}

/** The dashboard's Test tab as text: overall metrics, per-class accuracy, and
 *  the biggest off-diagonal confusions (matrix itself would flood the context). */
function getTestResults(graph: GraphToolApi): string {
  const t = graph.testResult;
  if (!t) {
    return 'no test results yet — run the run_test tool (or the Test button) to evaluate the trained model on the held-out test set (classification models only)';
  }
  return summarizeTestResult(t);
}

function summarizeTestResult(t: TestResult): string {
  const lines = [
    `Test set: ${t.testSamples} samples — accuracy ${fmtMetric(t.testAccuracy, true)}, loss ${fmtMetric(t.testLoss)}`,
  ];
  if (t.perClassAccuracy?.length) {
    const sorted = [...t.perClassAccuracy].sort((a, b) => a.accuracy - b.accuracy);
    const truncated = sorted.length > 20;
    const shown = truncated ? [...sorted.slice(0, 5), ...sorted.slice(-5)] : sorted;
    lines.push(`per-class accuracy, worst first${truncated ? ' (5 worst + 5 best)' : ''}:`);
    for (const c of shown) {
      lines.push(`- ${c.name}: ${fmtMetric(c.accuracy, true)} (${c.count} samples)`);
    }
  }
  const cm = t.confusionMatrix;
  if (cm?.data?.length) {
    const name = (i: number) =>
      cm.classNames?.[i] ?? t.perClassAccuracy.find((c) => c.cls === i)?.name ?? String(i);
    const confusions: { from: number; to: number; n: number }[] = [];
    for (let i = 0; i < cm.size; i++) {
      for (let j = 0; j < cm.size; j++) {
        if (i !== j && cm.data[i]?.[j] > 0) confusions.push({ from: i, to: j, n: cm.data[i][j] });
      }
    }
    confusions.sort((a, b) => b.n - a.n);
    if (confusions.length) {
      lines.push('top confusions:');
      for (const c of confusions.slice(0, 8)) {
        lines.push(`- true ${name(c.from)} predicted as ${name(c.to)}: ${c.n}×`);
      }
    }
  }
  return lines.join('\n');
}

/** The dashboard's Runs tab: the saved-runs list, or one run in detail. */
async function getSavedRuns(args: Args): Promise<string> {
  const id = typeof args.id === 'string' && args.id ? args.id : undefined;
  if (!id) {
    const res = await getBackend<{ runs: SavedRun[] }>('/runs');
    if (!res.ok) return `error: ${res.error}`;
    const runs = res.data.runs ?? [];
    if (!runs.length) return 'no saved runs yet — completed training runs are saved automatically';
    const lines = [`Saved runs (${runs.length}):`];
    for (const r of runs.slice(0, 20)) {
      lines.push(
        `- ${r.id} | ${r.timestamp} | ${r.datasetType} | ${r.epochs} epochs, ${r.optimizer} lr=${r.learningRate} | finalLoss=${fmtMetric(r.finalLoss)}, finalAcc=${fmtMetric(r.finalAccuracy, true)}, bestValAcc=${fmtMetric(r.bestValAccuracy, true)}`,
      );
    }
    if (runs.length > 20) lines.push(`… and ${runs.length - 20} more`);
    return lines.join('\n');
  }

  const res = await getBackend<{ run: FullRun }>(`/runs/${encodeURIComponent(id)}`);
  if (!res.ok) return `error: ${res.error}`;
  const r = res.data.run;
  const lines = [
    `Run ${r.id} (${r.timestamp})`,
    `dataset: ${r.datasetType} | ${r.epochs} epochs | ${r.optimizer} lr=${r.learningRate} | scheduler: ${r.scheduler} | batchSize=${r.batchSize}, valSplit=${r.valSplit}, seed=${r.seed}`,
    `model: ${r.nodeCount} nodes, ${r.totalParams} params | duration: ${r.duration.toFixed(1)}s`,
    `final: loss=${fmtMetric(r.finalLoss)}, acc=${fmtMetric(r.finalAccuracy, true)}, bestValAcc=${fmtMetric(r.bestValAccuracy, true)}`,
  ];
  const history = r.epochHistory ?? [];
  if (history.length) {
    const truncated = history.length > 12;
    const shown = truncated ? [...history.slice(0, 5), ...history.slice(-5)] : history;
    lines.push(`epoch history${truncated ? ` (first 5 + last 5 of ${history.length})` : ''}:`);
    for (const e of shown) lines.push(`- epoch ${e.epoch}: ${epochHeadline(e)}`);
  }
  return lines.join('\n');
}

/** Start a run the same way the Train button does — but only if pre-flight
 *  validation passes, so the model gets the problems instead of a dead start. */
function startTraining(graph: GraphToolApi, domain: Domain): string {
  if (graph.trainingActive) return 'error: a training run is already in progress';
  const report = validate(graph, domain, { mode: 'training' });
  if (!report.startsWith('ok')) {
    return `error: the graph is not ready to train — fix these first:\n${report}`;
  }
  // Fire and forget: runTrain's promise resolves when training ENDS (minutes
  // away); the tool result must return now so the turn can continue.
  void graph.runTrain();
  return 'ok: training started — it runs in the background and the user sees live progress in the dashboard. Use get_training_status to check on it later; do not poll in a loop.';
}

function stopTraining(graph: GraphToolApi): string {
  if (!graph.trainingActive) return 'error: no training run is in progress';
  graph.cancelTrain();
  return 'ok: cancel requested — training stops after the current epoch';
}

/** Evaluate on the held-out test set (the Test-button path) and report the
 *  fresh results — runTest returns them precisely because our captured
 *  graph.testResult is a render-time snapshot. */
async function runTestSet(graph: GraphToolApi): Promise<string> {
  if (graph.trainingActive) {
    return 'error: training is in progress — wait for it to finish before testing';
  }
  if (graph.testing) return 'error: a test evaluation is already running';
  if (!graph.modelTrained) return 'error: no trained model in memory — train first';
  if (graph.modelStale) {
    return 'error: the trained model is STALE — the graph changed after training; retrain first';
  }
  const res = await graph.runTest();
  if ('error' in res) return `error: ${res.error}`;
  return `ok: test evaluation complete\n${summarizeTestResult(res.result)}`;
}

/** GET /system-info — NOTE: this endpoint returns the raw info dict with no
 *  {status} envelope (the dashboard consumes it as-is), hence plain fetch. */
async function getSystemInfo(): Promise<string> {
  let info: SystemInfo;
  try {
    const r = await fetch(apiUrl('/system-info'));
    info = (await r.json()) as SystemInfo;
  } catch {
    return 'error: Cannot connect to backend — is the server running?';
  }
  const lines = [`Python ${info.python}, PyTorch ${info.pytorch}`];
  if (info.cudaAvailable && info.gpus?.length) {
    lines.push(`CUDA available — ${info.gpuCount} GPU(s):`);
    for (const gpu of info.gpus) {
      lines.push(`- ${gpu.name}: ${gpu.vram} GB VRAM, compute capability ${gpu.computeCapability}`);
    }
  } else {
    lines.push('CUDA not available');
  }
  if (info.mpsAvailable) lines.push('Apple MPS available');
  if (info.currentDevice) lines.push(`current training device: ${info.currentDevice}`);
  return lines.join('\n');
}

export async function executeGraphTool(
  graph: GraphToolApi,
  domain: Domain,
  name: string,
  args: Args,
): Promise<string> {
  try {
    switch (name) {
      case 'set_node_property':
        return await setNodeProperty(graph, domain, args);
      case 'add_node':
        return await addNode(graph, domain, args);
      case 'connect':
        return await connect(graph, args);
      case 'remove_node':
        return await removeNode(graph, args);
      case 'remove_edge':
        return await removeEdge(graph, args);
      case 'clear_graph':
        graph.clearGraph();
        return 'ok: cleared the canvas (all nodes and edges removed)';
      case 'organize_layout':
        graph.organizeGraph();
        return 'ok: re-arranged the node layout';
      case 'add_block':
        return await addBlock(graph, args);
      case 'enter_block':
        return enterBlock(graph, args);
      case 'exit_block':
        graph.exitSubgraph();
        return 'ok: exited to the parent canvas';
      case 'save_block':
        return await saveBlock(graph, args);
      case 'get_graph':
        return summarizeGraph(graph.getCurrentGraph());
      case 'get_node':
        return getNode(graph, args);
      case 'get_dataset_info':
        return await getDatasetInfo(graph, domain, args);
      case 'get_training_results':
        return getTrainingResults(graph);
      case 'get_training_status':
        return getTrainingStatus(graph);
      case 'get_epoch_detail':
        return getEpochDetail(graph, args);
      case 'get_test_results':
        return getTestResults(graph);
      case 'get_saved_runs':
        return await getSavedRuns(args);
      case 'start_training':
        return startTraining(graph, domain);
      case 'stop_training':
        return stopTraining(graph);
      case 'run_test':
        return await runTestSet(graph);
      case 'get_system_info':
        return await getSystemInfo();
      case 'validate':
        return validate(graph, domain, args);
      default:
        return `error: unknown tool "${name}"`;
    }
  } catch (e) {
    return `error: ${e instanceof Error ? e.message : String(e)}`;
  }
}

async function setNodeProperty(graph: GraphToolApi, domain: Domain, args: Args) {
  const nodeId = String(args.nodeId ?? '');
  const key = String(args.key ?? '');
  const node = graph.getCurrentGraph().nodes.get(nodeId);
  if (!node) return `error: no node with id "${nodeId}"`;
  const def = domain.nodeRegistry.get(node.type);
  const props = def?.getProperties() ?? [];
  if (def && !props.some((p) => p.id === key)) {
    return `error: "${node.type}" has no property "${key}". Valid keys: ${props.map((p) => p.id).join(', ') || '(none)'}`;
  }
  await graph.updateProperty(nodeId, key, args.value);
  return `ok: set ${nodeId}.${key} = ${JSON.stringify(args.value)}`;
}

async function addNode(graph: GraphToolApi, domain: Domain, args: Args) {
  const type = String(args.type ?? '');
  const def = domain.nodeRegistry.get(type);
  if (!def) return `error: unknown node type "${type}"`;
  const requestedId = typeof args.id === 'string' && args.id ? args.id : undefined;
  const id = await graph.addNode(type, nextPosition(graph.getCurrentGraph()), requestedId);
  if (!id) return `error: failed to add node "${type}"`;
  // Apply only property keys the node actually has — same check as
  // set_node_property — and report the rest so the model can self-correct.
  const valid = new Set(def.getProperties().map((p) => p.id));
  const unknown: string[] = [];
  if (args.properties && typeof args.properties === 'object') {
    for (const [k, v] of Object.entries(args.properties as Args)) {
      if (!valid.has(k)) {
        unknown.push(k);
        continue;
      }
      await graph.updateProperty(id, k, v);
    }
  }
  let msg = `ok: added ${type} as "${id}"`;
  if (unknown.length) {
    msg += ` (ignored unknown propert${unknown.length > 1 ? 'ies' : 'y'} ${unknown.join(', ')} — valid keys: ${[...valid].join(', ') || '(none)'})`;
  }
  return msg;
}

async function connect(graph: GraphToolApi, args: Args) {
  const sourceId = String(args.sourceId ?? '');
  const targetId = String(args.targetId ?? '');
  const sourcePort = String(args.sourcePort ?? '');
  const targetPort = String(args.targetPort ?? '');
  const g = graph.getCurrentGraph();
  if (!g.nodes.has(sourceId)) return `error: no node "${sourceId}"`;
  if (!g.nodes.has(targetId)) return `error: no node "${targetId}"`;
  const connection: RF.Connection = {
    source: sourceId,
    sourceHandle: sourcePort,
    target: targetId,
    targetHandle: targetPort,
  };
  if (!graph.isValidConnection(connection)) {
    return `error: invalid connection ${sourceId}.${sourcePort} -> ${targetId}.${targetPort} (check port ids and type compatibility)`;
  }
  await graph.connect(connection);
  return `ok: connected ${sourceId}.${sourcePort} -> ${targetId}.${targetPort}`;
}

async function removeNode(graph: GraphToolApi, args: Args) {
  const nodeId = String(args.nodeId ?? '');
  if (!graph.getCurrentGraph().nodes.has(nodeId)) return `error: no node "${nodeId}"`;
  await graph.removeNode(nodeId);
  return `ok: removed ${nodeId}`;
}

async function removeEdge(graph: GraphToolApi, args: Args) {
  const sourceId = String(args.sourceId ?? '');
  const sourcePort = String(args.sourcePort ?? '');
  const targetId = String(args.targetId ?? '');
  const targetPort = String(args.targetPort ?? '');
  const ok = await graph.removeEdge(sourceId, sourcePort, targetId, targetPort);
  return ok
    ? `ok: disconnected ${sourceId}.${sourcePort} -> ${targetId}.${targetPort}`
    : `error: no edge ${sourceId}.${sourcePort} -> ${targetId}.${targetPort}`;
}

async function addBlock(graph: GraphToolApi, args: Args) {
  const filename = String(args.filename ?? '');
  if (!filename) return 'error: filename required';
  await graph.addBlockFromTemplate(filename, nextPosition(graph.getCurrentGraph()));
  return `ok: added block "${filename}"`;
}

function enterBlock(graph: GraphToolApi, args: Args) {
  const nodeId = String(args.nodeId ?? '');
  const node = graph.getCurrentGraph().nodes.get(nodeId);
  if (!node) return `error: no node with id "${nodeId}"`;
  if (node.type !== 'subgraph.block' || !node.subgraph) {
    return `error: "${nodeId}" is not a custom block — only subgraph.block nodes can be entered`;
  }
  graph.enterSubgraph(nodeId);
  return `ok: entered block "${nodeId}" — add_node/connect now edit its inner graph (use exit_block when done)`;
}

async function saveBlock(graph: GraphToolApi, args: Args) {
  const nodeId = String(args.nodeId ?? '');
  const node = graph.getCurrentGraph().nodes.get(nodeId);
  if (!node) return `error: no node with id "${nodeId}"`;
  if (node.type !== 'subgraph.block' || !node.subgraph) {
    return `error: "${nodeId}" is not a custom block — only subgraph.block nodes can be saved`;
  }
  const name = typeof args.name === 'string' && args.name ? args.name : undefined;
  if (name) await graph.updateProperty(nodeId, 'blockName', name);
  await graph.saveBlock(nodeId);
  return `ok: saved "${nodeId}" to your block library${name ? ` as "${name}"` : ''}`;
}

function getNode(graph: GraphToolApi, args: Args) {
  const nodeId = String(args.nodeId ?? '');
  const node = graph.getCurrentGraph().nodes.get(nodeId);
  if (!node) return `error: no node with id "${nodeId}"`;
  return describeNode(node);
}

/** Text summary of a /dataset-detail payload for the model — everything except
 *  the pixel blobs (sampleImages would flood the context). */
function summarizeDatasetInfo(d: DatasetInfo): string {
  const lines = [`${d.name} — ${d.description}`];
  const counts = [`train samples: ${d.trainSamples}`];
  if (typeof d.testSamples === 'number') counts.push(`test samples: ${d.testSamples}`);
  lines.push(counts.join(', '));
  if (d.channels != null && d.imageSize) {
    lines.push(`images: ${d.channels}×${d.imageSize.join('×')} (C×H×W)`);
  }
  if (d.isLanguageModel) lines.push(`language-model corpus, vocab size: ${d.vocabSize ?? '?'}`);
  else if (d.isText) lines.push('text dataset');
  if (d.labels?.length) {
    const shown = d.labels.slice(0, 30).join(', ');
    lines.push(`classes (${d.labels.length}): ${shown}${d.labels.length > 30 ? ', …' : ''}`);
  }
  // A snippet or two so the model can see what a sample looks like.
  for (const [label, texts] of Object.entries(d.sampleTexts ?? {}).slice(0, 2)) {
    const t = texts?.[0];
    if (t) lines.push(`sample (${label}): "${t.slice(0, 150)}${t.length > 150 ? '…' : ''}"`);
  }
  return lines.join('\n');
}

/** Bridge to POST /dataset-detail with the node's live properties — the same
 *  endpoint the inspector modal uses, so data.custom resolves from its spec. */
async function getDatasetInfo(graph: GraphToolApi, domain: Domain, args: Args) {
  const nodeId = String(args.nodeId ?? '');
  const node = graph.getCurrentGraph().nodes.get(nodeId);
  if (!node) return `error: no node with id "${nodeId}"`;
  const def = domain.nodeRegistry.get(node.type);
  if (def?.category[0] !== 'Data') {
    return `error: "${nodeId}" (${node.type}) is not a data node`;
  }
  const res = await callBackend<{ detail: DatasetInfo }>('/dataset-detail', {
    type: node.type,
    id: node.id,
    properties: node.properties,
  });
  if (!res.ok) return `error: ${res.error}`;
  return summarizeDatasetInfo(res.data.detail);
}

function validate(graph: GraphToolApi, domain: Domain, args: Args) {
  const mode = args.mode === 'training' ? 'training' : 'forward';
  const g = graph.getCurrentGraph();

  // Structural checks (ports connected, required nodes present, reachability).
  const structural =
    mode === 'training'
      ? validateTraining(g, domain.nodeRegistry)
      : validateForward(g, domain.nodeRegistry);
  const lines = structural.map((e) => `- ${e.nodeId ? e.nodeId + ': ' : ''}${e.message}`);

  // Shape checks. validateForward/Training never run the shape math, so a node
  // can be fully wired (structurally valid) yet still have a red shape error in
  // its lastResult — e.g. a 4-D tensor reaching CrossEntropyLoss. Surface those
  // here so the finish-pass / auto-repair loop actually sees them.
  for (const [nodeId, node] of g.nodes) {
    const err = (node.lastResult?.metadata as Record<string, unknown> | undefined)?.error;
    if (typeof err === 'string' && err) lines.push(`- ${nodeId}: ${err}`);
  }

  if (!lines.length) return `ok: no ${mode} validation issues`;
  return `${lines.length} ${mode} issue(s):\n` + lines.join('\n');
}
