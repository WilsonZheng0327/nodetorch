// Types for step-through visualization data (mirrors backend/step_through.py schema).
// Designed to be forward-compatible: fields are optional and discriminated unions let
// new viz kinds be added without changing existing code paths.

export type VizKind = 'image' | 'feature_maps' | 'vector' | 'probabilities' | 'scalar';

export interface Stats {
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
  sparsity?: number;
  histBins?: number[];
  histCounts?: number[];
}

export interface FeatureMapsViz {
  maps: number[][][];    // [channel][y][x] of 0-255 byte values
  channels: number;      // total channels in the output
  showing: number;       // how many channels returned
  height: number;
  width: number;
}

export interface VectorViz {
  values: number[];
  totalLength: number;
  truncated: boolean;
}

export interface ProbabilitiesViz {
  values: number[];
  topK: { index: number; value: number }[];
}

export interface ScalarViz {
  value: number;
}

export interface Viz {
  kind: VizKind;
  featureMaps?: FeatureMapsViz;
  vector?: VectorViz;
  probabilities?: ProbabilitiesViz;
  scalar?: ScalarViz;
  image?: { pixels: number[][] | number[][][]; channels: number };
}

/** Extra visualizations for a stage — discriminated union.
 *  Forward-compatible: new kinds can be added without changing existing renderers. */
export type Extra =
  | { kind: 'before_after_histograms'; input: Stats; output: Stats }
  | { kind: 'conv_kernels'; kernels: number[][][]; showing: number; totalFilters: number; inChannels: number; kernelHeight: number; kernelWidth: number }
  | { kind: 'weight_matrix'; data: number[][]; rows: number; cols: number; actualRows: number; actualCols: number; min: number; max: number }
  | { kind: 'attention_map'; data: number[][]; rows: number; cols: number; actualRows: number; actualCols: number }
  | { kind: 'recurrent_state'; label: string; values: number[]; totalLength: number }
  | { kind: 'saliency_map'; pixels: number[][]; predictedClass: number; height: number; width: number }
  // Backward-specific extras
  | { kind: 'gradient_kernels'; kernels: number[][][]; showing: number; totalFilters: number; kernelHeight: number; kernelWidth: number }
  | { kind: 'spatial_gradient_heatmap'; pixels: number[][]; height: number; width: number }
  | { kind: 'per_unit_gradient_bars'; values: number[]; label: string }
  | { kind: 'gradient_weight_matrix'; data: number[][]; rows: number; cols: number; actualRows: number; actualCols: number; min: number; max: number };

export interface Stage {
  stageId: string;        // unique — used as React key, derived from path
  path: string[];         // hierarchical location, e.g. ["conv1"] at root, ["resblock1", "conv1"] inside
  nodeId: string;
  nodeType: string;
  displayName: string;
  depth: number;          // nesting level (0 at root; >0 for subgraph interior)
  blockName?: string;     // populated for inner stages — the containing block's name
  inputShape?: number[];
  outputShape?: number[];
  stats?: Stats;
  viz?: Viz;
  extras?: Extra[];       // additional type-specific visualizations (kernels, weight matrix, etc.)
  insight?: string;       // plain-English sentence about what this layer did
}

export interface SampleInfo {
  datasetType?: string;
  imagePixels?: number[][] | number[][][];
  imageChannels?: number;
  actualLabel?: number;
  tokenIds?: number[];
  sampleText?: string;
}

export interface ModelState {
  usingTrainedWeights: boolean;
  note: string;
}

export interface StepThroughResult {
  stages: Stage[];
  sample: SampleInfo;
  modelState?: ModelState;
}

// --- Backward step-through types ---

export type GradientHealth = 'healthy' | 'vanishing' | 'exploding';

export interface ParamGradStats {
  gradNorm: number;
  weightNorm: number;
  ratio: number;
  health: GradientHealth;
}

export interface BackwardStage extends Stage {
  gradientShape?: number[];
  paramGradStats?: ParamGradStats;
}

export interface BackwardStepThroughResult {
  stages: BackwardStage[];
  loss: number;
  sample: SampleInfo;
  modelState?: ModelState;
}

export type StepThroughMode = 'forward' | 'backward';
