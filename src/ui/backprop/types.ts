// Types for the backward step-through (`/backward-step-through`). A BackwardStage
// is a forward Stage plus the gradient info the backward pass produces.

import type { Stage } from '../step-through/types';

/** Stats of the gradient arriving at a layer (dL/d-output). */
export interface GradStats {
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  norm: number | null;
}

/** Weight-gradient magnitude vs the weights themselves — the "size of the nudge". */
export interface ParamGradStats {
  gradNorm: number | null;
  weightNorm: number | null;
  ratio: number | null; // gradNorm / weightNorm
  health: string; // e.g. "healthy" | "vanishing" | "exploding"
}

/** The seed of backprop, attached to the loss node: the loss and the error
 * signal ∂L/∂logits (= predicted − target) it emits into the network. */
export interface LossSeed {
  logits: number[];
  probabilities: number[];
  errorSignal: number[]; // ∂L/∂logits = p − y, the gradient backprop starts from
  trueLabel: number | null;
  trueLabelProb: number | null;
  loss: number | null;
  classNames: string[] | null;
  topK: { index: number; value: number }[];
}

/** One term of the chain-rule sum for a traced input neuron:
 * out_grad[i] · W[i, j] contributes to input j's error. */
export interface MechanismTerm {
  out: number;
  outGrad: number | null;
  weight: number | null;
  product: number | null;
}

/** The concrete chain-rule mechanism of a Linear layer: ∂L/∂input = ∂L/∂output · Wᵀ.
 * Shows how the output error routes back through the weights into the input error. */
export interface Mechanism {
  kind: 'linear';
  outDim: number;
  inDim: number;
  outGrad: number[]; // ∂L/∂output (capped)
  inGrad: number[]; // ∂L/∂input (capped)
  trace: {
    inputIndex: number;
    inputError: number | null;
    shownTerms: number;
    totalTerms: number;
    terms: MechanismTerm[];
  };
}

/** Loss-vs-one-weight sweep (from /wiggle-weight). The gradient is the slope of
 * this curve at `current.w`; a step w − lr·grad walks downhill along it. */
export interface WiggleResult {
  nodeId: string;
  displayName: string;
  coord: number[]; // per-dim index: Linear [out, in]; Conv [out, in, kh, kw]
  weightShape: number[]; // the weight tensor's shape
  numWeights: number; // total scalar weights (for "wiggle another")
  outDim: number;
  inDim: number;
  lr: number;
  optimizerType: string | null;
  sampleIdx: number | null;
  trueLabel: number | null;
  classNames: string[] | null;
  current: { w: number; loss: number; grad: number };
  step: { w: number; loss: number | null };
  /** loss (and the prediction) at each swept value of the weight. */
  curve: { w: number; loss: number | null; pTrue?: number | null; pred?: number | null }[];
}

export interface BackwardStage extends Stage {
  /** Shape of the gradient flowing into this layer (== its output shape). */
  gradientShape?: number[];
  /** Grad IN — ∂L/∂output, arriving from the downstream layer. */
  stats?: GradStats;
  /** Grad OUT — ∂L/∂input, what this layer passes back to the previous one. */
  gradOut?: { stats?: GradStats; shape?: number[] };
  /** Weight-gradient stats (absent for layers with no weights). */
  paramGradStats?: ParamGradStats;
  /** Loss node only — the backprop seed (prediction vs truth + emitted error). */
  lossSeed?: LossSeed;
  /** Linear layers — the concrete out-error → in-error mechanism. */
  mechanism?: Mechanism;
}

export interface BackwardSample {
  datasetType?: string;
  actualLabel?: number | null;
  classNames?: string[] | null;
  imagePixels?: number[][] | number[][][];
  imageChannels?: number;
  tokenIds?: number[];
  sampleText?: string;
}

export interface BackwardResult {
  stages: BackwardStage[];
  loss: number;
  /** Index of the sample the gradients were computed on — reused by /one-step. */
  sampleIdx: number | null;
  sample: BackwardSample;
}

// --- One gradient step (before/after), from /one-step ---

/** Classifier prediction summary at one point in time. */
export interface StepPrediction {
  predictedClass: number;
  confidence: number | null;
  trueLabelProb: number | null;
  probabilities: number[];
  topK: { index: number; value: number }[];
}

/** Per-layer weight change produced by the step. Keyed to a stage by nodeId. */
export interface StepLayer {
  nodeId: string;
  displayName: string;
  weightNormBefore: number | null;
  weightNormAfter: number | null;
  meanAbsDelta: number | null;
  maxAbsDelta: number | null;
}

export interface OneStepResult {
  lr: number;
  optimizerType: string | null;
  sampleIdx: number | null;
  trueLabel: number | null;
  classNames: string[] | null;
  loss: { before: number | null; after: number | null };
  prediction: { before: StepPrediction | null; after: StepPrediction | null };
  layers: StepLayer[];
}
