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

export interface BackwardStage extends Stage {
  /** Shape of the gradient flowing into this layer (== its output shape). */
  gradientShape?: number[];
  /** Stats of that incoming gradient. */
  stats?: GradStats;
  /** Weight-gradient stats (absent for layers with no weights). */
  paramGradStats?: ParamGradStats;
}

export interface BackwardResult {
  stages: BackwardStage[];
  loss: number;
  /** Index of the sample the gradients were computed on — reused by /one-step. */
  sampleIdx: number | null;
  sample: {
    datasetType?: string;
    actualLabel?: number | null;
    imagePixels?: number[][] | number[][][];
    imageChannels?: number;
    tokenIds?: number[];
    sampleText?: string;
  };
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
