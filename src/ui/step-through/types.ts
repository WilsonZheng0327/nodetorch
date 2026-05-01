// Types for step-through visualization data (mirrors backend schema).
// Each layer type gets its own Transformation showing HOW it transforms data.
// Every transformation includes before AND after data for consistent display.

// --- Shared building blocks ---

export interface FeatureMaps {
  maps: number[][][];    // [channel][y][x] of 0-255 byte values
  channels: number;      // total channels in the tensor
  showing: number;       // how many channels returned
  height: number;
  width: number;
}

export interface HistogramData {
  bins: number[];
  counts: number[];
  mean: number;
  std: number;
}

// --- Per-layer transformation types ---

/** Conv2d/Conv1d/ConvTranspose2d: input maps → kernels → output maps */
export interface Conv2dTransformation {
  type: 'conv2d';
  input: FeatureMaps;
  output: FeatureMaps;
  kernels: {
    data: number[][][];
    showing: number;
    totalFilters: number;
    kernelH: number;
    kernelW: number;
  } | null;
  // Interactive detail
  rawInputs?: number[][][][];   // [in_ch][H][W] actual float values per input channel
  rawInputH?: number;
  rawInputW?: number;
  rawOutputs?: number[][][];    // [filter][H][W] actual float values per filter
  allKernels?: number[][][][];  // [filter][in_ch][kH][kW] full kernel weights
  biases?: number[];
  stride?: number[];
  padding?: number[];
  inputChannels?: number;
  isTranspose?: boolean;
}

/** Linear: input vector → output vector */
export interface LinearTransformation {
  type: 'linear';
  inputVector: number[];
  outputVector: number[];
  inputDim: number;
  outputDim: number;
}

/** Activations: before/after feature maps and histograms with curve overlay */
export interface ActivationTransformation {
  type: 'activation';
  fn: 'relu' | 'sigmoid' | 'tanh' | 'gelu' | 'leaky_relu';
  points: { x: number; y: number }[];
  deadFraction?: number;
  saturatedFraction?: number;
  inputMaps?: FeatureMaps;
  outputMaps?: FeatureMaps;
  inputHist?: HistogramData;
  outputHist?: HistogramData;
  sharedXMin?: number;
  sharedXMax?: number;
  negativeSlope?: number;   // LeakyReLU actual slope
}

/** Softmax: raw logits → probabilities */
export interface SoftmaxTransformation {
  type: 'softmax';
  logits: number[];
  probabilities: number[];
  topK: { index: number; value: number }[];
}

/** Normalization: before/after distributions */
export interface NormTransformation {
  type: 'norm';
  normKind: string;
  inputHist: HistogramData;
  outputHist: HistogramData;
  gamma?: number[];
  beta?: number[];
}

/** Pooling: spatial downsampling */
export interface PoolTransformation {
  type: 'pool';
  poolKind: 'max' | 'avg' | 'adaptive_avg';
  input: FeatureMaps;
  output: FeatureMaps;
  rawInput?: number[][];
  rawInputH?: number;
  rawInputW?: number;
  rawOutput?: number[][];
  poolSize?: number[];
  strideSize?: number[];
  paddingSize?: number[];
  channelValues?: number[];   // per-channel scalar values when output is 1x1
}

/** Flatten: 3D → 1D (before feature maps + after pixel strip) */
export interface FlattenTransformation {
  type: 'flatten';
  inputShape: number[];
  inputMaps?: FeatureMaps;
  outputLength: number;
  /** All values as 0-255 pixel intensities (min-max normalized), for the flattened image strip */
  flatPixels: number[];
}

/** Upsample: spatial upsampling */
export interface UpsampleTransformation {
  type: 'upsample';
  input: FeatureMaps;
  output: FeatureMaps;
}

/** Dropout */
export interface DropoutTransformation {
  type: 'dropout';
  inputMaps?: FeatureMaps;
  outputMaps?: FeatureMaps;
  inputHist?: HistogramData;
  outputHist?: HistogramData;
  inputNonzero?: number;
  outputNonzero?: number;
  totalElements?: number;
}

/** CrossEntropy loss: shows prediction breakdown and loss calculation */
export interface CrossEntropyTransformation {
  type: 'cross_entropy';
  logits: number[];
  probabilities: number[];
  trueLabel: number;
  trueLabelProb: number;
  loss: number;
  topK: { index: number; value: number }[];
  classNames?: string[];
}

/** Data (input sample): raw input preview + optional normalization histograms */
export interface DataTransformation {
  type: 'data';
  featureMaps?: FeatureMaps;
  vector?: { values: number[]; totalLength: number };
  rawHist?: HistogramData;    // before normalization (0-1 pixel values)
  normHist?: HistogramData;   // after normalization (centered around 0)
}

/** GAN Loss: real vs fake discriminator scores */
export interface GanLossTransformation {
  type: 'gan_loss';
  realScore?: number;
  realProb?: number;
  fakeScore?: number;
  fakeProb?: number;
  dLossReal?: number;
  dLossFake?: number;
  totalLoss?: number;
}

/** VAE Loss: reconstruction + KL divergence */
export interface VaeLossTransformation {
  type: 'vae_loss';
  originalFmaps?: FeatureMaps;
  reconFmaps?: FeatureMaps;
  reconLoss?: number;
  klLoss?: number;
  totalLoss?: number;
  beta?: number;
}

/** Loss: scalar value (for non-cross-entropy losses) */
export interface LossTransformation {
  type: 'loss';
  value: number;
}

/** Reparameterize (VAE): mean + logvar → sampled z */
export interface ReparameterizeTransformation {
  type: 'reparameterize';
  meanValues?: number[];
  logvarValues?: number[];
  zValues?: number[];
  latentDim?: number;
  meanHist?: HistogramData;
  logvarHist?: HistogramData;
}

/** Pretrained model: model info + before/after */
export interface PretrainedTransformation {
  type: 'pretrained';
  modelName: string;
  pretrainedOn: string;
  topAcc: string;
  totalParams: string;
  trainableParams?: string;
  frozen?: boolean;
  mode?: string;
  architecture?: { name: string; detail: string }[];
  inputFmaps?: FeatureMaps;
  inputShape?: number[];
  outputShape?: number[];
  outputFmaps?: FeatureMaps;
  outputVector?: number[];
  outputDim?: number;
  outputHist?: HistogramData;
}

/** Reshape: same data, different layout */
export interface ReshapeTransformation {
  type: 'reshape';
  inputShape: number[];
  outputShape: number[];
  inputFmaps?: FeatureMaps;
  inputVector?: { values: number[]; totalLength: number };
  outputFmaps?: FeatureMaps;
  outputVector?: { values: number[]; totalLength: number };
}

/** Noise Scheduler: clean + noise = noisy + timestep channel */
export interface NoiseSchedulerTransformation {
  type: 'noise_scheduler';
  cleanFmaps?: FeatureMaps;
  cleanShape?: number[];
  noiseFmaps?: FeatureMaps;
  noisyFmaps?: FeatureMaps;
  noisyShape?: number[];
  timestep?: number;
  numTimesteps?: number;
  tNormalized?: number;
  timestepShape?: number[];
  signalRatio?: number;
  noiseRatio?: number;
  concatResult?: string;
  concatExplain?: string;
}

/** Concat: shows all inputs + concatenated output */
export interface ConcatTransformation {
  type: 'concat';
  inputs: {
    label: string;
    shape: number[];
    featureMaps?: FeatureMaps;
    vector?: { values: number[]; totalLength: number };
    isConstant?: boolean;
    constantValue?: number;
  }[];
  outputShape?: number[];
  outputFmaps?: FeatureMaps;
  outputVector?: { values: number[]; totalLength: number };
  dim: number;
}

/** MSE Loss: predictions vs targets */
export interface MseLossTransformation {
  type: 'mse_loss';
  predsFmaps?: FeatureMaps;
  predsShape?: number[];
  targetsFmaps?: FeatureMaps;
  targetsShape?: number[];
  loss?: number;
  numElements?: number;
  sumSquared?: number;
  meanSquared?: number;
  maxAbsError?: number;
  meanAbsError?: number;
  errorMap?: number[][];
  errorH?: number;
  errorW?: number;
}

/** Add (residual): shows all inputs and the element-wise sum */
export interface AddTransformation {
  type: 'add';
  inputs: { label: string; featureMaps?: FeatureMaps; vector?: { values: number[]; totalLength: number } }[];
  output: FeatureMaps | null;
  outputVector?: { values: number[]; totalLength: number };
}

/** Generic fallback */
export interface DefaultTransformation {
  type: 'default';
  featureMaps?: FeatureMaps;
  vector?: { values: number[]; totalLength: number };
  scalar?: number;
}

export type Transformation =
  | Conv2dTransformation
  | LinearTransformation
  | ActivationTransformation
  | SoftmaxTransformation
  | NormTransformation
  | PoolTransformation
  | FlattenTransformation
  | UpsampleTransformation
  | DropoutTransformation
  | CrossEntropyTransformation
  | NoiseSchedulerTransformation
  | ConcatTransformation
  | MseLossTransformation
  | AddTransformation
  | PretrainedTransformation
  | ReparameterizeTransformation
  | ReshapeTransformation
  | GanLossTransformation
  | VaeLossTransformation
  | DataTransformation
  | LossTransformation
  | DefaultTransformation;

// --- Stage ---

export interface Stage {
  stageId: string;
  path: string[];
  nodeId: string;
  nodeType: string;
  displayName: string;
  depth: number;
  blockName?: string;
  inputShape?: number[];
  outputShape?: number[];
  transformation?: Transformation;
  insight?: string;
}

export interface SampleInfo {
  datasetType?: string;
  imagePixels?: number[][] | number[][][];
  imageChannels?: number;
  actualLabel?: number;
  classNames?: string[];
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

// --- Backward (placeholder) ---
export interface BackwardStage extends Stage { gradientShape?: number[]; }
export interface BackwardStepThroughResult { stages: BackwardStage[]; loss: number; sample: SampleInfo; modelState?: ModelState; }

export type StepThroughMode = 'forward' | 'backward' | 'denoise' | 'generate';

// --- Text generation ---
export interface TextGenerationResult { prompt: string; generated: string; fullText: string; tokens: { char: string; prob: number }[]; }

// --- Diffusion denoising ---
export interface DenoiseStep { timestep: number; pixels: (number[][] | number[][][])[]; }
export interface DenoiseStepThroughResult { steps: DenoiseStep[]; numTimesteps: number; numSamples: number; imageH: number; imageW: number; channels: number; }
