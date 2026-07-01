// Shared types for the training dashboard. Split out of TrainingDashboard.tsx so
// the charts / panels / samples modules can share them without importing the
// container component.

export interface TrackedSampleProbe {
  idx: number;
  label: number | null;
  imagePixels?: number[][] | number[][][];
  imageChannels?: number;
  probabilities?: number[];
  predictedClass?: number;
  confidence?: number;
  outputNorm?: number;
  loss?: number;
}

export interface EpochData {
  epoch: number;
  perplexity?: number;
  valPerplexity?: number | null;
  generatedText?: string | null;
  totalEpochs?: number;
  loss: number;
  accuracy: number;
  valLoss?: number | null;
  valAccuracy?: number | null;
  learningRate?: number | null;
  time?: number;
  batches?: number;
  samples?: number;
  gradientFlow?: { name: string; norm: number }[];
  perClassAccuracy?: { cls: number; accuracy: number }[];
  trackedSamples?: TrackedSampleProbe[];
  generatedSamples?: (number[][] | number[][][])[];
  dLoss?: number;
  gLoss?: number;
  trainingMode?: string;
}

export interface SystemInfo {
  python: string;
  pytorch: string;
  cudaAvailable: boolean;
  gpuCount: number;
  gpus: { name: string; vram: number; computeCapability: string }[];
  mpsAvailable?: boolean;
  currentDevice?: string;
}

export interface ModelLayerInfo {
  name: string;
  type: string;
  paramCount?: number;
  outputShape?: number[] | string[];
}

export interface SavedRun {
  id: string;
  timestamp: string;
  datasetType: string;
  epochs: number;
  learningRate: number;
  optimizer: string;
  scheduler: string;
  finalLoss: number | null;
  finalAccuracy: number | null;
  bestValAccuracy: number | null;
  duration: number;
  totalParams: number;
  nodeCount: number;
}

export interface FullRun extends SavedRun {
  batchSize: number;
  seed: number;
  valSplit: number;
  epochHistory: EpochData[];
}

export interface TestResult {
  testLoss: number;
  testAccuracy: number;
  testSamples: number;
  perClassAccuracy: { cls: number; name: string; accuracy: number; count: number }[];
  confusionMatrix?: { data: number[][]; size: number; classNames?: string[] };
}
