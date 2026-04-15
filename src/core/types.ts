export const DEFAULT_SEED = 42;
export const DEFAULT_NUM_STEPS = 1000;
export const DEFAULT_LEARNING_RATE = 0.01;
export const DEFAULT_BETA1 = 0.85;
export const DEFAULT_BETA2 = 0.99;
export const DEFAULT_EPS_ADAM = 1e-8;
export const DEFAULT_WEIGHT_STD = 0.08;

export enum MajorBlock {
  DataPreparation = "data_preparation",
  EmbeddingAndNorm = "embedding_and_norm",
  AttentionQkv = "attention_qkv",
  AttentionWeights = "attention_weights",
  AttentionAggregate = "attention_aggregate",
  MlpFc1Relu = "mlp_fc1_relu",
  MlpFc2Residual = "mlp_fc2_residual",
  LogitsAndLoss = "logits_and_loss",
  BackwardPass = "backward_pass",
  AdamUpdate = "adam_update"
}

export type TraceOpKind =
  | "value_add"
  | "value_mul"
  | "value_pow"
  | "value_log"
  | "value_exp"
  | "value_relu"
  | "grad_accumulate";

export interface TraceEvent {
  index: number;
  step: number;
  majorBlock: MajorBlock;
  op: TraceOpKind;
  lhs?: number;
  rhs?: number;
  exponent?: number;
  result?: number;
  nodeId?: number;
  childNodeId?: number;
  delta?: number;
}

export interface EngineConfig {
  nLayer: number;
  nEmbd: number;
  blockSize: number;
  nHead: number;
  numSteps: number;
  seed: number;
  learningRate: number;
  beta1: number;
  beta2: number;
  epsAdam: number;
}

export interface EditableConfig {
  nLayer: number;
  nEmbd: number;
  blockSize: number;
  nHead: number;
}

export interface TrainingDataset {
  docs: string[];
}

export interface PersistentModelState {
  stateDict: Record<string, number[][]>;
  m: number[];
  v: number[];
}

export interface TrainingSession {
  config: EngineConfig;
  docs: string[];
  uchars: string[];
  bos: number;
  vocabSize: number;
  model: PersistentModelState;
  trainingStep: number;
  atomicCursor: number;
  lastLoss: number | null;
  lastEvent: TraceEvent | null;
}

export interface StepSnapshot {
  trainingStep: number;
  atomicCursor: number;
  done: boolean;
  lastLoss: number | null;
  lastEvent: TraceEvent | null;
}

export interface StepResult {
  snapshot: StepSnapshot;
  event: TraceEvent | null;
}

export const DEFAULT_EDITABLE_CONFIG: EditableConfig = {
  nLayer: 1,
  nEmbd: 16,
  blockSize: 16,
  nHead: 4
};

export function withDefaultConfig(editable: EditableConfig, seed = DEFAULT_SEED, numSteps = DEFAULT_NUM_STEPS): EngineConfig {
  return {
    ...editable,
    seed,
    numSteps,
    learningRate: DEFAULT_LEARNING_RATE,
    beta1: DEFAULT_BETA1,
    beta2: DEFAULT_BETA2,
    epsAdam: DEFAULT_EPS_ADAM
  };
}

export function parseDocs(raw: string): string[] {
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

export function validateConfig(config: EngineConfig): void {
  if (config.nLayer < 1 || !Number.isInteger(config.nLayer)) {
    throw new Error("nLayer must be a positive integer.");
  }
  if (config.nEmbd < 1 || !Number.isInteger(config.nEmbd)) {
    throw new Error("nEmbd must be a positive integer.");
  }
  if (config.blockSize < 1 || !Number.isInteger(config.blockSize)) {
    throw new Error("blockSize must be a positive integer.");
  }
  if (config.nHead < 1 || !Number.isInteger(config.nHead)) {
    throw new Error("nHead must be a positive integer.");
  }
  if (config.nEmbd % config.nHead !== 0) {
    throw new Error("nEmbd must be divisible by nHead.");
  }
  if (config.numSteps < 1 || !Number.isInteger(config.numSteps)) {
    throw new Error("numSteps must be a positive integer.");
  }
  if (!Number.isFinite(config.seed)) {
    throw new Error("seed must be a finite number.");
  }
}
