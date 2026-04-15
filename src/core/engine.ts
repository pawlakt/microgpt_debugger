import { PythonRandom } from "./random";
import { OperationTracer, StepBudgetExceededError } from "./tracer";
import {
  DEFAULT_WEIGHT_STD,
  EngineConfig,
  MajorBlock,
  PersistentModelState,
  StepResult,
  StepSnapshot,
  TraceEvent,
  TrainingDataset,
  TrainingSession,
  validateConfig
} from "./types";
import { Value, sumValues } from "./value";

type NumberMatrix = number[][];
type ValueMatrix = Value[][];

interface InternalStepResult {
  status: "paused" | "completed";
  events: TraceEvent[];
  loss: number | null;
  model: PersistentModelState | null;
}

function cloneMatrix(matrix: NumberMatrix): NumberMatrix {
  return matrix.map((row) => [...row]);
}

function cloneModel(model: PersistentModelState): PersistentModelState {
  const stateDict: Record<string, number[][]> = {};
  for (const [key, value] of Object.entries(model.stateDict)) {
    stateDict[key] = cloneMatrix(value);
  }
  return {
    stateDict,
    m: [...model.m],
    v: [...model.v]
  };
}

function matrixFromNumbers(source: NumberMatrix): ValueMatrix {
  return source.map((row) => row.map((value) => new Value(value)));
}

function matrixToNumbers(source: ValueMatrix): NumberMatrix {
  return source.map((row) => row.map((value) => value.data));
}

function buildInitialModel(config: EngineConfig, vocabSize: number, rng: PythonRandom): PersistentModelState {
  const headDim = config.nEmbd / config.nHead;
  if (!Number.isInteger(headDim)) {
    throw new Error("nEmbd must be divisible by nHead.");
  }

  const matrix = (nout: number, nin: number, std = DEFAULT_WEIGHT_STD): NumberMatrix =>
    Array.from({ length: nout }, () =>
      Array.from({ length: nin }, () => rng.gauss(0, std))
    );

  const stateDict: Record<string, NumberMatrix> = {
    wte: matrix(vocabSize, config.nEmbd),
    wpe: matrix(config.blockSize, config.nEmbd),
    lm_head: matrix(vocabSize, config.nEmbd)
  };

  for (let layer = 0; layer < config.nLayer; layer += 1) {
    stateDict[`layer${layer}.attn_wq`] = matrix(config.nEmbd, config.nEmbd);
    stateDict[`layer${layer}.attn_wk`] = matrix(config.nEmbd, config.nEmbd);
    stateDict[`layer${layer}.attn_wv`] = matrix(config.nEmbd, config.nEmbd);
    stateDict[`layer${layer}.attn_wo`] = matrix(config.nEmbd, config.nEmbd);
    stateDict[`layer${layer}.mlp_fc1`] = matrix(4 * config.nEmbd, config.nEmbd);
    stateDict[`layer${layer}.mlp_fc2`] = matrix(config.nEmbd, 4 * config.nEmbd);
  }

  const paramCount = Object.values(stateDict).reduce(
    (acc, mat) => acc + mat.reduce((rowAcc, row) => rowAcc + row.length, 0),
    0
  );

  return {
    stateDict,
    m: Array.from({ length: paramCount }, () => 0),
    v: Array.from({ length: paramCount }, () => 0)
  };
}

function linear(x: Value[], w: ValueMatrix): Value[] {
  return w.map((row) => sumValues(row.map((weight, i) => weight.mul(x[i] as Value))));
}

function softmax(logits: Value[]): Value[] {
  const maxVal = Math.max(...logits.map((value) => value.data));
  const exps = logits.map((value) => value.sub(maxVal).exp());
  const total = sumValues(exps);
  return exps.map((value) => value.div(total));
}

function rmsnorm(x: Value[]): Value[] {
  const squares = x.map((xi) => xi.mul(xi));
  const ms = sumValues(squares).div(x.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return x.map((xi) => xi.mul(scale));
}

function withMajorBlock<T>(tracer: OperationTracer, block: MajorBlock, fn: () => T): T {
  tracer.setMajorBlock(block);
  return fn();
}

function flattenParams(stateDict: Record<string, ValueMatrix>): Value[] {
  const params: Value[] = [];
  for (const matrix of Object.values(stateDict)) {
    for (const row of matrix) {
      for (const param of row) {
        params.push(param);
      }
    }
  }
  return params;
}

function gpt(
  tokenId: number,
  posId: number,
  keys: Value[][][],
  values: Value[][][],
  stateDict: Record<string, ValueMatrix>,
  config: EngineConfig,
  tracer: OperationTracer
): Value[] {
  let x = withMajorBlock(tracer, MajorBlock.EmbeddingAndNorm, () => {
    const tokenEmb = stateDict.wte[tokenId] as Value[];
    const posEmb = stateDict.wpe[posId] as Value[];
    const joint = tokenEmb.map((tokenValue, i) => tokenValue.add(posEmb[i] as Value));
    return rmsnorm(joint);
  });

  const headDim = config.nEmbd / config.nHead;
  for (let layer = 0; layer < config.nLayer; layer += 1) {
    const xResidual = x;

    const qkv = withMajorBlock(tracer, MajorBlock.AttentionQkv, () => {
      const normed = rmsnorm(x);
      const q = linear(normed, stateDict[`layer${layer}.attn_wq`] as ValueMatrix);
      const k = linear(normed, stateDict[`layer${layer}.attn_wk`] as ValueMatrix);
      const v = linear(normed, stateDict[`layer${layer}.attn_wv`] as ValueMatrix);
      return { q, k, v };
    });

    keys[layer]?.push(qkv.k);
    values[layer]?.push(qkv.v);

    const xAttn = withMajorBlock(tracer, MajorBlock.AttentionWeights, () => {
      const merged: Value[] = [];
      for (let head = 0; head < config.nHead; head += 1) {
        const hs = head * headDim;
        const qHead = qkv.q.slice(hs, hs + headDim);
        const kHead = (keys[layer] as Value[][]).map((kVec) => kVec.slice(hs, hs + headDim));
        const vHead = (values[layer] as Value[][]).map((vVec) => vVec.slice(hs, hs + headDim));

        const attnLogits = kHead.map((kToken) =>
          sumValues(
            qHead.map((qVal, j) => qVal.mul(kToken[j] as Value))
          ).div(Math.sqrt(headDim))
        );
        const attnWeights = softmax(attnLogits);

        const headOut = withMajorBlock(tracer, MajorBlock.AttentionAggregate, () =>
          Array.from({ length: headDim }, (_, dimIdx) =>
            sumValues(attnWeights.map((weight, t) => weight.mul((vHead[t] as Value[])[dimIdx] as Value)))
          )
        );
        merged.push(...headOut);
      }
      return merged;
    });

    x = withMajorBlock(tracer, MajorBlock.AttentionAggregate, () => {
      const projected = linear(xAttn, stateDict[`layer${layer}.attn_wo`] as ValueMatrix);
      return projected.map((value, i) => value.add(xResidual[i] as Value));
    });

    const mlpResidual = x;
    x = withMajorBlock(tracer, MajorBlock.MlpFc1Relu, () => {
      const normed = rmsnorm(x);
      const fc1 = linear(normed, stateDict[`layer${layer}.mlp_fc1`] as ValueMatrix);
      return fc1.map((value) => value.relu());
    });

    x = withMajorBlock(tracer, MajorBlock.MlpFc2Residual, () => {
      const fc2 = linear(x, stateDict[`layer${layer}.mlp_fc2`] as ValueMatrix);
      return fc2.map((value, i) => value.add(mlpResidual[i] as Value));
    });
  }

  return withMajorBlock(tracer, MajorBlock.LogitsAndLoss, () =>
    linear(x, stateDict.lm_head as ValueMatrix)
  );
}

function executeSingleTrainingStep(
  session: TrainingSession,
  opLimit: number | null
): InternalStepResult {
  const tracer = new OperationTracer(session.trainingStep, opLimit);
  Value.setTracer(tracer);

  try {
    const config = session.config;
    const stateDict: Record<string, ValueMatrix> = {};
    for (const [key, matrix] of Object.entries(session.model.stateDict)) {
      stateDict[key] = matrixFromNumbers(matrix);
    }
    const params = flattenParams(stateDict);

    const doc = withMajorBlock(tracer, MajorBlock.DataPreparation, () => {
      const index = session.trainingStep % session.docs.length;
      return session.docs[index] as string;
    });

    const tokenMap = new Map<string, number>();
    session.uchars.forEach((char, idx) => tokenMap.set(char, idx));
    const tokens = [session.bos, ...doc.split("").map((char) => tokenMap.get(char) as number), session.bos];
    const n = Math.min(config.blockSize, tokens.length - 1);

    const keys = Array.from({ length: config.nLayer }, () => [] as Value[][]);
    const values = Array.from({ length: config.nLayer }, () => [] as Value[][]);
    const losses: Value[] = [];

    for (let posId = 0; posId < n; posId += 1) {
      const tokenId = tokens[posId] as number;
      const targetId = tokens[posId + 1] as number;
      const logits = gpt(tokenId, posId, keys, values, stateDict, config, tracer);
      const probs = withMajorBlock(tracer, MajorBlock.LogitsAndLoss, () => softmax(logits));
      const tokenLoss = withMajorBlock(tracer, MajorBlock.LogitsAndLoss, () => probs[targetId].log().neg());
      losses.push(tokenLoss);
    }

    const loss = withMajorBlock(tracer, MajorBlock.LogitsAndLoss, () =>
      sumValues(losses).mul(1 / n)
    );

    withMajorBlock(tracer, MajorBlock.BackwardPass, () => {
      loss.backward();
    });

    withMajorBlock(tracer, MajorBlock.AdamUpdate, () => {
      const lrT = config.learningRate * (1 - session.trainingStep / config.numSteps);
      for (let i = 0; i < params.length; i += 1) {
        const param = params[i] as Value;
        session.model.m[i] = config.beta1 * (session.model.m[i] as number) + (1 - config.beta1) * param.grad;
        session.model.v[i] = config.beta2 * (session.model.v[i] as number) + (1 - config.beta2) * param.grad ** 2;
        const mHat = (session.model.m[i] as number) / (1 - config.beta1 ** (session.trainingStep + 1));
        const vHat = (session.model.v[i] as number) / (1 - config.beta2 ** (session.trainingStep + 1));
        param.data -= lrT * mHat / (Math.sqrt(vHat) + config.epsAdam);
        param.grad = 0;
      }
    });

    const nextModelState: PersistentModelState = {
      stateDict: Object.fromEntries(
        Object.entries(stateDict).map(([key, value]) => [key, matrixToNumbers(value)])
      ),
      m: [...session.model.m],
      v: [...session.model.v]
    };

    return {
      status: "completed",
      events: tracer.getEvents(),
      loss: loss.data,
      model: nextModelState
    };
  } catch (error: unknown) {
    if (error instanceof StepBudgetExceededError) {
      return {
        status: "paused",
        events: tracer.getEvents(),
        loss: null,
        model: null
      };
    }
    throw error;
  } finally {
    Value.setTracer(null);
  }
}

export function createTrainingSession(config: EngineConfig, dataset: TrainingDataset): TrainingSession {
  validateConfig(config);
  if (dataset.docs.length === 0) {
    throw new Error("Dataset is empty.");
  }

  const docs = [...dataset.docs];
  const rng = new PythonRandom(config.seed);
  rng.shuffleInPlace(docs);

  const uchars = Array.from(new Set(docs.join("").split(""))).sort((a, b) => a.localeCompare(b));
  const bos = uchars.length;
  const vocabSize = uchars.length + 1;

  return {
    config,
    docs,
    uchars,
    bos,
    vocabSize,
    model: buildInitialModel(config, vocabSize, rng),
    trainingStep: 0,
    atomicCursor: 0,
    lastLoss: null,
    lastEvent: null
  };
}

function buildSnapshot(session: TrainingSession): StepSnapshot {
  return {
    trainingStep: session.trainingStep,
    atomicCursor: session.atomicCursor,
    done: session.trainingStep >= session.config.numSteps,
    lastLoss: session.lastLoss,
    lastEvent: session.lastEvent
  };
}

function replayUntilCursor(session: TrainingSession, cursor: number | null): InternalStepResult {
  const replaySession: TrainingSession = {
    ...session,
    model: cloneModel(session.model),
    atomicCursor: 0,
    lastEvent: null,
    lastLoss: session.lastLoss
  };
  return executeSingleTrainingStep(replaySession, cursor);
}

export function nextAtomicStep(session: TrainingSession): StepResult {
  if (session.trainingStep >= session.config.numSteps) {
    return { snapshot: buildSnapshot(session), event: null };
  }

  const targetCursor = session.atomicCursor + 1;
  const replay = replayUntilCursor(session, targetCursor);
  const event = replay.events[targetCursor - 1] ?? null;

  if (replay.status === "completed") {
    session.model = replay.model as PersistentModelState;
    session.trainingStep += 1;
    session.atomicCursor = 0;
    session.lastLoss = replay.loss;
    session.lastEvent = replay.events[replay.events.length - 1] ?? null;
  } else {
    session.atomicCursor = targetCursor;
    session.lastEvent = event;
  }

  return {
    snapshot: buildSnapshot(session),
    event
  };
}

export function nextMajorBlock(session: TrainingSession): StepResult {
  if (session.trainingStep >= session.config.numSteps) {
    return { snapshot: buildSnapshot(session), event: null };
  }

  let result = nextAtomicStep(session);
  if (!result.event) {
    return result;
  }

  const currentBlock = result.event.majorBlock;
  while (
    session.trainingStep < session.config.numSteps &&
    session.atomicCursor > 0
  ) {
    const probe = replayUntilCursor(session, session.atomicCursor + 1);
    const nextEvent = probe.events[session.atomicCursor] ?? null;
    if (!nextEvent) {
      break;
    }
    if (nextEvent.majorBlock !== currentBlock) {
      return nextAtomicStep(session);
    }
    result = nextAtomicStep(session);
    if (!result.event) {
      break;
    }
  }

  return result;
}

export function runToEnd(session: TrainingSession): StepSnapshot {
  while (session.trainingStep < session.config.numSteps) {
    const replay = replayUntilCursor(session, null);
    if (replay.status !== "completed" || !replay.model) {
      throw new Error("Unexpected paused status during full execution.");
    }
    session.model = replay.model;
    session.trainingStep += 1;
    session.atomicCursor = 0;
    session.lastLoss = replay.loss;
    session.lastEvent = replay.events[replay.events.length - 1] ?? null;
  }
  return buildSnapshot(session);
}

export function getSnapshot(session: TrainingSession): StepSnapshot {
  return buildSnapshot(session);
}
