import { EngineConfig, StepSnapshot, TrainingDataset, TrainingSession } from "../core/types";

export type WorkerRequest =
  | { id: string; type: "init"; payload: { config: EngineConfig; dataset: TrainingDataset } }
  | { id: string; type: "nextAtomicStep" }
  | { id: string; type: "nextMajorBlock" }
  | { id: string; type: "runToEnd" }
  | { id: string; type: "snapshot" }
  | { id: string; type: "replaceSession"; payload: { session: TrainingSession } }
  | { id: string; type: "reset" };

export type WorkerResponse =
  | { id: string; ok: true; type: "init"; payload: { snapshot: StepSnapshot; session: TrainingSession } }
  | { id: string; ok: true; type: "nextAtomicStep"; payload: { snapshot: StepSnapshot; session: TrainingSession } }
  | { id: string; ok: true; type: "nextMajorBlock"; payload: { snapshot: StepSnapshot; session: TrainingSession } }
  | { id: string; ok: true; type: "runToEnd"; payload: { snapshot: StepSnapshot; session: TrainingSession } }
  | { id: string; ok: true; type: "snapshot"; payload: { snapshot: StepSnapshot | null; session: TrainingSession | null } }
  | { id: string; ok: true; type: "replaceSession"; payload: { snapshot: StepSnapshot; session: TrainingSession } }
  | { id: string; ok: true; type: "reset"; payload: { snapshot: StepSnapshot | null; session: TrainingSession | null } }
  | { id: string; ok: false; error: string };

export type WorkerRequestWithoutId = WorkerRequest extends infer Request
  ? Request extends { id: string }
    ? Omit<Request, "id">
    : never
  : never;

export type WorkerSuccessResponse = Extract<WorkerResponse, { ok: true }>;
