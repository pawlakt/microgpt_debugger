import { EngineConfig, StepSnapshot, TrainingDataset, TrainingSession } from "../core/types";
import {
  WorkerRequest,
  WorkerRequestWithoutId,
  WorkerResponse,
  WorkerSuccessResponse
} from "./workerTypes";

interface PendingRequest {
  resolve: (response: WorkerSuccessResponse) => void;
  reject: (reason?: unknown) => void;
}

export class EngineWorkerClient {
  private readonly pending = new Map<string, PendingRequest>();
  private nextId = 1;

  constructor(private readonly worker: Worker) {
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const response = event.data;
      const handler = this.pending.get(response.id);
      if (!handler) {
        return;
      }
      this.pending.delete(response.id);
      if (response.ok) {
        handler.resolve(response);
      } else {
        handler.reject(new Error(response.error));
      }
    };
  }

  async init(config: EngineConfig, dataset: TrainingDataset): Promise<{ snapshot: StepSnapshot; session: TrainingSession }> {
    const response = await this.send({ type: "init", payload: { config, dataset } });
    if (response.type !== "init") {
      throw new Error(`Unexpected response type: ${response.type}`);
    }
    return response.payload;
  }

  async nextAtomicStep(): Promise<{ snapshot: StepSnapshot; session: TrainingSession }> {
    const response = await this.send({ type: "nextAtomicStep" });
    if (response.type !== "nextAtomicStep") {
      throw new Error(`Unexpected response type: ${response.type}`);
    }
    return response.payload;
  }

  async nextMajorBlock(): Promise<{ snapshot: StepSnapshot; session: TrainingSession }> {
    const response = await this.send({ type: "nextMajorBlock" });
    if (response.type !== "nextMajorBlock") {
      throw new Error(`Unexpected response type: ${response.type}`);
    }
    return response.payload;
  }

  async runToEnd(): Promise<{ snapshot: StepSnapshot; session: TrainingSession }> {
    const response = await this.send({ type: "runToEnd" });
    if (response.type !== "runToEnd") {
      throw new Error(`Unexpected response type: ${response.type}`);
    }
    return response.payload;
  }

  async snapshot(): Promise<{ snapshot: StepSnapshot | null; session: TrainingSession | null }> {
    const response = await this.send({ type: "snapshot" });
    if (response.type !== "snapshot") {
      throw new Error(`Unexpected response type: ${response.type}`);
    }
    return response.payload;
  }

  async replaceSession(session: TrainingSession): Promise<{ snapshot: StepSnapshot; session: TrainingSession }> {
    const response = await this.send({ type: "replaceSession", payload: { session } });
    if (response.type !== "replaceSession") {
      throw new Error(`Unexpected response type: ${response.type}`);
    }
    return response.payload;
  }

  async reset(): Promise<void> {
    const response = await this.send({ type: "reset" });
    if (response.type !== "reset") {
      throw new Error(`Unexpected response type: ${response.type}`);
    }
  }

  private send(message: WorkerRequestWithoutId): Promise<WorkerSuccessResponse> {
    const id = `req-${this.nextId}`;
    this.nextId += 1;
    const payload = { ...message, id } as WorkerRequest;
    return new Promise<WorkerSuccessResponse>((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage(payload);
    });
  }
}
