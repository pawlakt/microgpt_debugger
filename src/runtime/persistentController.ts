import { EngineConfig, StepSnapshot, TrainingDataset } from "../core/types";
import { EngineWorkerClient } from "./workerClient";
import { LocalSessionStore } from "./persistence";

export class PersistentEngineController {
  constructor(
    private readonly workerClient: EngineWorkerClient,
    private readonly store: LocalSessionStore
  ) {}

  async init(config: EngineConfig, dataset: TrainingDataset): Promise<StepSnapshot> {
    const restored = this.store.load();
    if (restored) {
      const { snapshot, session } = await this.workerClient.replaceSession(restored);
      this.store.save(session);
      return snapshot;
    }
    const { snapshot, session } = await this.workerClient.init(config, dataset);
    this.store.save(session);
    return snapshot;
  }

  async nextAtomicStep(): Promise<StepSnapshot> {
    const { snapshot, session } = await this.workerClient.nextAtomicStep();
    this.store.save(session);
    return snapshot;
  }

  async nextMajorBlock(): Promise<StepSnapshot> {
    const { snapshot, session } = await this.workerClient.nextMajorBlock();
    this.store.save(session);
    return snapshot;
  }

  async runToEnd(): Promise<StepSnapshot> {
    const { snapshot, session } = await this.workerClient.runToEnd();
    this.store.save(session);
    return snapshot;
  }

  async snapshot(): Promise<StepSnapshot | null> {
    const { snapshot, session } = await this.workerClient.snapshot();
    if (session) {
      this.store.save(session);
    }
    return snapshot;
  }

  async reset(): Promise<void> {
    await this.workerClient.reset();
    this.store.clear();
  }
}
