import { TrainingSession } from "../core/types";

const SESSION_STORAGE_VERSION = 1;

interface PersistedEnvelope {
  version: number;
  savedAt: string;
  session: TrainingSession;
}

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

export class LocalSessionStore {
  constructor(private readonly key: string, private readonly storage: StorageLike) {}

  save(session: TrainingSession): void {
    const envelope: PersistedEnvelope = {
      version: SESSION_STORAGE_VERSION,
      savedAt: new Date().toISOString(),
      session
    };
    this.storage.setItem(this.key, JSON.stringify(envelope));
  }

  load(): TrainingSession | null {
    const raw = this.storage.getItem(this.key);
    if (!raw) {
      return null;
    }

    const parsed = JSON.parse(raw) as PersistedEnvelope;
    if (parsed.version !== SESSION_STORAGE_VERSION || !parsed.session) {
      return null;
    }
    return parsed.session;
  }

  clear(): void {
    this.storage.removeItem(this.key);
  }
}
