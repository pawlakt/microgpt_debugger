import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import {
  createTrainingSession,
  getSnapshot,
  nextAtomicStep,
  nextMajorBlock,
  parseDocs,
  runToEnd,
  withDefaultConfig
} from "../src";
import { LocalSessionStore, StorageLike } from "../src/runtime/persistence";
import { describe, expect, it } from "vitest";

function loadDocs(limit = 24): string[] {
  const inputPath = resolve(process.cwd(), "core", "input.txt");
  const raw = readFileSync(inputPath, "utf8");
  return parseDocs(raw).slice(0, limit);
}

function createConfig() {
  return withDefaultConfig(
    {
      nLayer: 1,
      nEmbd: 8,
      blockSize: 8,
      nHead: 2
    },
    42,
    4
  );
}

class MemoryStorage implements StorageLike {
  private readonly map = new Map<string, string>();

  getItem(key: string): string | null {
    return this.map.get(key) ?? null;
  }

  setItem(key: string, value: string): void {
    this.map.set(key, value);
  }

  removeItem(key: string): void {
    this.map.delete(key);
  }
}

describe("phase1 engine", () => {
  it("runs deterministically to completion for fixed seed", () => {
    const docs = loadDocs();
    const config = createConfig();
    const left = createTrainingSession(config, { docs });
    const right = createTrainingSession(config, { docs });

    runToEnd(left);
    runToEnd(right);

    expect(left.trainingStep).toBe(config.numSteps);
    expect(right.trainingStep).toBe(config.numSteps);
    expect(left.lastLoss).toBeCloseTo(right.lastLoss as number, 12);
    expect(left.model.stateDict.wte?.[0]?.[0]).toBeCloseTo(right.model.stateDict.wte?.[0]?.[0] as number, 12);
    expect(left.model.stateDict.lm_head?.[0]?.[0]).toBeCloseTo(right.model.stateDict.lm_head?.[0]?.[0] as number, 12);
    expect(left.model.m[0]).toBeCloseTo(right.model.m[0] as number, 12);
    expect(left.model.v[0]).toBeCloseTo(right.model.v[0] as number, 12);
  });

  it("advances by exactly one atomic operation", () => {
    const session = createTrainingSession(createConfig(), { docs: loadDocs() });
    const start = getSnapshot(session);
    expect(start.atomicCursor).toBe(0);

    const result = nextAtomicStep(session);
    expect(result.event).not.toBeNull();
    expect(result.event?.index).toBe(0);
    expect(result.snapshot.trainingStep).toBe(0);
    expect(result.snapshot.atomicCursor).toBe(1);
    expect(result.snapshot.lastEvent?.index).toBe(0);
  });

  it("advances to the next major block boundary", () => {
    const session = createTrainingSession(createConfig(), { docs: loadDocs() });
    const first = nextAtomicStep(session);
    const firstBlock = first.event?.majorBlock;
    expect(firstBlock).toBeTruthy();

    const next = nextMajorBlock(session);
    if (next.snapshot.trainingStep === 0 && next.snapshot.atomicCursor > 0) {
      expect(next.event?.majorBlock).not.toBe(firstBlock);
    }
  });

  it("persists and restores session through storage adapter", () => {
    const storage = new MemoryStorage();
    const store = new LocalSessionStore("microgpt:phase1", storage);
    const session = createTrainingSession(createConfig(), { docs: loadDocs() });

    nextAtomicStep(session);
    store.save(session);
    const restored = store.load();

    expect(restored).not.toBeNull();
    expect(restored?.atomicCursor).toBe(session.atomicCursor);
    expect(restored?.trainingStep).toBe(session.trainingStep);
    expect(restored?.lastEvent?.index).toBe(session.lastEvent?.index);
  });
});
