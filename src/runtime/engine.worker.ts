/// <reference lib="webworker" />

import {
  createTrainingSession,
  getSnapshot,
  nextAtomicStep,
  nextMajorBlock,
  runToEnd
} from "../core/engine";
import { WorkerRequest, WorkerResponse } from "./workerTypes";
import { TrainingSession } from "../core/types";

declare const self: DedicatedWorkerGlobalScope;

let session: TrainingSession | null = null;

function reply(response: WorkerResponse): void {
  self.postMessage(response);
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;

  try {
    switch (message.type) {
      case "init": {
        session = createTrainingSession(message.payload.config, message.payload.dataset);
        reply({
          id: message.id,
          ok: true,
          type: "init",
          payload: { snapshot: getSnapshot(session), session }
        });
        return;
      }
      case "nextAtomicStep": {
        if (!session) {
          throw new Error("Session is not initialized.");
        }
        const result = nextAtomicStep(session);
        reply({
          id: message.id,
          ok: true,
          type: "nextAtomicStep",
          payload: { snapshot: result.snapshot, session }
        });
        return;
      }
      case "nextMajorBlock": {
        if (!session) {
          throw new Error("Session is not initialized.");
        }
        const result = nextMajorBlock(session);
        reply({
          id: message.id,
          ok: true,
          type: "nextMajorBlock",
          payload: { snapshot: result.snapshot, session }
        });
        return;
      }
      case "runToEnd": {
        if (!session) {
          throw new Error("Session is not initialized.");
        }
        const snapshot = runToEnd(session);
        reply({
          id: message.id,
          ok: true,
          type: "runToEnd",
          payload: { snapshot, session }
        });
        return;
      }
      case "snapshot": {
        reply({
          id: message.id,
          ok: true,
          type: "snapshot",
          payload: { snapshot: session ? getSnapshot(session) : null, session }
        });
        return;
      }
      case "replaceSession": {
        session = message.payload.session;
        reply({
          id: message.id,
          ok: true,
          type: "replaceSession",
          payload: { snapshot: getSnapshot(session), session }
        });
        return;
      }
      case "reset": {
        session = null;
        reply({
          id: message.id,
          ok: true,
          type: "reset",
          payload: { snapshot: null, session: null }
        });
        return;
      }
      default: {
        const impossibleCase: never = message;
        throw new Error(`Unsupported worker message: ${JSON.stringify(impossibleCase)}`);
      }
    }
  } catch (error: unknown) {
    const errorMessage = error instanceof Error ? error.message : "Unknown worker error.";
    reply({ id: message.id, ok: false, error: errorMessage });
  }
};

export {};
