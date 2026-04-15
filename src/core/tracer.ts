import { MajorBlock, TraceEvent, TraceOpKind } from "./types";

export class StepBudgetExceededError extends Error {
  constructor() {
    super("Step budget exceeded.");
    this.name = "StepBudgetExceededError";
  }
}

interface TraceDraft {
  op: TraceOpKind;
  lhs?: number;
  rhs?: number;
  exponent?: number;
  result?: number;
  nodeId?: number;
  childNodeId?: number;
  delta?: number;
}

export class OperationTracer {
  private readonly events: TraceEvent[] = [];
  private currentMajorBlock: MajorBlock = MajorBlock.DataPreparation;

  constructor(
    private readonly step: number,
    private readonly opLimit: number | null
  ) {}

  setMajorBlock(block: MajorBlock): void {
    this.currentMajorBlock = block;
  }

  record(draft: TraceDraft): void {
    if (this.opLimit !== null && this.events.length >= this.opLimit) {
      throw new StepBudgetExceededError();
    }
    this.events.push({
      index: this.events.length,
      step: this.step,
      majorBlock: this.currentMajorBlock,
      ...draft
    });
  }

  getEvents(): TraceEvent[] {
    return this.events;
  }
}
