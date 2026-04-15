import { OperationTracer } from "./tracer";

export class Value {
  private static tracer: OperationTracer | null = null;
  private static nextId = 1;

  static setTracer(tracer: OperationTracer | null): void {
    Value.tracer = tracer;
  }

  readonly id: number;
  data: number;
  grad = 0;
  private readonly children: Value[];
  private readonly localGrads: number[];

  constructor(data: number, children: Value[] = [], localGrads: number[] = []) {
    this.id = Value.nextId;
    Value.nextId += 1;
    this.data = data;
    this.children = children;
    this.localGrads = localGrads;
  }

  static from(value: Value | number): Value {
    return value instanceof Value ? value : new Value(value);
  }

  add(other: Value | number): Value {
    const rhs = Value.from(other);
    const result = this.data + rhs.data;
    Value.tracer?.record({
      op: "value_add",
      lhs: this.data,
      rhs: rhs.data,
      result,
      nodeId: this.id
    });
    return new Value(result, [this, rhs], [1, 1]);
  }

  mul(other: Value | number): Value {
    const rhs = Value.from(other);
    const result = this.data * rhs.data;
    Value.tracer?.record({
      op: "value_mul",
      lhs: this.data,
      rhs: rhs.data,
      result,
      nodeId: this.id
    });
    return new Value(result, [this, rhs], [rhs.data, this.data]);
  }

  pow(exponent: number): Value {
    const result = this.data ** exponent;
    Value.tracer?.record({
      op: "value_pow",
      lhs: this.data,
      exponent,
      result,
      nodeId: this.id
    });
    return new Value(result, [this], [exponent * this.data ** (exponent - 1)]);
  }

  log(): Value {
    const result = Math.log(this.data);
    Value.tracer?.record({
      op: "value_log",
      lhs: this.data,
      result,
      nodeId: this.id
    });
    return new Value(result, [this], [1 / this.data]);
  }

  exp(): Value {
    const result = Math.exp(this.data);
    Value.tracer?.record({
      op: "value_exp",
      lhs: this.data,
      result,
      nodeId: this.id
    });
    return new Value(result, [this], [result]);
  }

  relu(): Value {
    const result = Math.max(0, this.data);
    Value.tracer?.record({
      op: "value_relu",
      lhs: this.data,
      result,
      nodeId: this.id
    });
    return new Value(result, [this], [Number(this.data > 0)]);
  }

  neg(): Value {
    return this.mul(-1);
  }

  sub(other: Value | number): Value {
    return this.add(Value.from(other).neg());
  }

  div(other: Value | number): Value {
    return this.mul(Value.from(other).pow(-1));
  }

  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();

    const buildTopo = (value: Value): void => {
      if (visited.has(value)) {
        return;
      }
      visited.add(value);
      for (const child of value.children) {
        buildTopo(child);
      }
      topo.push(value);
    };

    buildTopo(this);
    this.grad = 1;

    for (let i = topo.length - 1; i >= 0; i -= 1) {
      const value = topo[i] as Value;
      for (let childIndex = 0; childIndex < value.children.length; childIndex += 1) {
        const child = value.children[childIndex] as Value;
        const localGrad = value.localGrads[childIndex] as number;
        const delta = localGrad * value.grad;
        Value.tracer?.record({
          op: "grad_accumulate",
          nodeId: value.id,
          childNodeId: child.id,
          lhs: child.grad,
          rhs: delta,
          result: child.grad + delta,
          delta
        });
        child.grad += delta;
      }
    }
  }
}

export function sumValues(values: Value[]): Value {
  let total = new Value(0);
  for (const value of values) {
    total = total.add(value);
  }
  return total;
}
