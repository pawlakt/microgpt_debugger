const MT_N = 624;
const MT_M = 397;
const MT_MATRIX_A = 0x9908b0df;
const MT_UPPER_MASK = 0x80000000;
const MT_LOWER_MASK = 0x7fffffff;
const TWO_POW_53 = 9007199254740992;

function uint32(value: number): number {
  return value >>> 0;
}

export class PythonRandom {
  private readonly mt = new Uint32Array(MT_N);
  private index = MT_N;
  private gaussNext: number | null = null;

  constructor(seed: number) {
    this.seed(seed);
  }

  seed(seed: number): void {
    const normalized = uint32(Math.trunc(seed));
    this.mt[0] = normalized;
    for (let i = 1; i < MT_N; i += 1) {
      const prev = this.mt[i - 1] ?? 0;
      this.mt[i] = uint32((1812433253 * (prev ^ (prev >>> 30)) + i) >>> 0);
    }
    this.index = MT_N;
    this.gaussNext = null;
  }

  random(): number {
    const a = this.extract32() >>> 5;
    const b = this.extract32() >>> 6;
    return (a * 67108864 + b) / TWO_POW_53;
  }

  gauss(mu = 0, sigma = 1): number {
    if (this.gaussNext !== null) {
      const value = this.gaussNext;
      this.gaussNext = null;
      return mu + value * sigma;
    }
    const x2pi = this.random() * 2 * Math.PI;
    const g2rad = Math.sqrt(-2 * Math.log(1 - this.random()));
    const z = Math.cos(x2pi) * g2rad;
    this.gaussNext = Math.sin(x2pi) * g2rad;
    return mu + z * sigma;
  }

  shuffleInPlace<T>(items: T[]): void {
    for (let i = items.length - 1; i > 0; i -= 1) {
      const j = this.randBelow(i + 1);
      const tmp = items[i];
      items[i] = items[j] as T;
      items[j] = tmp as T;
    }
  }

  choiceWeightedIndex(weights: number[]): number {
    if (weights.length === 0) {
      throw new Error("weights cannot be empty.");
    }
    const total = weights.reduce((acc, value) => acc + value, 0);
    if (!(total > 0) || !Number.isFinite(total)) {
      throw new Error("weights total must be a finite positive value.");
    }
    const target = this.random() * total;
    let cumulative = 0;
    for (let index = 0; index < weights.length; index += 1) {
      cumulative += weights[index] as number;
      if (target < cumulative) {
        return index;
      }
    }
    return weights.length - 1;
  }

  private randBelow(n: number): number {
    if (!Number.isInteger(n) || n <= 0) {
      throw new Error("randBelow requires a positive integer.");
    }
    const limit = Math.floor(0x100000000 / n) * n;
    while (true) {
      const candidate = this.extract32();
      if (candidate < limit) {
        return candidate % n;
      }
    }
  }

  private extract32(): number {
    if (this.index >= MT_N) {
      this.twist();
    }
    let y = this.mt[this.index] as number;
    this.index += 1;
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return uint32(y);
  }

  private twist(): void {
    for (let i = 0; i < MT_N; i += 1) {
      const y = ((this.mt[i] as number) & MT_UPPER_MASK) | ((this.mt[(i + 1) % MT_N] as number) & MT_LOWER_MASK);
      let next = (this.mt[(i + MT_M) % MT_N] as number) ^ (y >>> 1);
      if (y & 1) {
        next ^= MT_MATRIX_A;
      }
      this.mt[i] = uint32(next);
    }
    this.index = 0;
  }
}
