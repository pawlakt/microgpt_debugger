# Detailed Description of `microgpt.py`

## 1. What this program is

`microgpt.py` is a single-file, dependency-free implementation of GPT-style training and sampling in pure Python.  
It contains:

- dataset loading
- character-level tokenization
- a minimal scalar autograd engine
- a tiny GPT-like forward pass (attention + MLP)
- full training loop with Adam
- inference/sampling loop

The core design intent is educational clarity over speed. Every scalar operation participates in a computation graph, so the code is mathematically explicit but computationally very slow compared to tensor libraries.

---

## 2. High-level execution flow

When you run the script, it executes top-to-bottom:

1. Imports standard library modules and seeds RNG.
2. Loads `input.txt` (downloads names dataset if missing).
3. Builds a character vocabulary and BOS token.
4. Defines the `Value` autograd class.
5. Initializes model parameters in `state_dict`.
6. Defines helper ops (`linear`, `softmax`, `rmsnorm`) and `gpt(...)`.
7. Initializes Adam optimizer buffers.
8. Runs training for `num_steps = 1000`.
9. Runs inference to generate 20 sample names.

There is no function wrapper around the training script; this is an executable script, not a package module.

---

## 3. Dataset and tokenization

### 3.1 Dataset loading

- If `input.txt` does not exist, the script downloads `names.txt` from Karpathy's `makemore` repository.
- It reads non-empty lines into `docs` and shuffles them.
- Each document is a training sequence (example: one name).

### 3.2 Vocabulary

- `uchars = sorted(set(''.join(docs)))`
- Every unique character gets an integer token id from `0..len(uchars)-1`.
- A special `BOS` token id is appended as `BOS = len(uchars)`.
- `vocab_size = len(uchars) + 1`.

### 3.3 Sequence construction per training step

For one document:

- Convert chars to token ids.
- Wrap sequence with BOS on both sides:
  - `[BOS] + chars + [BOS]`
- Predict next token at each position (autoregressive objective).

This makes BOS work as both start marker and end marker.

---

## 4. Autograd engine (`Value`)

`Value` stores scalar nodes for forward and backward passes:

- `data`: scalar numeric value
- `grad`: d(loss)/d(this)
- `_children`: upstream nodes
- `_local_grads`: local derivatives wrt children

### 4.1 Supported operations

It overloads scalar math operators/functions:

- add, sub, mul, div, pow
- negation
- `log`, `exp`, `relu`

Each operation returns a new `Value` with local derivatives recorded.

### 4.2 Backward pass

`backward()`:

1. Builds topological ordering via DFS.
2. Seeds final node gradient with `1`.
3. Traverses reversed topo order and accumulates:
   - `child.grad += local_grad * current.grad`

This is reverse-mode autodiff on a scalar graph.

---

## 5. Model hyperparameters and parameter layout

Hardcoded architecture:

- `n_layer = 1`
- `n_embd = 16`
- `block_size = 16`
- `n_head = 4`
- `head_dim = 4`

Weight initialization:

- Gaussian `N(0, 0.08)` for every scalar weight.
- Stored as nested Python lists of `Value`.

### 5.1 `state_dict` entries

- `wte`: token embedding, shape `[vocab_size, n_embd]`
- `wpe`: position embedding, shape `[block_size, n_embd]`
- `lm_head`: output projection, shape `[vocab_size, n_embd]`

For each layer `i`:

- `layer{i}.attn_wq`: `[n_embd, n_embd]`
- `layer{i}.attn_wk`: `[n_embd, n_embd]`
- `layer{i}.attn_wv`: `[n_embd, n_embd]`
- `layer{i}.attn_wo`: `[n_embd, n_embd]`
- `layer{i}.mlp_fc1`: `[4*n_embd, n_embd]`
- `layer{i}.mlp_fc2`: `[n_embd, 4*n_embd]`

All parameters are flattened into `params` for optimization.

---

## 6. Core helper functions

### 6.1 `linear(x, w)`

- Input vector `x` length = `nin`
- Weight matrix `w` shape `[nout, nin]`
- Output vector length = `nout`
- Computed as scalar dot products using `Value`.

### 6.2 `softmax(logits)`

- Uses max-shift for numerical stability:
  - `exps = exp(logit - max_logit)`
- Returns normalized probabilities as `Value` scalars.

### 6.3 `rmsnorm(x)`

- Computes mean-square:
  - `ms = (1/d) * sum(x_i^2)`
- Scale:
  - `(ms + 1e-5)^(-1/2)`
- Returns `x_i * scale`.

This is RMSNorm (no learnable gain/bias in this file).

---

## 7. GPT forward path (`gpt(token_id, pos_id, keys, values)`)

The function processes one token position at a time and relies on external key/value caches.

### 7.1 Embedding + pre-layer norm

1. Lookup token embedding `wte[token_id]`.
2. Lookup position embedding `wpe[pos_id]`.
3. Add elementwise.
4. Apply `rmsnorm`.

### 7.2 Attention block (per layer)

1. Save residual.
2. RMSNorm.
3. Project to `q`, `k`, `v`.
4. Append `k`, `v` to caches (`keys[layer]`, `values[layer]`).
5. For each attention head:
   - Slice head dimensions.
   - Compute dot-product attention logits vs all cached keys.
   - Scale by `sqrt(head_dim)`.
   - Softmax to weights.
   - Weighted sum of cached values.
6. Concatenate head outputs.
7. Output projection `attn_wo`.
8. Add residual.

Because caches grow step-by-step, this behaves like causal self-attention during autoregressive processing.

### 7.3 MLP block (per layer)

1. Save residual.
2. RMSNorm.
3. `fc1` projection to `4*n_embd`.
4. ReLU activation.
5. `fc2` projection back to `n_embd`.
6. Add residual.

### 7.4 Final logits

- Apply `lm_head` linear projection to get logits over vocabulary.

---

## 8. Training objective and loop

### 8.1 Per-step sampling

- Uses one document per step: `docs[step % len(docs)]`.
- No mini-batching; batch size is effectively 1 sequence.

### 8.2 Loss construction

For each position `pos_id`:

1. Input token = current token.
2. Target token = next token.
3. Compute logits via `gpt(...)`.
4. Convert to probabilities with softmax.
5. Token loss = negative log-likelihood:
   - `-log(prob[target])`

Sequence loss is arithmetic mean over token losses.

### 8.3 Backpropagation

- Call `loss.backward()`.
- Gradients accumulate into every `Value` parameter in `params`.

### 8.4 Adam update

For each parameter index `i`:

- `m[i] = beta1*m[i] + (1-beta1)*grad`
- `v[i] = beta2*v[i] + (1-beta2)*grad^2`
- bias corrections:
  - `m_hat = m[i] / (1 - beta1^(t))`
  - `v_hat = v[i] / (1 - beta2^(t))`
- update:
  - `p -= lr_t * m_hat / (sqrt(v_hat) + eps)`

Learning rate decays linearly:

- `lr_t = learning_rate * (1 - step / num_steps)`

After update, parameter gradients are reset to zero.

---

## 9. Inference / sampling

After training:

1. For each of 20 samples, start with `token_id = BOS`.
2. Run autoregressive loop up to `block_size`.
3. Apply temperature scaling:
   - `softmax(logits / temperature)`
4. Sample next token via `random.choices(..., weights=probs)`.
5. Stop when BOS is sampled (acts as end token).
6. Convert token ids back to chars and print.

Lower `temperature` (e.g. `0.5`) makes outputs more conservative.

---

## 10. Important implementation details

### 10.1 Why this is slow

- Every scalar is a Python object (`Value`).
- Every arithmetic op builds graph nodes in Python.
- No vectorized tensor backend (NumPy/PyTorch).

This is pedagogical code, not production code.

### 10.2 Numerical behavior

- Softmax uses max-subtraction for stability.
- RMSNorm includes epsilon `1e-5`.
- Still vulnerable to instability for larger models/longer runs due to scalar implementation limits.

### 10.3 Sequence length handling

- `n = min(block_size, len(tokens) - 1)`.
- Long documents are truncated to model context window.

### 10.4 Causal behavior

- Causality is enforced implicitly by appending K/V over time and only attending to cached prefix.
- No explicit triangular mask is needed in this incremental formulation.

### 10.5 Architecture differences from full GPT-2

- RMSNorm instead of LayerNorm.
- ReLU instead of GeLU.
- No biases.
- Single-layer, tiny dimension.
- Character-level tokenizer.

---

## 11. Practical run requirements

- Python standard library only (script itself has no third-party imports).
- Write access to working directory for `input.txt` download (if missing).
- Internet only needed for first run when `input.txt` is absent.

Run:

```bash
python microgpt.py
```

Expected console output:

- number of docs
- vocabulary details
- parameter count
- step-by-step training loss (single-line carriage-return update)
- generated sample strings after training

---

## 12. Summary

`microgpt.py` is a complete minimal language-model training system:

- data -> tokenize -> forward -> loss -> backward -> Adam -> sample

It intentionally exposes the full algorithm in plain Python, making it useful for learning how GPT-style autoregressive training works at the lowest level.
