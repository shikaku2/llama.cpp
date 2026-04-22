# Token Recycling — Implementation Spec for llama.cpp

## Source paper
"Turning Trash into Treasure: Accelerating Inference of Large Language Models
with Token Recycling"
Luo et al., ACL 2025
arXiv: 2408.08696

## What it does (plain English)
Every time the target model samples a token, it already computed a full
probability distribution (logits) over the vocabulary. Normally the top-1 is
kept and the rest are discarded — "trash". Token Recycling keeps the top-k
runners-up in a small adjacency matrix: for each token X, record which tokens
Y1..Yk most often follow X according to the model's own logits.

On the next draft step, start from the most-recently-accepted token and do a
breadth-first walk through the adjacency matrix up to draft_max hops. Each hop
produces one candidate; the whole walk produces a draft sequence (or draft tree
if tree-attention is available). The target model verifies all candidates in one
forward pass.

No training. No corpus. No external files. Memory overhead ≈ vocab_size * k * 8
bytes. For a 32k-vocab model at k=5 that is 1.3 MB.

Reported speedup: ~2× on Vicuna 7/13/33B. Outperforms PLD, REST, and Lookahead
by ~30%, and rivals Medusa on some benchmarks — with zero extra model weights.

---

## Why it fits llama.cpp right now
- llama.cpp already computes the full logit distribution before sampling.
- `common_sampler_sample()` in `common/sampling.cpp` is the natural hook.
- The linear (greedy-chain) variant needs NO tree attention — just pick the
  top-1 successor at each BFS step, producing a regular token vector.
- The tree variant can be added later as a separate `--spec-type` flag.

---

## New files to create

### `common/token-recycling.h`
```cpp
#pragma once
#include "llama.h"
#include <vector>
#include <cstdint>

// Adjacency entry: a candidate next-token + its accumulated score.
struct tr_entry {
    llama_token token;
    float       score;   // running softmax-weighted sum
};

// Sparse adjacency matrix.
// adj[token_id] holds the top-k most likely successors for that token,
// sorted descending by score.
// Uses a flat vector-of-vectors; vocab_size is only known at runtime.
struct token_recycling_cache {
    int                              k;          // top-k to keep per token
    std::vector<std::vector<tr_entry>> adj;      // [vocab_size][k]
    float                            decay;      // EMA decay applied each update

    // Call after target model produces logits for the current token.
    // `last_token`  = the token that was just accepted/sampled
    // `logits`      = raw logits array, length vocab_size
    // `vocab_size`  = obvious
    void update(llama_token last_token,
                const float * logits,
                int32_t vocab_size);

    // Produce a linear draft of length `n_draft` starting from `start_token`
    // by greedily walking adj.  Returns fewer tokens if a dead end is hit.
    std::vector<llama_token> draft_linear(llama_token start_token,
                                          int n_draft) const;
};

// Alloc / free
token_recycling_cache * tr_cache_init(int32_t vocab_size, int k = 5,
                                      float decay = 0.9f);
void tr_cache_free(token_recycling_cache * c);

// Optional: persist adjacency matrix across sessions (binary dump)
bool tr_cache_save(const token_recycling_cache * c, const char * path);
bool tr_cache_load(      token_recycling_cache * c, const char * path);
```

### `common/token-recycling.cpp`
Key implementation notes:

**`update()`**
```
1. Convert logits[0..vocab_size] to probabilities via softmax (temperature=1).
2. Extract the top-k (last_token, prob) pairs.
3. For each (tok, prob) in top-k:
     existing = find tok in adj[last_token]
     if found:  existing.score = decay * existing.score + (1-decay) * prob
     if not and len(adj[last_token]) < k:  append {tok, prob}
     if not and len >= k:  replace lowest-score entry if prob > lowest.score
4. Re-sort adj[last_token] descending by score.
```

The softmax is optional for performance — using raw logits with a temperature-1
rescale is fine since we only need rank ordering.

**`draft_linear()`**
```
cur = start_token
result = []
for i in 0..n_draft:
    if adj[cur] is empty: break
    next = adj[cur][0].token   // top-1 greedy
    result.push_back(next)
    cur = next
return result
```

**`tr_cache_save()` / `tr_cache_load()`**
Binary format:
```
[magic u32: 0x54524341]  // "TRCA"
[version u16: 1]
[vocab_size i32]
[k i32]
[decay f32]
for each token 0..vocab_size:
    [count u8: number of entries for this token, 0..k]
    for each entry:
        [token i32]
        [score f32]
```
Total size for k=5, vocab=32k: ~32768 * (1 + 5*8) = ~1.3 MB.

---

## Files to modify

### `common/speculative.h`
Add new spec type in the `common_speculative_type` enum:
```cpp
COMMON_SPECULATIVE_TYPE_TOKEN_RECYCLING,
```

Add field to `common_speculative_params`:
```cpp
int    spec_tr_k;          // top-k adjacency entries (default: 5)
float  spec_tr_decay;      // EMA decay (default: 0.9)
char * spec_tr_cache_path; // optional persist path (default: nullptr)
```

Add field to `common_speculative_state` (the opaque impl struct):
```cpp
token_recycling_cache * tr_cache;
```

### `common/speculative.cpp`

**`common_speculative_init()`**: allocate `tr_cache` when type is
`TOKEN_RECYCLING`. Optionally call `tr_cache_load()`.

**`common_speculative_draft()`**: add a new branch:
```cpp
case COMMON_SPECULATIVE_TYPE_TOKEN_RECYCLING: {
    // `inp` is the accepted token sequence so far.
    llama_token last = inp.back();
    draft = state->tr_cache->draft_linear(last, params.n_draft_max);
    break;
}
```

**After every accepted token** (in the verification loop that calls
`common_sampler_sample_and_accept_n`): call `tr_cache->update(accepted_token,
logits_ptr, vocab_size)`. The logits are available in
`llama_get_logits_ith(ctx, 0)` for the last target forward pass.

Hook location: `common_speculative.cpp`, in the loop after
`common_sampler_sample_and_accept_n` returns, before the next draft call.

**`common_speculative_free()`**: call `tr_cache_save()` if path set, then
`tr_cache_free()`.

### `common/arg.cpp`
Add CLI flags:
```
--spec-tr-k N          top-k adjacency (default 5)
--spec-tr-decay F      EMA decay 0..1 (default 0.9)
--spec-tr-cache PATH   persist adjacency matrix to file
```
Wire to `common_speculative_params`.

### `examples/lookup/lookup.cpp` (or new `examples/token-recycling/`)
Optional: standalone example mirroring the lookup example structure but using
`COMMON_SPECULATIVE_TYPE_TOKEN_RECYCLING`.

---

## Integration sequence (suggested order)

1. Write `token-recycling.h` / `token-recycling.cpp` and unit-test `update()`
   and `draft_linear()` in isolation with a fake 100-token vocabulary.
2. Add the enum value and params to `speculative.h`.
3. Wire into `speculative.cpp` (init, draft branch, update hook, free).
4. Add CLI flags in `arg.cpp`.
5. Add to CMakeLists: `common/token-recycling.cpp`.
6. Test: run `llama-lookup` with a code-editing prompt; compare acceptance rate
   to `--spec-type ngram-map-k` baseline.

---

## Edge cases and gotchas

- **vocab_size** is only available after `llama_model_load`; do NOT alloc the
  adjacency matrix in `common_speculative_init` before the model is loaded.
  Use `llama_vocab_n_tokens(llama_get_model(ctx))`.
- **The logits pointer** from `llama_get_logits_ith(ctx, 0)` is only valid
  until the next `llama_decode()` call. Copy or consume it immediately.
- **Cold-start**: for the first few tokens the adjacency matrix is empty and
  draft_linear returns nothing. That is correct — no speculation happens, no
  harm done. Cache warms up within 20-30 tokens.
- **Persist file mismatch**: if the cache file was saved with a different vocab
  size, `tr_cache_load()` must detect and reject it (compare stored
  `vocab_size` vs current model's vocab size). Silently proceed without loading
  rather than crashing.
- **Memory**: for very large vocabs (e.g., DeepSeek BPE at 100k+ tokens) k=5
  costs ~4 MB which is fine. k=10 costs ~8 MB, also fine.

---

## Acceptance rate measurement
Print at session end (same style as existing ngram stats):
```
statistics token_recycling: #calls = N, #acc = M, #gen = K
  adjacency fill rate = V / vocab_size
  adj cache path = PATH (or "memory-only")
```
