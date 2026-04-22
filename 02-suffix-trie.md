# SuffixDecoding (Simplified) — Implementation Spec for llama.cpp

## Source paper
"SuffixDecoding: A Model-Free Speculative Decoding Approach for High-Throughput
and Low-Latency LLM Inference"
Oliaro et al. (CMU/Snowflake), NeurIPS 2025 Spotlight
arXiv: 2411.04975
Production deployment: Snowflake ArcticInference + vLLM

## What it does (plain English)
SuffixDecoding builds a **suffix trie** over every token the model has ever
emitted across all sessions, stored in CPU RAM. On each draft step it finds the
longest suffix of the current generation that appears as a subtring anywhere in
that global history, then reads off the continuation tokens as draft candidates.

It is essentially a turbocharged, persistent version of llama.cpp's
`ngram-simple` — the same "find a prior occurrence of the recent tokens and
copy what came after" idea, but with:
1. A proper suffix trie instead of a linear scan.
2. A **global** history across sessions, not just the current context window.
3. Adaptive `MAX_SPEC` (draft length) proportional to how many history
   occurrences matched (more confidence → longer draft).
4. A **score** per candidate proportional to its occurrence frequency.

Reported speedup: 5.3× on AgenticSQL, 2.5× on SWE-Bench, 2.8× over EAGLE-2/3
on agentic workloads with structured/repetitive outputs.

## Linear-only simplification used here
The full paper uses tree attention to verify multiple candidates in parallel.
This spec implements only the **linear (single-chain) variant**: pick the
most-frequent continuation branch at each hop, produce a flat draft vector.
This is usable today with zero changes to llama.cpp's verification path.
Tree-attention verification can be added later.

---

## Why it fits llama.cpp right now
- It is a strict superset of `ngram-simple` / `ngram-mod` with a better backing
  data structure.
- The trie lives entirely in CPU RAM; no GPU involvement in drafting.
- The existing `--lookup-cache-dynamic` pipeline already saves/loads a binary
  ngram file; we extend it with a trie-format file.
- No new model weights, no training, no extra GPU memory.

---

## New files to create

### `common/suffix-trie.h`
```cpp
#pragma once
#include "llama.h"
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <string>

// A single node in the suffix trie.
// Children are token IDs → child node index.
// Using indices into a flat pool rather than pointers avoids fragmentation
// and makes serialization trivial.
struct st_node {
    std::unordered_map<llama_token, int32_t> children; // token → child index
    int32_t  count;   // number of times this path was traversed (for scoring)
    // No need to store the token itself — it is the edge label.
};

struct suffix_trie {
    std::vector<st_node> nodes;   // flat node pool; nodes[0] = root
    int32_t              n;       // maximum suffix length to index (default 16)
    int32_t              max_nodes; // evict when pool exceeds this (default 2M)

    void init(int n = 16, int max_nodes = 2000000);

    // Insert all suffixes of `tokens[0..len)` of length 1..n into the trie.
    // Call this once per newly-generated segment (e.g., after each assistant
    // turn completes, or after every token for online updating).
    void insert(const llama_token * tokens, int len);

    // Produce a linear draft: find the longest suffix of `context` (up to n
    // tokens) that exists in the trie, then greedily follow the highest-count
    // child for `n_draft` steps.
    // Returns empty vector if no match found.
    std::vector<llama_token> draft_linear(
        const std::vector<llama_token> & context,
        int n_draft) const;

    // Returns the depth of the longest suffix match (0 = no match).
    // Used to compute adaptive draft length: longer match → longer draft.
    int match_depth(const std::vector<llama_token> & context) const;

    // Save/load to binary file.
    bool save(const char * path) const;
    bool load(const char * path);

    // Evict least-count leaf nodes when max_nodes is exceeded.
    void evict_lru(int target_nodes);
};
```

### `common/suffix-trie.cpp`
Key implementation notes:

**`init()`**
Allocate node pool. Push one default-constructed root node (index 0).
```cpp
nodes.clear();
nodes.reserve(1024);
st_node root;
root.count = 0;
nodes.push_back(root);
this->n = n;
this->max_nodes = max_nodes;
```

**`insert(tokens, len)`**
For each suffix starting position `i` in `[max(0, len-n) .. len-1]`:
```
cur = 0  // root
for j in [i .. min(i+n, len)):
    tok = tokens[j]
    if tok not in nodes[cur].children:
        nodes[cur].children[tok] = nodes.size()
        nodes.push_back({})
    cur = nodes[cur].children[tok]
    nodes[cur].count++
if nodes.size() > max_nodes:
    evict_lru(max_nodes * 0.9)
```

**`draft_linear(context, n_draft)`**
```
// Find longest suffix of context that exists in trie
best_node = 0
best_depth = 0
for start in [max(0, ctx.size()-n) .. ctx.size()-1]:
    cur = 0
    depth = 0
    for j in [start .. ctx.size()-1]:
        tok = ctx[j]
        if tok not in nodes[cur].children: break
        cur = nodes[cur].children[tok]
        depth++
    if depth == ctx.size() - start:  // full suffix matched
        if depth > best_depth:
            best_depth = depth
            best_node = cur

if best_depth == 0: return {}

// Walk greedily for n_draft steps
result = []
cur = best_node
for i in 0..n_draft:
    if nodes[cur].children is empty: break
    // pick child with highest count
    best_tok = argmax over children of nodes[cur].children by count
    result.push_back(best_tok)
    cur = nodes[cur].children[best_tok]
return result
```

**`evict_lru(target)`**
Simple leaf pruning: collect all leaf nodes (children empty), sort by count,
delete the lowest-count ones until pool size <= target. This requires a rebuild
pass to reindex; alternatively, mark nodes as tombstones and compact lazily.
For a first implementation, just evict by removing low-count children from
their parents (walk the trie, collect (parent, child_tok, child_count) for leaf
nodes, sort by count, remove lowest until size is right).

**Save/load binary format**
```
[magic u32: 0x53554654]   // "SUFT"
[version u16: 1]
[n i32]
[node_count i32]
for each node:
    [count i32]
    [child_count u16]
    for each child:
        [token i32]
        [child_index i32]
```
For 2M nodes at average 2 children each: ~2M * (4 + 2 + 2*(4+4)) = ~36 MB.
Compress with zlib or lz4 for storage; mmap on load for fast startup.

---

## Files to modify

### `common/speculative.h`
Add enum:
```cpp
COMMON_SPECULATIVE_TYPE_SUFFIX_TRIE,
```

Add params:
```cpp
int         spec_st_n;              // max suffix length to index (default: 16)
int         spec_st_max_nodes;      // eviction threshold (default: 2000000)
int         spec_st_adaptive_max;   // max draft if deep match (default: 64)
int         spec_st_adaptive_min;   // min draft if shallow match (default: 4)
const char * spec_st_cache_path;    // persist trie to disk
bool        spec_st_online;         // update trie after each token (vs per-turn)
```

Add to `common_speculative_state`:
```cpp
suffix_trie * st;
```

### `common/speculative.cpp`

**`common_speculative_init()`**:
```cpp
case COMMON_SPECULATIVE_TYPE_SUFFIX_TRIE:
    state->st = new suffix_trie();
    state->st->init(params.spec_st_n, params.spec_st_max_nodes);
    if (params.spec_st_cache_path)
        state->st->load(params.spec_st_cache_path);
    break;
```

**`common_speculative_draft()`**:
```cpp
case COMMON_SPECULATIVE_TYPE_SUFFIX_TRIE: {
    // Adaptive draft length: deeper match → longer draft
    int depth = state->st->match_depth(inp);
    int n = params.spec_st_adaptive_min;
    if (depth >= 4) n = params.spec_st_adaptive_max;
    else if (depth >= 2) n = (params.spec_st_adaptive_min + params.spec_st_adaptive_max) / 2;
    draft = state->st->draft_linear(inp, n);
    break;
}
```

**After each accepted token** (online mode) OR **after each complete turn**
(offline mode):
```cpp
if (params.spec_st_online && type == COMMON_SPECULATIVE_TYPE_SUFFIX_TRIE) {
    // insert just the single new token appended to the running context
    state->st->insert(inp.data(), (int)inp.size());
}
```
For the per-turn (offline) variant, insert the full assistant response as a
batch at the end of each turn. Per-turn is cheaper CPU-wise; online is warmer
faster.

**`common_speculative_free()`**:
```cpp
if (state->st) {
    if (params.spec_st_cache_path)
        state->st->save(params.spec_st_cache_path);
    delete state->st;
}
```

### `common/arg.cpp`
```
--spec-type suffix-trie            enable suffix trie speculative decoding
--spec-st-n N                      max suffix length to index (default: 16)
--spec-st-max-nodes N              node pool limit (default: 2000000, ~40MB)
--spec-st-cache PATH               persist/load trie to/from file
--spec-st-online                   update trie after every token (default: per-turn)
--spec-st-adaptive-max N           max draft tokens on deep match (default: 64)
--spec-st-adaptive-min N           min draft tokens on shallow match (default: 4)
```

### `CMakeLists.txt` (common library section)
Add `common/suffix-trie.cpp`.

---

## Integration sequence (suggested order)

1. Write and test `suffix_trie` class standalone:
   - Unit test: insert 1000 tokens of Lorem Ipsum, query for a 4-gram that
     definitely appeared, verify draft is non-empty.
   - Perf test: insert 100k tokens, time a query (should be <1 ms).
2. Add enum + params + state field to `speculative.h`.
3. Wire init / draft / update / free in `speculative.cpp`.
4. Add CLI flags in `arg.cpp`.
5. Test: `llama-cli` on a long document summarization prompt. Compare acceptance
   rate vs `ngram-simple` baseline.

---

## Edge cases and gotchas

- **Thread safety**: if `llama-server` runs multiple slots sharing a single
  trie, protect `insert()` with a mutex. The draft path (`draft_linear`) is
  read-only and can run lock-free if the trie is not being concurrently
  modified. Easiest fix: separate per-slot tries (one shared static trie + one
  dynamic per-slot trie, merged or queried in priority order).
- **Node pool index stability**: using `std::vector<st_node>` means realloc
  will invalidate all child indices. Use `nodes.reserve(max_nodes)` at init so
  the vector never resizes. If it hits capacity, evict before inserting.
- **Memory on disk vs RAM**: the binary file for a 2M-node trie is ~40MB
  uncompressed. Wrap with a `FILE*` + zlib stream for `save()`; decompress into
  RAM on `load()`. Do NOT mmap a tree with pointer-like indices — reindex on
  load instead.
- **Match length vs context length**: if `n` (max suffix length) > context
  length, just use context length as the upper bound. Never read before
  `inp.data()`.
- **Empty trie at cold start**: `draft_linear` returns `{}` → verification
  skips speculative step → normal autoregressive generation. This is correct.
  No special handling needed.
- **Interaction with `--lookup-cache-static`**: if both are specified, draft
  from the suffix trie first; fall back to static ngram cache if trie returns
  nothing. Add a `spec_st_fallback_to_ngram` flag.

---

## Relationship to existing ngram types
This replaces `ngram-simple` for most use cases:
- `ngram-simple`: O(context_len) linear scan, no persistence, single context.
- `suffix-trie`: O(n) lookup regardless of history size, persists cross-session,
  global history. Better hit rate, same verification path.

`ngram-map-k` and `ngram-mod` are complementary (frequency-counting vs
occurrence-finding) — run both and take whichever produces a longer draft.

---

## Statistics output
```
statistics suffix_trie: #calls = N, #match = M (depth avg 4.2),
  #gen tokens = K, #acc tokens = J, trie nodes = 1234567
  match depth histogram: 0=12% 1=8% 2=15% 3=20% 4+=45%
```
