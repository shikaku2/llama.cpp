# SAM Decoding (Suffix Automaton) — Implementation Spec for llama.cpp

## Source paper
"SAM Decoding: Speculative Decoding via Suffix Automaton"
Hu et al., 2024
arXiv: 2411.10666

## What it does (plain English)
A **suffix automaton (SAM)** is a compact finite-state machine that represents
all suffixes of a string — it can be built in O(N) time and space and answers
"does this substring occur in my text, and where does it continue?" in O(M) time
where M is the query length.

SAM Decoding builds a suffix automaton over the current context window (and
optionally over session history). On each draft step it feeds the last few
tokens into the automaton as a query, reaches the corresponding state, and reads
out the possible continuations (outgoing transitions from that state) as draft
candidates.

This is similar to the existing `ngram-simple` (scan for last n-gram and copy
what follows) but:
- O(M) query time instead of O(context_len * n) linear scan.
- The automaton is updated incrementally in O(1) amortized per token.
- It handles all suffix lengths at once — there is no fixed `n` parameter.
- It naturally produces a **set of candidate next tokens** (the outgoing
  transitions) which can be used as a draft tree or pruned to a single chain.

Reported speedup: **1.9–2.27×** standalone; **2.27× stacked with Token
Recycling** and **2.49× stacked with EAGLE-2**.

## Simplified scope of this spec
This spec implements the **linear greedy variant**: at each draft step, take the
most-frequent transition from the current SAM state, producing a single draft
token. This needs no tree attention.

The SAM is built **over the current session's accepted tokens only** (not a
static corpus). It is rebuilt or extended incrementally each turn.

---

## Why this fits llama.cpp right now
- The SAM lives entirely in CPU RAM.
- It is a strictly better O(M) replacement for the linear scan inside
  `ngram-simple`.
- Incremental extension: each newly accepted token extends the SAM in O(1)
  amortized — no full rebuild required.
- No training, no external files needed (though persistence is optional).

---

## Suffix Automaton background (implementation-level)

A SAM has nodes (states) and transitions (edges labeled by token). Each state
represents an equivalence class of substrings. Key fields per state:
- `len`: length of the longest substring in this equivalence class.
- `link`: suffix link (pointer to another state representing the longest proper
  suffix of this class).
- `next`: map from token → next state (the transitions).
- `count`: number of times this state's substrings appear in the text.

The standard SAM construction is Blumer's algorithm (also called the "online
SAM construction"). It processes one character (token) at a time and is O(1)
amortized. Total states <= 2*N, total transitions <= 3*N for a string of
length N. For a 32k-token context window: at most 64k states, 96k transitions
— around 8-10 MB of RAM.

Reference implementation (dozens of line of C++): see CP-algorithms.com/string/suffix-automaton.html

---

## New files to create

### `common/suffix-automaton.h`
```cpp
#pragma once
#include "llama.h"
#include <vector>
#include <unordered_map>
#include <cstdint>

struct sam_state {
    int32_t len;
    int32_t link;
    int32_t count;  // occurrence count (set during "extend" traversal)
    std::unordered_map<llama_token, int32_t> next;  // token → state index
};

struct suffix_automaton {
    std::vector<sam_state> st;  // state pool; st[0] = initial state
    int32_t last;               // index of state representing the whole string
    int32_t size;               // number of states allocated

    void init();

    // Extend the automaton with one new token.  O(1) amortized.
    void extend(llama_token tok);

    // Reset the automaton (clears all states).
    void reset();

    // Given the last `max_query_len` tokens of the context, walk the automaton
    // as far as possible, then return the most-frequent outgoing transition
    // token from the matched state.
    // Returns LLAMA_TOKEN_NULL if no match or no continuation.
    llama_token best_next(const std::vector<llama_token> & context,
                           int max_query_len = 32) const;

    // Produce a linear draft of length n_draft using greedy best_next walks.
    // After each draft token, advances the automaton state along that token
    // (if the transition exists) or returns early.
    std::vector<llama_token> draft_linear(
        const std::vector<llama_token> & context,
        int n_draft,
        int max_query_len = 32) const;

    // After all tokens in the session have been extended, compute occurrence
    // counts by a DFS/topological traversal of suffix links.
    // Call once before drafting (or incrementally after bulk extends).
    void compute_counts();

    // Current state count
    int n_states() const { return size; }
};
```

### `common/suffix-automaton.cpp`

**`init()`**:
```cpp
st.clear();
st.reserve(128);
sam_state s0;
s0.len = 0; s0.link = -1; s0.count = 0;
st.push_back(s0);
size = 1; last = 0;
```

**`extend(tok)`** — standard SAM extend algorithm (Blumer):
```cpp
// Check if transition already exists from `last`
if (st[last].next.count(tok)) {
    int q = st[last].next[tok];
    if (st[q].len == st[last].len + 1) {
        last = q;
        st[last].count++;
        return;
    }
    int clone = size++;
    st.push_back(st[q]);
    st[clone].len = st[last].len + 1;
    st[clone].count = st[last].count + 1;
    int p = last;
    while (p != -1 && st[p].next.count(tok) && st[p].next[tok] == q) {
        st[p].next[tok] = clone;
        p = st[p].link;
    }
    st[q].link = clone;
    last = clone;
    return;
}
// Normal extend path
int cur = size++;
st.push_back({});
st[cur].len = st[last].len + 1;
st[cur].count = 1;
int p = last;
while (p != -1 && !st[p].next.count(tok)) {
    st[p].next[tok] = cur;
    p = st[p].link;
}
if (p == -1) {
    st[cur].link = 0;
} else {
    int q = st[p].next[tok];
    if (st[q].len == st[p].len + 1) {
        st[cur].link = q;
    } else {
        int clone = size++;
        st.push_back(st[q]);
        st[clone].len = st[p].len + 1;
        while (p != -1 && st[p].next.count(tok) && st[p].next[tok] == q) {
            st[p].next[tok] = clone;
            p = st[p].link;
        }
        st[q].link = clone;
        st[cur].link = clone;
    }
}
last = cur;
if ((int)st.size() > size) st.resize(size);
```

**`compute_counts()`**:
Topological sort of states by `len` descending, then propagate counts up
through `link` edges:
```cpp
// Sort by len descending
vector<int> order(size);
iota(order.begin(), order.end(), 0);
sort(order.begin(), order.end(), [&](int a, int b) {
    return st[a].len > st[b].len;
});
for (int v : order) {
    if (st[v].link >= 0)
        st[st[v].link].count += st[v].count;
}
```

**`best_next(context, max_query_len)`**:
```cpp
int cur = 0;
int start = max(0, (int)context.size() - max_query_len);
for (int i = start; i < (int)context.size(); i++) {
    auto it = st[cur].next.find(context[i]);
    if (it == st[cur].next.end()) {
        // Back off: reset to initial and try from next position
        cur = 0;
        i++;  // skip this token and retry from initial
        if (i < (int)context.size()) {
            it = st[cur].next.find(context[i]);
            if (it == st[cur].next.end()) continue;
            cur = it->second;
        }
    } else {
        cur = it->second;
    }
}
// cur is now the state matching the longest suffix of context.
// Pick the child transition with the highest count.
if (st[cur].next.empty()) return LLAMA_TOKEN_NULL;
llama_token best_tok = LLAMA_TOKEN_NULL;
int32_t best_count = -1;
for (auto & [tok, state_idx] : st[cur].next) {
    if (st[state_idx].count > best_count) {
        best_count = st[state_idx].count;
        best_tok = tok;
    }
}
return best_tok;
```

**`draft_linear(context, n_draft, max_query_len)`**:
```cpp
vector<llama_token> draft;
auto ctx_copy = context;
for (int i = 0; i < n_draft; i++) {
    llama_token next = best_next(ctx_copy, max_query_len);
    if (next == LLAMA_TOKEN_NULL) break;
    draft.push_back(next);
    ctx_copy.push_back(next);
}
return draft;
```

---

## Files to modify

### `common/speculative.h`
Add enum:
```cpp
COMMON_SPECULATIVE_TYPE_SAM,
```

Add params:
```cpp
int  spec_sam_max_query;   // max suffix length for query (default: 32)
bool spec_sam_recompute;   // recompute counts after each accepted batch (default: true)
```

Add to state:
```cpp
suffix_automaton * sam;
bool sam_dirty;  // needs compute_counts() before next draft
```

### `common/speculative.cpp`

**Init**:
```cpp
case COMMON_SPECULATIVE_TYPE_SAM:
    state->sam = new suffix_automaton();
    state->sam->init();
    state->sam_dirty = false;
    break;
```

**After each accepted token** (feed into the automaton):
```cpp
for (llama_token tok : accepted_tokens) {
    state->sam->extend(tok);
}
state->sam_dirty = true;
```

**Draft**:
```cpp
case COMMON_SPECULATIVE_TYPE_SAM: {
    if (state->sam_dirty && params.spec_sam_recompute) {
        state->sam->compute_counts();
        state->sam_dirty = false;
    }
    draft = state->sam->draft_linear(inp, params.n_draft_max,
                                     params.spec_sam_max_query);
    break;
}
```

**Free**:
```cpp
delete state->sam;
```

### `common/arg.cpp`
```
--spec-type sam
--spec-sam-max-query N    max suffix length for query (default: 32)
--spec-sam-no-recompute   skip count recomputation (faster but less accurate)
```

---

## Integration sequence

1. Implement `suffix_automaton` standalone. Unit test with a known string:
   extend "abcabcabc" character by character, verify that querying "bc"
   returns "a" as the best continuation.
2. Adapt for llama_token (int32_t) instead of char. Verify extend handles
   any token ID including 0 and negative values.
3. Add enum, params, state to speculative.h.
4. Wire init/extend/draft/free.
5. Add CLI flags.
6. Test: llama-cli on a repetitive generation (reasoning model with long think
   blocks). SAM should achieve similar acceptance to ngram-mod but with faster
   lookup.

---

## Performance notes

- **Memory**: for a 32k-token context, up to 64k states × ~64 bytes per state
  (unordered_map overhead is significant) ≈ 4-8 MB. Acceptable.
- **compute_counts()**: O(N) with sort, called at most once per draft step.
  For 64k states this is ~1 ms. Fine.
- **extend()**: O(1) amortized. Called once per accepted token. Negligible.
- **draft_linear()**: O(n_draft * max_query) ≈ O(n_draft * 32) calls to
  unordered_map lookups. Fast in practice.
- **Context reset**: if `--ctx-size` is hit and the context is truncated,
  `reset()` and rebuild from the surviving tokens. Rebuilding 32k tokens
  takes ~10 ms. This is acceptable for a rare event.

---

## Comparison to suffix-trie (spec 02)

| Feature              | Suffix Trie (02)        | Suffix Automaton (this)  |
|----------------------|-------------------------|--------------------------|
| Build time           | O(N*n) per insert       | O(N) total               |
| Query time           | O(n) per query          | O(M) per query           |
| Memory               | ~40 MB (2M nodes)       | ~8 MB (64k states)       |
| Cross-session persist| Yes (file)              | No (rebuild each session)|
| Backoff handling     | Manual (try shorter)    | Automatic (suffix links) |
| Count accuracy       | Exact                   | Exact after propagation  |

For **within-session** drafting (current context only), the suffix automaton is
strictly better in time and memory. For **cross-session** drafting (global
history), the suffix trie with persistence wins. Run both and take whichever
produces a non-empty draft, or combine as: SAM for recency, suffix-trie for
global history.
