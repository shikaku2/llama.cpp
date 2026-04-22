# Ouroboros Phrase Harvesting — Implementation Spec for llama.cpp

## Source paper
"Ouroboros: Generating Longer Drafts Phrase by Phrase for Faster Speculative
Decoding"
Zhao et al., EMNLP 2024
arXiv: 2402.13720
GitHub: https://github.com/thunlp/Ouroboros

## What it does (plain English)
Standard speculative decoding proposes tokens one at a time and is limited by
the draft model's single-step accuracy. Ouroboros instead proposes entire
**phrases** at once — runs of tokens that previously appeared verbatim in
verified output — and chains them together to build a long draft quickly.

It harvests candidates from three sources:
1. **Lookahead phrases**: tokens already predicted by running lookahead
   (Jacobi-style) in parallel with the main draft step.
2. **Verified phrases**: token sequences that the target model already accepted
   in this session, stored in a phrase bank keyed by their first token.
3. **Historical phrases** (optional): accepted phrases from prior sessions,
   loaded from a file (same idea as `--lookup-cache-dynamic` but at phrase
   granularity).

The paper reports up to **2.8× over autoregressive** and **~1.5× over standard
speculative decoding**, especially on code generation (HumanEval, MBPP).

## Simplified scope of this spec
Ouroboros in its full form reuses the lookahead computation (Jacobi windows)
that llama.cpp's existing `--spec-type lookahead` already partially implements.
This spec implements **source 2 and 3 only** — the phrase bank — which is pure
CPU bookkeeping and needs no change to the inference graph.

Source 1 (lookahead harvest) is left as a follow-up TODO.

---

## Why this fits llama.cpp right now
- `common_speculative_draft()` already runs after each accepted segment.
- The phrase bank is a simple hash-map keyed by (first token) → list of phrases.
- The verification path is identical to standard linear speculative decoding —
  no tree attention required.
- It layers cleanly on top of any existing `--spec-type`; if the phrase bank
  hits, use the phrase; otherwise fall back to ngram or whatever is configured.

---

## New files to create

### `common/phrase-bank.h`
```cpp
#pragma once
#include "llama.h"
#include <vector>
#include <unordered_map>
#include <deque>
#include <cstdint>

// A verified phrase: a token sequence of length >= 2 that the target model
// accepted verbatim at some point.
struct phrase_entry {
    std::vector<llama_token> tokens;  // full phrase including anchor token
    uint32_t                 hits;    // number of times this phrase was used
};

// The phrase bank maps: anchor_token → list of phrases starting with that token
// Lookup: given the last accepted token, retrieve all phrases starting with it,
// then filter by requiring that phrase[1..match_len] == next tokens in context
// (for disambiguation if anchor alone is ambiguous).
struct phrase_bank {
    // Map from anchor (first token) to phrases beginning with that anchor.
    std::unordered_map<llama_token, std::vector<phrase_entry>> index;

    int max_phrases;        // max total phrases stored (default: 50000)
    int min_phrase_len;     // ignore phrases shorter than this (default: 3)
    int max_phrase_len;     // truncate phrases longer than this (default: 48)

    void init(int max_phrases = 50000,
              int min_phrase_len = 3,
              int max_phrase_len = 48);

    // Add an accepted token sequence as a phrase (and all its suffixes >= min_len).
    // Call this after each accept batch of length >= min_phrase_len.
    void add_phrase(const llama_token * tokens, int len);

    // Given the current context (last few tokens), produce a draft by:
    // 1. Finding phrases whose first token == context.back()
    // 2. Among those, preferring phrases whose next tokens match context
    //    (disambiguation / longest prefix match)
    // 3. Returning the highest-hit phrase's token sequence (minus the anchor)
    // Returns empty vector if no match found.
    std::vector<llama_token> draft_linear(
        const std::vector<llama_token> & context,
        int n_draft) const;

    // Total phrase count across all anchors.
    int size() const;

    // Evict lowest-hit phrases when max_phrases is exceeded.
    void evict();

    // Persist to / load from file.
    bool save(const char * path) const;
    bool load(const char * path);
};
```

### `common/phrase-bank.cpp`
Key implementation notes:

**`add_phrase(tokens, len)`**
```
// Add the full phrase and all sub-phrases >= min_phrase_len
for start in 0..len:
    phrase_len = len - start
    if phrase_len < min_phrase_len: continue
    phrase_len = min(phrase_len, max_phrase_len)
    anchor = tokens[start]
    phrase = tokens[start .. start+phrase_len]

    // Check if this phrase already exists under this anchor
    existing = find phrase in index[anchor] where .tokens == phrase
    if found:
        existing.hits++
    else:
        index[anchor].push_back({phrase, 1})

if size() > max_phrases:
    evict()
```

**`draft_linear(context, n_draft)`**
```
if context is empty: return {}
anchor = context.back()
if anchor not in index: return {}

candidates = index[anchor]
if candidates is empty: return {}

// Disambiguation: prefer candidates whose second token matches context
// penultimate (if available).  Among matching, pick highest hits.
best = nullptr
best_hits = -1
ctx_len = context.size()
for c in candidates:
    if c.tokens.size() < 2: continue
    // Require c.tokens[0] == anchor (always true by construction)
    // Bonus: check c.tokens[1] vs context[ctx_len-2] if available
    match_score = c.hits
    if ctx_len >= 2 and c.tokens.size() >= 2:
        if c.tokens[1] != context[ctx_len-2]:
            match_score = 0   // penalize mismatched context
    if match_score > best_hits:
        best_hits = match_score
        best = &c

if best == nullptr or best_hits == 0:
    // Fall back: just use highest-hit candidate ignoring disambiguation
    best = max(candidates by hits)

// Return the phrase tokens AFTER the anchor, up to n_draft
return best->tokens[1 .. min(best->tokens.size(), n_draft+1)]
```

**`evict()`**
```
Collect all (anchor, phrase_index, hits) triples.
Sort by hits ascending.
Remove lowest-hits entries until size() <= max_phrases * 0.8.
```

**Save/load binary format:**
```
[magic u32: 0x5048524B]   // "PHRK"
[version u16: 1]
[phrase_count i32]
for each phrase (flattened across all anchors):
    [anchor i32]
    [phrase_len u16]
    [hits u32]
    [tokens i32 * phrase_len]
```

---

## Files to modify

### `common/speculative.h`
Add enum:
```cpp
COMMON_SPECULATIVE_TYPE_PHRASE_BANK,
```

Add params:
```cpp
int          spec_pb_max_phrases;    // default 50000
int          spec_pb_min_phrase_len; // default 3
int          spec_pb_max_phrase_len; // default 48
const char * spec_pb_cache_path;     // optional persist path
bool         spec_pb_fallback_ngram; // fall back to ngram on miss (default: true)
```

Add to `common_speculative_state`:
```cpp
phrase_bank * pb;
```

### `common/speculative.cpp`

**`common_speculative_init()`**:
```cpp
case COMMON_SPECULATIVE_TYPE_PHRASE_BANK:
    state->pb = new phrase_bank();
    state->pb->init(params.spec_pb_max_phrases,
                    params.spec_pb_min_phrase_len,
                    params.spec_pb_max_phrase_len);
    if (params.spec_pb_cache_path)
        state->pb->load(params.spec_pb_cache_path);
    break;
```

**`common_speculative_draft()`**:
```cpp
case COMMON_SPECULATIVE_TYPE_PHRASE_BANK: {
    draft = state->pb->draft_linear(inp, params.n_draft_max);
    // Optional fallback to ngram if phrase bank misses:
    if (draft.empty() && params.spec_pb_fallback_ngram) {
        draft = ngram_simple_draft(inp, params.n_draft_max); // existing helper
    }
    break;
}
```

**After each accept batch** (in the verify loop, after
`common_sampler_sample_and_accept_n`): if `n_accepted >= spec_pb_min_phrase_len`:
```cpp
state->pb->add_phrase(accepted_tokens.data(), (int)accepted_tokens.size());
```
Where `accepted_tokens` is the segment of tokens just verified and accepted in
this round. This is the key feedback loop: good predictions become future
candidates.

**`common_speculative_free()`**:
```cpp
if (state->pb) {
    if (params.spec_pb_cache_path)
        state->pb->save(params.spec_pb_cache_path);
    delete state->pb;
}
```

### `common/arg.cpp`
```
--spec-type phrase-bank
--spec-pb-max-phrases N       max phrases stored in bank (default: 50000)
--spec-pb-min-phrase-len N    min phrase length to index (default: 3)
--spec-pb-max-phrase-len N    max phrase length to index (default: 48)
--spec-pb-cache PATH          persist phrase bank to file across sessions
--spec-pb-no-fallback         disable ngram fallback on phrase miss
```

---

## Integration sequence

1. Write and test `phrase_bank` standalone:
   - Insert known phrases, verify `draft_linear` returns the right continuations.
   - Test disambiguation: two phrases with same anchor, different second tokens.
   - Test eviction: insert 60k phrases, verify size stays bounded.
2. Add enum/params/state to `speculative.h`.
3. Wire init/draft/update/free in `speculative.cpp`.
4. Add CLI flags.
5. Add `common/phrase-bank.cpp` to CMakeLists.
6. Test: code-editing task (refactoring a file) where the model repeats large
   code blocks. Phrase bank should reach very high acceptance rates.

---

## Combining with llama-server multi-slot

llama-server currently runs multiple slots and speculative decoding is shared
via `--spec-type ngram-map-k` (shared rolling hash pool). For phrase-bank:
- **Option A (simple)**: one phrase bank per slot. Phrases are private.
- **Option B (better)**: one shared phrase bank, protected by a mutex.
  `draft_linear` is read-only; `add_phrase` requires a write lock.
  Hot phrases appear quickly across all slots — useful for chat where many
  users discuss the same topics.

---

## Why this particularly fits the SillyTavern use case

For Alastor roleplay sessions, the character's speech patterns are extremely
repetitive: specific exclamations, verbal tics, sign-off phrases, and
multi-sentence monologues that recur across scenes. These accepted phrases will
fill the bank quickly and give very high hit rates on subsequent turns — exactly
the "iterating over a block of text/code" scenario where `ngram-mod` already
shines, but with longer match windows and cross-session persistence.

Recommended settings for roleplay:
```
--spec-type phrase-bank
--spec-pb-min-phrase-len 4
--spec-pb-max-phrase-len 64
--spec-pb-max-phrases 100000
--spec-pb-cache /path/to/alastor-phrases.bin
```

---

## Statistics output
```
statistics phrase_bank: #calls = N, #hits = M (%.1f%%), #miss = K
  fallback_to_ngram = J,  phrases stored = P / max_phrases
  avg accepted phrase len = 8.3 tokens
```

---

## Extensions (future)

**Source 1: Lookahead harvest** — run `W` Jacobi steps in parallel with the main
draft; any Jacobi-produced n-gram (N >= min_phrase_len) that later gets accepted
by the target is added to the phrase bank. This requires access to the lookahead
buffers in `common/speculative.cpp` (they already exist for `--spec-type
lookahead`) and an extraction hook after `llama_decode()` in the lookahead
path. High ROI for code generation; adds ~5-10 lines to the lookahead branch.

**Source 3: External corpus** — tokenize a character card / lorebook / chat
history file at startup and bulk-insert all sub-sequences >= min_phrase_len into
the phrase bank. Same as `llama-lookup-create` but at phrase granularity. Can
be a one-shot offline script.
