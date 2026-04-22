#pragma once

#include "llama.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

struct sam_state {
    int32_t len;
    int32_t link;
    int32_t count;   // propagated occurrence count (set by compute_counts())
    int32_t endpos;  // direct endpos count: 1 for non-clone primary states, 0 for clones
    std::unordered_map<llama_token, int32_t> next;
};

// Online suffix automaton over token sequences (Blumer's algorithm).
// Built incrementally via extend() — O(1) amortized per token.
// Session-only: no persistence.
struct suffix_automaton {
    std::vector<sam_state> st;  // state pool; st[0] = initial state
    int32_t last = 0;           // state representing the whole current string
    int32_t size = 0;           // number of allocated states

    void init();

    // Extend the automaton with one new token. O(1) amortized.
    void extend(llama_token tok);

    // Reset to empty (clears all states, re-initializes).
    void reset();

    // Propagate endpos counts through suffix links to compute occurrence counts.
    // Must be called before drafting after any extend() calls.
    void compute_counts();

    // Walk the last max_query_len tokens of context through the automaton (with
    // suffix-link backoff on miss) and return the highest-count transition from
    // the matched state.  Returns LLAMA_TOKEN_NULL if no continuation found.
    llama_token best_next(const std::vector<llama_token> & context,
                          int max_query_len = 32) const;

    // Produce a linear draft by iteratively calling best_next.
    std::vector<llama_token> draft_linear(
        const std::vector<llama_token> & context,
        int n_draft,
        int max_query_len = 32) const;

    int n_states() const { return size; }
};
