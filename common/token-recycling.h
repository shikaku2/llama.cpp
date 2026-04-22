#pragma once
#include "llama.h"
#include <vector>
#include <cstdint>
#include <string>

// Adjacency entry: a candidate next-token and its EMA-weighted probability score.
struct tr_entry {
    llama_token token;
    float       score;
};

// Sparse adjacency matrix built from the target model's own logits.
// adj[token_id] holds the top-k most likely successors for that token,
// sorted descending by score.
struct token_recycling_cache {
    int32_t k;
    float   decay;
    std::vector<std::vector<tr_entry>> adj; // [vocab_size][0..k]

    int64_t n_updates = 0;
    int64_t n_filled  = 0; // number of token slots with at least 1 entry

    token_recycling_cache() : k(0), decay(0.9f) {}
    token_recycling_cache(int32_t vocab_size, int32_t k, float decay);

    // Update adjacency for last_token using raw logits (length vocab_size).
    // Computes softmax probabilities internally for EMA weighting.
    void update(llama_token last_token, const float * logits, int32_t vocab_size);

    // Produce a greedy linear draft of up to n_draft tokens starting from start_token.
    // Stops early if the best-candidate score drops below min_score (0 = no threshold).
    std::vector<llama_token> draft_linear(llama_token start_token, int n_draft, float min_score = 0.0f) const;
};

// Optional: persist adjacency matrix across sessions (binary format).
// Magic: 0x54524341 ("TRCA"), version: 1
// On vocab_size mismatch, tr_cache_load returns false and leaves cache unchanged.
bool tr_cache_save(const token_recycling_cache & c, const std::string & path);
bool tr_cache_load(      token_recycling_cache & c, const std::string & path);
