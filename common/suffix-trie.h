#pragma once

#include "llama.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// A node in the flat-pool suffix trie.
// Children are stored as token → child index into the nodes vector.
struct st_node {
    std::unordered_map<llama_token, int32_t> children;
    int32_t count = 0;
};

// Suffix trie over token sequences.  Stores all substrings of length ≤ n seen
// across all inserted sequences.  Serialises to/from a binary file so history
// accumulates across server restarts.
//
// Thread safety: external locking required if insert() and draft_linear() are
// called concurrently from different threads.
struct suffix_trie {
    std::vector<st_node> nodes;  // nodes[0] = root
    int32_t n         = 16;
    int32_t max_nodes = 2000000;

    void init(int n = 16, int max_nodes = 2000000);

    // Insert all suffixes of tokens[0..len) that start within the last n
    // positions.  Call once per new position in the token stream (see state
    // wrapper in speculative.cpp for the correct sliding-window usage).
    void insert(const llama_token * tokens, int len);

    // Returns the depth of the longest suffix of context that exists in the
    // trie (0 = no match).  Used to compute adaptive draft length.
    int match_depth(const std::vector<llama_token> & context) const;

    // Find the longest suffix of context that exists in the trie, then
    // greedily follow the highest-count child for up to n_draft steps.
    std::vector<llama_token> draft_linear(
        const std::vector<llama_token> & context,
        int n_draft) const;

    // Persist to / load from a binary file.
    bool save(const char * path) const;
    bool load(const char * path);

    // Evict lowest-count leaf nodes until pool size <= target_nodes.
    void evict_lru(int target_nodes);

    int n_nodes() const { return (int) nodes.size(); }
};
