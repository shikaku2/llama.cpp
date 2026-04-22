#include "suffix-automaton.h"
#include "log.h"

#include <algorithm>
#include <numeric>

void suffix_automaton::init() {
    st.clear();
    st.reserve(128);
    sam_state s0{};
    s0.len    = 0;
    s0.link   = -1;
    s0.count  = 0;
    s0.endpos = 0;
    st.push_back(s0);
    size = 1;
    last = 0;
}

void suffix_automaton::reset() {
    init();
}

void suffix_automaton::extend(llama_token tok) {
    // Fast path: transition already exists from `last`
    auto fast_it = st[last].next.find(tok);
    if (fast_it != st[last].next.end()) {
        const int q = fast_it->second;
        if (st[q].len == st[last].len + 1) {
            st[q].endpos++;
            last = q;
            return;
        }
        // Clone q
        sam_state clone_state = st[q];  // copy before potential realloc
        const int clone_idx = size++;
        st.push_back(std::move(clone_state));
        st[clone_idx].len    = st[last].len + 1;
        st[clone_idx].endpos = 0;

        int p = last;
        while (p != -1) {
            auto it = st[p].next.find(tok);
            if (it == st[p].next.end() || it->second != q) break;
            it->second = clone_idx;
            p = st[p].link;
        }
        st[q].link   = clone_idx;
        last         = clone_idx;
        st[last].endpos++;
        return;
    }

    // Standard path: create new state for the new suffix
    const int cur_idx = size++;
    st.push_back({});
    st[cur_idx].len    = st[last].len + 1;
    st[cur_idx].link   = -1;
    st[cur_idx].count  = 0;
    st[cur_idx].endpos = 1;

    int p = last;
    while (p != -1 && st[p].next.find(tok) == st[p].next.end()) {
        st[p].next[tok] = cur_idx;
        p = st[p].link;
    }

    if (p == -1) {
        st[cur_idx].link = 0;
    } else {
        const int q = st[p].next.at(tok);
        if (st[q].len == st[p].len + 1) {
            st[cur_idx].link = q;
        } else {
            sam_state clone_state2 = st[q];  // copy before push_back
            const int clone_idx2 = size++;
            st.push_back(std::move(clone_state2));
            st[clone_idx2].len    = st[p].len + 1;
            st[clone_idx2].endpos = 0;

            while (p != -1) {
                auto it = st[p].next.find(tok);
                if (it == st[p].next.end() || it->second != q) break;
                it->second = clone_idx2;
                p = st[p].link;
            }
            st[q].link      = clone_idx2;
            st[cur_idx].link = clone_idx2;
        }
    }

    last = cur_idx;
}

void suffix_automaton::compute_counts() {
    // Reset counts to direct endpos counts
    for (auto & s : st) {
        s.count = s.endpos;
    }

    // Topological sort by len descending, then propagate through suffix links
    std::vector<int> order(size);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return st[a].len > st[b].len;
    });

    for (const int v : order) {
        if (st[v].link >= 0) {
            st[st[v].link].count += st[v].count;
        }
    }
}

llama_token suffix_automaton::best_next(
        const std::vector<llama_token> & context,
        int max_query_len) const {

    if (size <= 1) return LLAMA_TOKEN_NULL;

    const int ctx_size  = (int) context.size();
    const int start_pos = std::max(0, ctx_size - max_query_len);

    int cur = 0;

    // Walk context with suffix-link backoff on miss
    for (int i = start_pos; i < ctx_size; ++i) {
        const llama_token tok = context[i];
        while (cur > 0 && st[cur].next.find(tok) == st[cur].next.end()) {
            cur = st[cur].link;
        }
        auto it = st[cur].next.find(tok);
        if (it != st[cur].next.end()) {
            cur = it->second;
        }
        // else tok never appeared; cur stays at 0
    }

    if (st[cur].next.empty()) return LLAMA_TOKEN_NULL;

    llama_token best     = LLAMA_TOKEN_NULL;
    int32_t     best_cnt = -1;
    for (const auto & kv : st[cur].next) {
        if (st[kv.second].count > best_cnt) {
            best_cnt = st[kv.second].count;
            best     = kv.first;
        }
    }
    return best;
}

std::vector<llama_token> suffix_automaton::draft_linear(
        const std::vector<llama_token> & context,
        int n_draft,
        int max_query_len) const {

    std::vector<llama_token> draft;
    draft.reserve(n_draft);

    std::vector<llama_token> ctx = context;

    for (int i = 0; i < n_draft; ++i) {
        const llama_token next = best_next(ctx, max_query_len);
        if (next == LLAMA_TOKEN_NULL) break;
        draft.push_back(next);
        ctx.push_back(next);
    }

    return draft;
}
