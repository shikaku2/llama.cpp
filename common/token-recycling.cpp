#include "token-recycling.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <utility>

static constexpr uint32_t TR_MAGIC   = 0x54524341u;
static constexpr uint16_t TR_VERSION = 1;

token_recycling_cache::token_recycling_cache(int32_t vocab_size, int32_t k, float decay)
    : k(k), decay(decay), adj(vocab_size) {}

void token_recycling_cache::update(llama_token last_token, const float * logits, int32_t vocab_size) {
    if (last_token < 0 || last_token >= vocab_size || last_token >= (int32_t) adj.size()) {
        return;
    }

    n_updates++;

    // Find max logit for numerical stability.
    float max_logit = *std::max_element(logits, logits + vocab_size);

    // Compute denominator of softmax.
    double sum_exp = 0.0;
    for (int32_t i = 0; i < vocab_size; ++i) {
        sum_exp += std::exp((double)(logits[i] - max_logit));
    }

    // Find top-k tokens by logit value.
    // Use a small vector of (logit, index) pairs for partial sort.
    std::vector<std::pair<float, int32_t>> cands(vocab_size);
    for (int32_t i = 0; i < vocab_size; ++i) {
        cands[i] = {logits[i], i};
    }
    std::partial_sort(cands.begin(), cands.begin() + k, cands.end(),
        [](const std::pair<float,int32_t> & a, const std::pair<float,int32_t> & b) {
            return a.first > b.first;
        });

    auto & row = adj[last_token];

    for (int i = 0; i < k; ++i) {
        const llama_token tok = cands[i].second;
        const float prob = (float)(std::exp((double)(cands[i].first - max_logit)) / sum_exp);

        auto it = std::find_if(row.begin(), row.end(),
            [tok](const tr_entry & e) { return e.token == tok; });

        if (it != row.end()) {
            it->score = decay * it->score + (1.0f - decay) * prob;
        } else if ((int32_t) row.size() < k) {
            if (row.empty()) { n_filled++; }
            row.push_back({tok, prob});
        } else {
            auto min_it = std::min_element(row.begin(), row.end(),
                [](const tr_entry & a, const tr_entry & b) { return a.score < b.score; });
            if (prob > min_it->score) {
                min_it->token = tok;
                min_it->score = decay * min_it->score + (1.0f - decay) * prob;
            }
        }
    }

    std::sort(row.begin(), row.end(),
        [](const tr_entry & a, const tr_entry & b) { return a.score > b.score; });
}

std::vector<llama_token> token_recycling_cache::draft_linear(llama_token start_token, int n_draft, float min_score) const {
    std::vector<llama_token> result;
    result.reserve(n_draft);

    llama_token cur = start_token;
    for (int i = 0; i < n_draft; ++i) {
        if (cur < 0 || cur >= (int32_t) adj.size() || adj[cur].empty()) {
            break;
        }
        if (min_score > 0.0f && adj[cur][0].score < min_score) {
            break;
        }
        const llama_token next = adj[cur][0].token;
        result.push_back(next);
        cur = next;
    }

    return result;
}

bool tr_cache_save(const token_recycling_cache & c, const std::string & path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        LOG_ERR("%s: failed to open '%s' for writing\n", __func__, path.c_str());
        return false;
    }

    const uint32_t magic   = TR_MAGIC;
    const uint16_t version = TR_VERSION;
    const int32_t  vocab_size = (int32_t) c.adj.size();
    const int32_t  k      = c.k;
    const float    decay  = c.decay;

    f.write((const char *) &magic,      sizeof(magic));
    f.write((const char *) &version,    sizeof(version));
    f.write((const char *) &vocab_size, sizeof(vocab_size));
    f.write((const char *) &k,          sizeof(k));
    f.write((const char *) &decay,      sizeof(decay));

    for (int32_t t = 0; t < vocab_size; ++t) {
        const auto & row = c.adj[t];
        const uint8_t cnt = (uint8_t) row.size();
        f.write((const char *) &cnt, sizeof(cnt));
        for (const auto & e : row) {
            f.write((const char *) &e.token, sizeof(e.token));
            f.write((const char *) &e.score, sizeof(e.score));
        }
    }

    return f.good();
}

bool tr_cache_load(token_recycling_cache & c, const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }

    uint32_t magic   = 0;
    uint16_t version = 0;
    int32_t  vocab_size = 0;
    int32_t  k       = 0;
    float    decay   = 0.0f;

    f.read((char *) &magic,      sizeof(magic));
    f.read((char *) &version,    sizeof(version));
    f.read((char *) &vocab_size, sizeof(vocab_size));
    f.read((char *) &k,          sizeof(k));
    f.read((char *) &decay,      sizeof(decay));

    if (!f || magic != TR_MAGIC || version != TR_VERSION) {
        LOG_ERR("%s: invalid token-recycling cache file '%s'\n", __func__, path.c_str());
        return false;
    }

    if (vocab_size != (int32_t) c.adj.size()) {
        LOG_WRN("%s: vocab_size mismatch (file=%d, model=%d) – skipping load\n",
                __func__, vocab_size, (int32_t) c.adj.size());
        return false;
    }

    c.k     = k;
    c.decay = decay;
    c.n_filled = 0;

    for (int32_t t = 0; t < vocab_size; ++t) {
        uint8_t cnt = 0;
        f.read((char *) &cnt, sizeof(cnt));

        auto & row = c.adj[t];
        row.resize(cnt);
        if (cnt > 0) { c.n_filled++; }

        for (auto & e : row) {
            f.read((char *) &e.token, sizeof(e.token));
            f.read((char *) &e.score, sizeof(e.score));
        }
    }

    if (!f) {
        LOG_ERR("%s: read error in '%s'\n", __func__, path.c_str());
        c.adj.assign(vocab_size, {});
        c.n_filled = 0;
        return false;
    }

    LOG_INF("%s: loaded token-recycling cache from '%s' (vocab=%d, k=%d, filled=%lld)\n",
            __func__, path.c_str(), vocab_size, k, (long long) c.n_filled);
    return true;
}
