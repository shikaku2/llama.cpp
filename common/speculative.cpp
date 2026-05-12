#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "suffix-automaton.h"
#include "suffix-trie.h"
#include "token-recycling.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <cinttypes>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"token-recycling", COMMON_SPECULATIVE_TYPE_TOKEN_RECYCLING},
    {"token_recycling", COMMON_SPECULATIVE_TYPE_TOKEN_RECYCLING},
    {"suffix-trie",   COMMON_SPECULATIVE_TYPE_SUFFIX_TRIE},
    {"suffix_trie",   COMMON_SPECULATIVE_TYPE_SUFFIX_TRIE},
    {"sam",           COMMON_SPECULATIVE_TYPE_SAM}
};

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};
static bool common_speculative_are_compatible(
    const struct llama_context * ctx_tgt,
    const struct llama_context * ctx_dft) {
    const struct llama_model * model_tgt = llama_get_model(ctx_tgt);
    const struct llama_model * model_dft = llama_get_model(ctx_dft);

    const struct llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const struct llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_WRN("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.
    int32_t last_draft_size = 0;
    float ema_rate = 0.5f;

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted) = 0;

    virtual void observe(llama_seq_id seq_id, const llama_tokens & prompt, llama_token id_last) {
        GGML_UNUSED(seq_id);
        GGML_UNUSED(prompt);
        GGML_UNUSED(id_last);
    }

    virtual void update_logits(
            llama_context              * ctx_tgt,
            const std::vector<int32_t> & i_batch_dft,
            const llama_tokens         & ids,
            int32_t                      vocab_size) {
        GGML_UNUSED(ctx_tgt);
        GGML_UNUSED(i_batch_dft);
        GGML_UNUSED(ids);
        GGML_UNUSED(vocab_size);
    }
};

struct common_speculative_state_draft : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_state_draft(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT, n_seq)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(ctx_tgt, ctx_dft);
        LOG_DBG("%s: vocab_cmpt = %d\n", __func__, vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_ERR("%s: the target and draft vocabs are not compatible\n", __func__);

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            LOG_ERR("%s: n_seq mismatch: %d != %d\n", __func__, n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_state_draft() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        const int ret = llama_decode(ctx_dft, batch);

        if (ret != 0) {
            LOG_ERR("%s: failed to decode draft batch, ret = %d\n", __func__, ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }
};

struct common_speculative_state_eagle3 : public common_speculative_impl {
    //common_params_speculative_eagle3 params;

    common_speculative_state_eagle3(const common_params_speculative & /*params*/, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_EAGLE3, n_seq) {}

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & /*dparams*/) override {
        // TODO: implement
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_state_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_state_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , params(params.ngram_simple)
        , config(config) {}

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }
};

struct common_speculative_state_ngram_map_k : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_state_ngram_map_k(
            const common_params_speculative & params,
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, n_seq)
        , params(params.ngram_map_k) {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        common_ngram_map_accept(config[seq_id], n_accepted);
    }
};

struct common_speculative_state_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n‑gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_state_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        LOG_INF("%s: initialized ngram_mod with n_match=%d, size=%zu (%.3f MB)\n", __func__,
                this->params.n_match, mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            LOG_WRN("%s: ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted) override {
        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.5) {
                sinfo.n_low++;
                if (sinfo.n_low >= 3) {
                    if (verbose) {
                        LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }
};

struct common_speculative_state_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_state_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, params.min_size, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static,
                params.min_count, params.min_percent);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    void observe(llama_seq_id seq_id, const llama_tokens & prompt, llama_token id_last) override {
        auto & sinfo = sinfos[seq_id];
        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(id_last);

            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }
    }
};

struct common_speculative_state_token_recycling : public common_speculative_impl {
    token_recycling_cache tr_cache;
    std::string cache_path;
    bool passive;
    bool dirty = false;

    common_speculative_state_token_recycling(
            const common_params_speculative & params,
            uint32_t n_seq,
            int32_t vocab_size,
            bool passive)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_TOKEN_RECYCLING, n_seq)
        , tr_cache(vocab_size, params.token_recycling.k, params.token_recycling.decay)
        , cache_path(params.token_recycling.cache_path)
        , passive(passive) {
        if (!cache_path.empty()) {
            tr_cache_load(tr_cache, cache_path);
        }
    }

    ~common_speculative_state_token_recycling() override {
        if (!cache_path.empty()) {
            LOG_INF("%s: saving token-recycling cache to '%s'\n", __func__, cache_path.c_str());
            tr_cache_save(tr_cache, cache_path);
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        if (dirty && !cache_path.empty()) {
            tr_cache_save(tr_cache, cache_path);
            dirty = false;
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        if (passive) {
            return;
        }

        constexpr float K = 10.0f;
        const float adjusted_rate = (n_gen_drafts * ema_rate + K * 0.5f) / ((float) n_gen_drafts + K);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            const int n_draft = std::max(1, (int) std::round(dp.n_max * adjusted_rate));
            *dp.result = tr_cache.draft_linear(dp.id_last, n_draft, 0.0f);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

    void update_logits(
            llama_context              * ctx_tgt,
            const std::vector<int32_t> & i_batch_dft,
            const llama_tokens         & ids,
            int32_t                      n_vocab) override {
        for (size_t j = 0; j + 1 < ids.size() && j + 1 < i_batch_dft.size(); ++j) {
            const float * logits = llama_get_logits_ith(ctx_tgt, i_batch_dft[j + 1]);
            if (logits) {
                tr_cache.update(ids[j], logits, n_vocab);
                dirty = true;
            }
        }
    }
};

struct common_speculative_state_suffix_trie : public common_speculative_impl {
    suffix_trie trie;
    common_params_speculative_suffix_trie params;
    std::vector<size_t> n_tokens_processed;
    bool dirty = false;

    common_speculative_state_suffix_trie(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_SUFFIX_TRIE, n_seq)
        , params(params.suffix_trie)
        , n_tokens_processed(n_seq, 0) {
        trie.init(this->params.n, this->params.max_nodes);
        if (!this->params.cache_path.empty()) {
            trie.load(this->params.cache_path.c_str());
        }
    }

    ~common_speculative_state_suffix_trie() override {
        if (!params.cache_path.empty()) {
            LOG_INF("%s: saving suffix-trie to '%s'\n", __func__, params.cache_path.c_str());
            trie.save(params.cache_path.c_str());
        }
    }

    void begin(llama_seq_id seq_id, const llama_tokens & /*prompt*/) override {
        if (dirty && !params.cache_path.empty()) {
            trie.save(params.cache_path.c_str());
            dirty = false;
        }
        n_tokens_processed[seq_id] = 0;
    }

    bool process(const llama_batch & /*batch*/) override {
        return true;
    }

    void draft_one(llama_seq_id seq_id, common_speculative_draft_params & dp) {
        std::vector<llama_token> full;
        full.reserve(dp.prompt->size() + 1);
        full.insert(full.end(), dp.prompt->begin(), dp.prompt->end());
        full.push_back(dp.id_last);

        const int n = trie.n;
        const int cur_len = (int) full.size();
        auto & processed = n_tokens_processed[seq_id];
        if ((int) processed > cur_len) {
            processed = 0;
        }

        for (int p = (int) processed; p < cur_len; ++p) {
            const int seg_start = std::max(0, p - n + 1);
            const int seg_len = p - seg_start + 1;
            trie.insert(full.data() + seg_start, seg_len);
        }
        processed = (size_t) cur_len;
        dirty = true;

        const int depth = trie.match_depth(full);
        int adaptive_n = params.adaptive_min;
        if (depth >= 4) {
            adaptive_n = params.adaptive_max;
        } else if (depth >= 2) {
            adaptive_n = (params.adaptive_min + params.adaptive_max) / 2;
        }
        if (dp.n_max > 0) {
            adaptive_n = std::min(adaptive_n, dp.n_max);
        }

        *dp.result = trie.draft_linear(full, adaptive_n);
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (dp.drafting) {
                draft_one(seq_id, dp);
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

};

struct common_speculative_state_sam : public common_speculative_impl {
    suffix_automaton sam;
    common_params_speculative_sam params;
    std::vector<size_t> n_tokens_extended;

    common_speculative_state_sam(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_SAM, n_seq)
        , params(params.sam)
        , n_tokens_extended(n_seq, 0) {
        sam.init();
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (prompt.size() < n_tokens_extended[seq_id]) {
            sam.reset();
            n_tokens_extended[seq_id] = 0;
            for (const auto tok : prompt) {
                sam.extend(tok);
            }
            n_tokens_extended[seq_id] = prompt.size();
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        return true;
    }

    void draft_one(llama_seq_id seq_id, common_speculative_draft_params & dp) {
        const int cur_len = (int) dp.prompt->size();
        auto & extended = n_tokens_extended[seq_id];
        if ((int) extended > cur_len + 1) {
            sam.reset();
            extended = 0;
        }

        for (int i = (int) extended; i < cur_len; ++i) {
            sam.extend((*dp.prompt)[i]);
        }
        sam.extend(dp.id_last);
        extended = (size_t) (cur_len + 1);

        sam.compute_counts();

        std::vector<llama_token> full;
        full.reserve(dp.prompt->size() + 1);
        full.insert(full.end(), dp.prompt->begin(), dp.prompt->end());
        full.push_back(dp.id_last);

        const int n_draft = dp.n_max > 0 ? dp.n_max : 16;
        *dp.result = sam.draft_linear(full, n_draft, params.max_query);
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (dp.drafting) {
                draft_one(seq_id, dp);
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/) override {
        // noop
    }

};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_state_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = (uint16_t) config.params.draft.n_max;

    bool save_static = false;
    bool save_dynamic = !path_dynamic.empty();

    common_speculative_state_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

static std::vector<common_speculative_type> common_speculative_types_from_str(const std::string & str) {
    std::vector<common_speculative_type> result;
    std::string token;
    std::istringstream ss(str);
    while (std::getline(ss, token, ',')) {
        if (token == "none") {
            return {};        }
        const auto t = common_speculative_type_from_name(token);
        if (t == COMMON_SPECULATIVE_TYPE_COUNT) {
            throw std::invalid_argument("unknown speculative decoding type: " + token);
        }
        for (const auto & existing : result) {
            if (existing == t) {
                throw std::invalid_argument("duplicate speculative decoding types are not allowed: " + token);
            }
        }
        result.push_back(t);
    }
    return result;
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT:         return "draft";
        case COMMON_SPECULATIVE_TYPE_EAGLE3:        return "eagle3";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        case COMMON_SPECULATIVE_TYPE_TOKEN_RECYCLING: return "token-recycling";
        case COMMON_SPECULATIVE_TYPE_SUFFIX_TRIE:   return "suffix-trie";
        case COMMON_SPECULATIVE_TYPE_SAM:           return "sam";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);
    {
        bool has_draft = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT));
        bool has_draft_model = !params.draft.mparams.path.empty();

        // bool has_mtp = false; // TODO: add MTP here
        bool has_draft_eagle3 = has_draft_model && params.draft.eagle3;

        bool has_ngram_cache   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_CACHE));
        bool has_ngram_simple  = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE));
        bool has_ngram_map_k   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K));
        bool has_ngram_map_k4v = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V));
        bool has_ngram_mod     = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MOD));
        bool has_token_recycling = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_TOKEN_RECYCLING));
        bool has_suffix_trie   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_SUFFIX_TRIE));
        bool has_sam           = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_SAM));

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 11);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_token_recycling) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_TOKEN_RECYCLING, params));
        }
        if (has_suffix_trie) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_SUFFIX_TRIE, params));
        }
        if (has_sam) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_SAM, params));
        }
        if (has_draft) {
            if (!has_draft_model) {
                LOG_WRN("%s: draft model is not specified - cannot use 'draft' type\n", __func__);
                has_draft = false;
            }
        } else if (has_draft_model) {
            LOG_WRN("%s: draft model is specified but 'draft' speculative type is not explicitly enabled - enabling it\n", __func__);
            has_draft = true;
        }

        if (has_draft) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT, params));
        }
        // TODO: add MTP here
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_EAGLE3, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                impls.push_back(std::make_unique<common_speculative_state_draft>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_state_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_state_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_state_ngram_map_k>(
                            config.params, get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_state_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_state_ngram_cache>(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_TOKEN_RECYCLING: {
                const int32_t vocab_size = llama_vocab_n_tokens(
                        llama_model_get_vocab(llama_get_model(params.draft.ctx_tgt)));
                impls.push_back(std::make_unique<common_speculative_state_token_recycling>(
                            config.params, n_seq, vocab_size, false));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_SUFFIX_TRIE: {
                impls.push_back(std::make_unique<common_speculative_state_suffix_trie>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_SAM: {
                impls.push_back(std::make_unique<common_speculative_state_sam>(config.params, n_seq));
                break;
            }
            default:
                break;
        }
    }

    if (!params.token_recycling.cache_path.empty() &&
            !(enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_TOKEN_RECYCLING))) {
        const int32_t vocab_size = llama_vocab_n_tokens(
                llama_model_get_vocab(llama_get_model(params.draft.ctx_tgt)));
        LOG_INF("%s: adding passive token-recycling observer (cache: %s)\n",
                __func__, params.token_recycling.cache_path.c_str());
        impls.push_back(std::make_unique<common_speculative_state_token_recycling>(
                    params, n_seq, vocab_size, true));
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        LOG_DBG("%s: truncating draft to %d tokens\n", __func__, dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->last_draft_size = (int32_t) result.size();
                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();

                    for (auto & other : spec->impls) {
                        if (other.get() != impl.get()) {
                            other->observe(seq_id, *dp.prompt, dp.id_last);
                        }
                    }
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(
        common_speculative         * spec,
        llama_seq_id                 seq_id,
        uint16_t                     n_accepted,
        llama_context              * ctx_tgt,
        const std::vector<int32_t> & i_batch_dft,
        const llama_tokens         & ids) {
    if (spec == nullptr) {
        return;
    }

    if (ctx_tgt && !i_batch_dft.empty() && !ids.empty()) {
        const int32_t n_vocab = llama_vocab_n_tokens(
                llama_model_get_vocab(llama_get_model(ctx_tgt)));
        for (auto & each : spec->impls) {
            each->update_logits(ctx_tgt, i_batch_dft, ids, n_vocab);
        }
    }

    common_speculative_impl * impl = spec->impl_last[seq_id];
    spec->impl_last[seq_id] = nullptr;

    if (impl == nullptr) {
        return;
    }


    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted);
        impl->n_call_accept++;
    }

    if (impl->last_draft_size > 0) {
        const float rate = (float) n_accepted / (float) impl->last_draft_size;
        impl->ema_rate = 0.9f * impl->ema_rate + 0.1f * rate;
    }

    auto ema_score = [](const std::unique_ptr<common_speculative_impl> & s) {
        constexpr float K = 10.0f;
        return (s->n_gen_drafts * s->ema_rate + K * 0.5f) / ((float) s->n_gen_drafts + K);
    };
    std::stable_sort(spec->impls.begin(), spec->impls.end(),
            [&ema_score](const std::unique_ptr<common_speculative_impl> & a,
                         const std::unique_ptr<common_speculative_impl> & b) {
                return ema_score(a) > ema_score(b);
            });
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());

        if (impl->type == COMMON_SPECULATIVE_TYPE_TOKEN_RECYCLING) {
            const auto * tr = static_cast<const common_speculative_state_token_recycling *>(impl.get());
            const int32_t vocab = (int32_t) tr->tr_cache.adj.size();
            LOG_INF("  token-recycling adjacency: filled=%lld/%d (%.1f%%), updates=%lld, cache=%s\n",
                    (long long) tr->tr_cache.n_filled, vocab,
                    vocab > 0 ? 100.0 * (double) tr->tr_cache.n_filled / vocab : 0.0,
                    (long long) tr->tr_cache.n_updates,
                    tr->cache_path.empty() ? "memory-only" : tr->cache_path.c_str());
        }
        if (impl->type == COMMON_SPECULATIVE_TYPE_SUFFIX_TRIE) {
            const auto * st = static_cast<const common_speculative_state_suffix_trie *>(impl.get());
            LOG_INF("  suffix-trie: nodes=%d/%d, cache=%s\n",
                    st->trie.n_nodes(), st->params.max_nodes,
                    st->params.cache_path.empty() ? "memory-only" : st->params.cache_path.c_str());
        }
        if (impl->type == COMMON_SPECULATIVE_TYPE_SAM) {
            const auto * s = static_cast<const common_speculative_state_sam *>(impl.get());
            size_t n_tokens = 0;
            for (const auto n : s->n_tokens_extended) {
                n_tokens = std::max(n_tokens, n);
            }
            LOG_INF("  sam: states=%d, max_tokens_extended=%zu\n",
                    s->sam.n_states(), n_tokens);
        }
    }
}

std::vector<common_speculative_impl_stats> common_speculative_get_impl_stats(const common_speculative * spec) {
    std::vector<common_speculative_impl_stats> result;
    if (spec == nullptr) {
        return result;
    }
    for (const auto & impl : spec->impls) {
        common_speculative_impl_stats s;
        s.type_name    = common_speculative_type_to_str(impl->type);
        s.n_gen_tokens = impl->n_gen_tokens;
        s.n_acc_tokens = impl->n_acc_tokens;
        s.ema_rate     = impl->ema_rate;
        result.push_back(s);
    }
    // Sort by type_name so CSV column order is stable regardless of EMA ranking.
    std::sort(result.begin(), result.end(),
        [](const common_speculative_impl_stats & a, const common_speculative_impl_stats & b) {
            return a.type_name < b.type_name;
        });
    return result;
}
