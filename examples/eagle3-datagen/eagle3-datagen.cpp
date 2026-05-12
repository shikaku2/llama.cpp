#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::ordered_json;

struct datagen_params {
    std::string input_path = "alastor_datagen.jsonl";
    std::string output_dir = "alastor_hidden_states";
    std::vector<int32_t> layers = { 2, 15, 27 };
    int32_t max_seq_len = 8192;
    std::string char_filter;
};

static void npy_header(FILE * f, const char * dtype, const std::vector<int64_t> & shape) {
    std::string h = "{'descr': '";
    h += dtype;
    h += "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        h += std::to_string(shape[i]);
        if (i + 1 < shape.size()) {
            h += ", ";
        }
    }
    if (shape.size() == 1) {
        h += ",";
    }
    h += "), }";

    const size_t hlen = 10 + h.size() + 1;
    h += std::string((64 - hlen % 64) % 64, ' ');
    h += '\n';

    const uint8_t magic[] = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 };
    fwrite(magic, 1, sizeof(magic), f);
    const uint16_t hl = (uint16_t) h.size();
    fwrite(&hl, sizeof(hl), 1, f);
    fwrite(h.data(), 1, h.size(), f);
}

template <typename T>
static void write_npy(const std::string & path, const char * dtype, const T * data, const std::vector<int64_t> & shape) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        throw std::runtime_error("failed to open output file: " + path);
    }

    npy_header(f, dtype, shape);

    int64_t n = 1;
    for (int64_t dim : shape) {
        n *= dim;
    }
    fwrite(data, sizeof(T), (size_t) n, f);
    fclose(f);
}

static void parse_layers(datagen_params & params, const std::string & value) {
    params.layers.clear();
    std::stringstream ss(value);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) {
            params.layers.push_back(std::stoi(tok));
        }
    }
}

static bool parse_datagen_args(int argc, char ** argv, datagen_params & dparams, std::vector<char *> & llama_argv) {
    llama_argv.clear();
    llama_argv.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char * name) {
            if (i + 1 >= argc) {
                LOG_ERR("%s requires a value\n", name);
                return false;
            }
            return true;
        };

        if (arg == "--input") {
            if (!need_value("--input")) {
                return false;
            }
            dparams.input_path = argv[++i];
        } else if (arg == "--output-dir") {
            if (!need_value("--output-dir")) {
                return false;
            }
            dparams.output_dir = argv[++i];
        } else if (arg == "--layers") {
            if (!need_value("--layers")) {
                return false;
            }
            parse_layers(dparams, argv[++i]);
        } else if (arg == "--max-seq-len") {
            if (!need_value("--max-seq-len")) {
                return false;
            }
            dparams.max_seq_len = std::stoi(argv[++i]);
        } else if (arg == "--char") {
            if (!need_value("--char")) {
                return false;
            }
            dparams.char_filter = argv[++i];
        } else {
            llama_argv.push_back(argv[i]);
        }
    }

    return !dparams.layers.empty() && dparams.max_seq_len > 0;
}

static std::string apply_chat_template(
        const std::string & tmpl,
        const std::vector<llama_chat_message> & msgs,
        bool add_assistant) {
    int32_t cap = 4096;
    std::vector<char> buf(cap);

    while (true) {
        const int32_t n = llama_chat_apply_template(
                tmpl.empty() ? nullptr : tmpl.c_str(),
                msgs.data(),
                msgs.size(),
                add_assistant,
                buf.data(),
                (int32_t) buf.size());

        if (n < 0) {
            break;
        }
        if (n < (int32_t) buf.size()) {
            return std::string(buf.data(), n);
        }

        cap = std::max(cap * 2, n + 1);
        buf.resize(cap);
    }

    if (tmpl.find("<|turn>") == std::string::npos) {
        throw std::runtime_error("failed to apply chat template");
    }

    std::string out;
    for (const llama_chat_message & msg : msgs) {
        std::string role = msg.role ? msg.role : "";
        std::string content = msg.content ? msg.content : "";
        if (role == "assistant") {
            role = "model";
        }
        if (role == "system" || role == "developer") {
            out += "<|turn>system\n" + content + "<turn|>\n";
        } else if (role == "user" || role == "model") {
            out += "<|turn>" + role + "\n" + content + "<turn|>\n";
        }
    }
    if (add_assistant) {
        out += "<|turn>model\n";
    }
    return out;
}

static llama_batch make_full_output_batch(const std::vector<llama_token> & tokens) {
    llama_batch batch = llama_batch_init((int32_t) tokens.size(), 0, 1);
    for (int32_t i = 0; i < (int32_t) tokens.size(); ++i) {
        common_batch_add(batch, tokens[i], i, { 0 }, true);
    }
    return batch;
}

int main(int argc, char ** argv) {
    datagen_params dparams;
    std::vector<char *> llama_argv;
    if (!parse_datagen_args(argc, argv, dparams, llama_argv)) {
        LOG_ERR("usage: %s -m MODEL [llama args] --input data.jsonl --output-dir out --layers 2,15,27 --max-seq-len 8192\n", argv[0]);
        return 1;
    }

    common_params params;
    if (!common_params_parse((int) llama_argv.size(), llama_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    params.n_ctx = dparams.max_seq_len;
    params.n_batch = std::max(params.n_batch, dparams.max_seq_len);
    params.n_ubatch = std::max(params.n_ubatch, dparams.max_seq_len);

    auto llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * ctx = llama_init->context();
    if (!model || !ctx) {
        return 1;
    }

    llama_enable_hidden_state_extraction(ctx, dparams.layers.data(), (int32_t) dparams.layers.size());

    fs::create_directories(dparams.output_dir);

    const char * tmpl_ptr = llama_model_chat_template(model, nullptr);
    const std::string tmpl = tmpl_ptr ? tmpl_ptr : "";
    if (tmpl.empty()) {
        LOG_WRN("no chat template in model metadata; using llama_chat_apply_template default\n");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t vocab_size = llama_vocab_n_tokens(vocab);
    std::vector<int64_t> token_freq(vocab_size, 0);

    std::ifstream input(dparams.input_path);
    if (!input) {
        LOG_ERR("cannot open input: %s\n", dparams.input_path.c_str());
        return 1;
    }

    std::string line;
    int32_t n_written = 0;
    int32_t n_skipped = 0;

    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }

        json item = json::parse(line, nullptr, false);
        if (item.is_discarded() || !item.contains("messages") || !item["messages"].is_array()) {
            ++n_skipped;
            continue;
        }

        std::vector<std::string> roles;
        std::vector<std::string> contents;
        std::vector<llama_chat_message> msgs;
        bool has_user = false;
        bool char_match = dparams.char_filter.empty();

        for (const auto & m : item["messages"]) {
            if (!m.contains("role") || !m.contains("content")) {
                continue;
            }
            roles.push_back(m["role"].get<std::string>());
            contents.push_back(m["content"].get<std::string>());
            has_user = has_user || roles.back() == "user";
            char_match = char_match || contents.back().find(dparams.char_filter) != std::string::npos;
            msgs.push_back({ roles.back().c_str(), contents.back().c_str() });
        }

        if (msgs.empty() || !has_user || !char_match || std::string(msgs.back().role) != "assistant") {
            ++n_skipped;
            continue;
        }

        std::string full_text;
        std::string prefix_text;
        try {
            full_text = apply_chat_template(tmpl, msgs, false);
            std::vector<llama_chat_message> prefix_msgs(msgs.begin(), msgs.end() - 1);
            prefix_text = apply_chat_template(tmpl, prefix_msgs, true);
        } catch (const std::exception & err) {
            LOG_WRN("sample %d template failed: %s\n", n_written + n_skipped, err.what());
            ++n_skipped;
            continue;
        }

        std::vector<llama_token> tokens = common_tokenize(vocab, full_text, true, true);
        if (tokens.empty() || (int32_t) tokens.size() > dparams.max_seq_len) {
            ++n_skipped;
            continue;
        }

        std::vector<llama_token> prefix_tokens = common_tokenize(vocab, prefix_text, true, true);
        const int32_t prefix_len = std::min((int32_t) prefix_tokens.size(), (int32_t) tokens.size());

        std::vector<int32_t> loss_mask(tokens.size(), 0);
        for (int32_t i = prefix_len; i < (int32_t) tokens.size(); ++i) {
            loss_mask[i] = 1;
            if (tokens[i] >= 0 && tokens[i] < vocab_size) {
                token_freq[tokens[i]]++;
            }
        }

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = make_full_output_batch(tokens);
        const int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (ret != 0) {
            ++n_skipped;
            continue;
        }

        const fs::path sample_dir = fs::path(dparams.output_dir) / string_format("sample_%06d", n_written);
        fs::create_directories(sample_dir);

        bool ok = true;
        for (int32_t layer : dparams.layers) {
            int32_t n_tokens = 0;
            int32_t n_embd = 0;
            const float * hs = llama_get_layer_hidden_states(ctx, layer, &n_tokens, &n_embd);
            if (!hs || n_tokens != (int32_t) tokens.size()) {
                LOG_WRN("layer %d extraction failed for sample %d: got %d tokens, expected %zu\n",
                        layer, n_written, n_tokens, tokens.size());
                ok = false;
                break;
            }

            write_npy((sample_dir / string_format("hidden_states_layer%d.npy", layer)).string(),
                    "<f4", hs, { n_tokens, n_embd });
        }

        if (!ok) {
            fs::remove_all(sample_dir);
            ++n_skipped;
            continue;
        }

        write_npy((sample_dir / "input_ids.npy").string(), "<i4", tokens.data(), { (int64_t) tokens.size() });
        write_npy((sample_dir / "loss_mask.npy").string(), "<i4", loss_mask.data(), { (int64_t) loss_mask.size() });

        ++n_written;
        if (n_written % 50 == 0) {
            LOG_INF("%d written, %d skipped\n", n_written, n_skipped);
        }
    }

    write_npy((fs::path(dparams.output_dir) / "token_freq.npy").string(), "<i8", token_freq.data(), { vocab_size });

    json cfg;
    cfg["n_samples"] = n_written;
    cfg["layers"] = dparams.layers;
    cfg["vocab_size"] = vocab_size;
    cfg["generator"] = "llama-eagle3-datagen";
    cfg["model"] = params.model.path;
    std::ofstream(fs::path(dparams.output_dir) / "data_config.json") << cfg.dump(2) << '\n';

    LOG_INF("done: %d samples written, %d skipped\n", n_written, n_skipped);
    return 0;
}
