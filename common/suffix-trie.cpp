#include "suffix-trie.h"
#include "log.h"

#include <algorithm>
#include <fstream>
#include <numeric>

static constexpr uint32_t ST_MAGIC   = 0x53554654u;
static constexpr uint16_t ST_VERSION = 1;

void suffix_trie::init(int n_, int max_nodes_) {
    nodes.clear();
    nodes.reserve(std::min(1024, max_nodes_));
    st_node root;
    root.count = 0;
    nodes.push_back(root);
    n         = n_;
    max_nodes = max_nodes_;
}

void suffix_trie::insert(const llama_token * tokens, int len) {
    if (len <= 0 || nodes.empty()) return;

    const int i_start = (len > n) ? len - n : 0;

    for (int i = i_start; i < len; ++i) {
        int cur = 0;
        const int end = std::min(i + n, len);
        for (int j = i; j < end; ++j) {
            const llama_token tok = tokens[j];
            auto it = nodes[cur].children.find(tok);
            if (it == nodes[cur].children.end()) {
                if ((int) nodes.size() >= max_nodes) {
                    evict_lru((int)(max_nodes * 0.9));
                    // cur index may be invalid after compaction; skip suffix
                    goto next_suffix;
                }
                const int new_idx          = (int) nodes.size();
                nodes[cur].children[tok]   = new_idx;
                nodes.emplace_back();
                nodes.back().count = 0;
                cur = new_idx;
            } else {
                cur = it->second;
            }
            nodes[cur].count++;
        }
        next_suffix:;
    }
}

int suffix_trie::match_depth(const std::vector<llama_token> & context) const {
    if (nodes.empty() || context.empty()) return 0;

    int best_depth = 0;
    const int ctx_size = (int) context.size();

    for (int start = std::max(0, ctx_size - n); start < ctx_size; ++start) {
        int cur   = 0;
        int depth = 0;
        for (int j = start; j < ctx_size; ++j) {
            auto it = nodes[cur].children.find(context[j]);
            if (it == nodes[cur].children.end()) break;
            cur = it->second;
            depth++;
        }
        // Only count as a full match if the entire suffix from `start` matched
        if (depth == ctx_size - start && depth > best_depth) {
            best_depth = depth;
        }
    }
    return best_depth;
}

std::vector<llama_token> suffix_trie::draft_linear(
        const std::vector<llama_token> & context,
        int n_draft) const {

    if (nodes.empty() || context.empty() || n_draft <= 0) return {};

    const int ctx_size = (int) context.size();
    int best_node  = 0;
    int best_depth = 0;

    for (int start = std::max(0, ctx_size - n); start < ctx_size; ++start) {
        int cur   = 0;
        int depth = 0;
        for (int j = start; j < ctx_size; ++j) {
            auto it = nodes[cur].children.find(context[j]);
            if (it == nodes[cur].children.end()) break;
            cur = it->second;
            depth++;
        }
        if (depth == ctx_size - start && depth > best_depth) {
            best_depth = depth;
            best_node  = cur;
        }
    }

    if (best_depth == 0) return {};

    std::vector<llama_token> result;
    result.reserve(n_draft);

    int cur = best_node;
    for (int i = 0; i < n_draft; ++i) {
        if (nodes[cur].children.empty()) break;
        llama_token best_tok  = LLAMA_TOKEN_NULL;
        int32_t     best_count = -1;
        for (const auto & kv : nodes[cur].children) {
            if (nodes[kv.second].count > best_count) {
                best_count = nodes[kv.second].count;
                best_tok   = kv.first;
            }
        }
        if (best_tok == LLAMA_TOKEN_NULL) break;
        result.push_back(best_tok);
        cur = nodes[cur].children.at(best_tok);
    }

    return result;
}

void suffix_trie::evict_lru(int target_nodes) {
    // Collect (parent, edge_token, child_index) for all leaf nodes
    struct leaf_ref { int parent; llama_token tok; int child; };
    std::vector<leaf_ref> leaves;

    for (int i = 0; i < (int) nodes.size(); ++i) {
        for (const auto & kv : nodes[i].children) {
            if (nodes[kv.second].children.empty()) {
                leaves.push_back({i, kv.first, kv.second});
            }
        }
    }

    std::sort(leaves.begin(), leaves.end(), [&](const leaf_ref & a, const leaf_ref & b) {
        return nodes[a.child].count < nodes[b.child].count;
    });

    int need = (int) nodes.size() - target_nodes;
    for (const auto & lf : leaves) {
        if (need <= 0) break;
        auto it = nodes[lf.parent].children.find(lf.tok);
        if (it != nodes[lf.parent].children.end() && it->second == lf.child) {
            nodes[lf.parent].children.erase(it);
            --need;
        }
    }

    // Compact: BFS from root, rebuild with remapped child indices
    std::vector<int> remap(nodes.size(), -1);
    std::vector<st_node> compact;
    compact.reserve(nodes.size());

    std::vector<int> bfs;
    bfs.reserve(nodes.size());
    bfs.push_back(0);
    remap[0] = 0;
    compact.push_back({});
    compact[0].count = nodes[0].count;

    for (int qi = 0; qi < (int) bfs.size(); ++qi) {
        const int old = bfs[qi];
        const int nw  = remap[old];
        for (const auto & kv : nodes[old].children) {
            const int oc = kv.second;
            if (remap[oc] < 0) {
                remap[oc] = (int) compact.size();
                compact.push_back({});
                compact.back().count = nodes[oc].count;
                bfs.push_back(oc);
            }
            compact[nw].children[kv.first] = remap[oc];
        }
    }

    nodes = std::move(compact);
}

bool suffix_trie::save(const char * path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        LOG_ERR("%s: failed to open '%s' for writing\n", __func__, path);
        return false;
    }

    const uint32_t magic      = ST_MAGIC;
    const uint16_t version    = ST_VERSION;
    const int32_t  n_val      = n;
    const int32_t  node_count = (int32_t) nodes.size();

    f.write((const char *) &magic,      sizeof(magic));
    f.write((const char *) &version,    sizeof(version));
    f.write((const char *) &n_val,      sizeof(n_val));
    f.write((const char *) &node_count, sizeof(node_count));

    for (const auto & node : nodes) {
        f.write((const char *) &node.count, sizeof(node.count));
        const uint16_t child_count = (uint16_t) node.children.size();
        f.write((const char *) &child_count, sizeof(child_count));
        for (const auto & kv : node.children) {
            f.write((const char *) &kv.first,  sizeof(kv.first));
            f.write((const char *) &kv.second, sizeof(kv.second));
        }
    }

    return f.good();
}

bool suffix_trie::load(const char * path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    uint32_t magic      = 0;
    uint16_t version    = 0;
    int32_t  n_val      = 0;
    int32_t  node_count = 0;

    f.read((char *) &magic,      sizeof(magic));
    f.read((char *) &version,    sizeof(version));
    f.read((char *) &n_val,      sizeof(n_val));
    f.read((char *) &node_count, sizeof(node_count));

    if (!f || magic != ST_MAGIC || version != ST_VERSION) {
        LOG_ERR("%s: invalid suffix-trie file '%s'\n", __func__, path);
        return false;
    }

    nodes.clear();
    nodes.reserve(node_count);
    n = n_val;

    for (int32_t i = 0; i < node_count; ++i) {
        st_node node;
        f.read((char *) &node.count, sizeof(node.count));
        uint16_t child_count = 0;
        f.read((char *) &child_count, sizeof(child_count));
        for (uint16_t j = 0; j < child_count; ++j) {
            llama_token tok       = 0;
            int32_t     child_idx = 0;
            f.read((char *) &tok,       sizeof(tok));
            f.read((char *) &child_idx, sizeof(child_idx));
            node.children[tok] = child_idx;
        }
        nodes.push_back(std::move(node));
    }

    if (!f && !f.eof()) {
        LOG_ERR("%s: read error in '%s'\n", __func__, path);
        nodes.clear();
        nodes.push_back({});
        return false;
    }

    LOG_INF("%s: loaded suffix-trie from '%s' (n=%d, nodes=%d)\n",
            __func__, path, n, (int) nodes.size());
    return true;
}
