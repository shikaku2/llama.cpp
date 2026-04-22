# CREST-Style Cache Compaction — Implementation Spec for llama.cpp

## Source paper
"CREST: Effectively Compacting a Datastore For Retrieval-Based Speculative
Decoding"
Ho, Park, Wang (CMU), arXiv 2408.04678, August 2024

## What it does (plain English)
The existing `--lookup-cache-dynamic` file grows unboundedly: every n-gram the
model has ever seen gets recorded with a count. Over long multi-session use, the
file becomes large, load time slows down, and many entries are rare noise that
reduce lookup precision rather than helping.

CREST's core insight: **keeping only the smallest (shortest) and most common
n-grams out-performs keeping everything**, both in acceptance length and in
storage. Specifically it "decouples" n-grams from each other so that individual
entries can be pruned independently, then filters to retain only high-frequency
short n-grams.

Results: CREST matches REST's full-datastore acceptance length with **10.6–13.5×
less storage** and achieves **16.5–17.1% higher acceptance length** than the
full datastore at equal storage budget.

## Why this is the easiest item on the list
This is purely a modification to the existing `common_ngram_cache_save()`
function — no new data structures, no new spec types, no new verify paths. It
is a compression/eviction policy applied at save time (and optionally at load
time). The change is ~100 lines total.

---

## Changes required

### New function in `common/ngram-cache.h`
```cpp
struct common_ngram_cache_compaction_params {
    int   min_hits;          // drop entries with total count < this (default: 2)
    int   prefer_short;      // prefer n-grams of size <= this (default: 2)
    int   max_entries;       // hard cap on total n-gram entries (default: 0=no limit)
    float top_freq_fraction; // if max_entries set, keep top this fraction by count
                             // (default: 1.0 = no fraction filter)
};

// Save the ngram cache to `path` after applying compaction.
// Drops rare and long n-grams according to params.
// Returns false on write error.
bool common_ngram_cache_save_compact(
    const common_ngram_cache & nc,
    const char * path,
    common_ngram_cache_compaction_params params = {});

// Load a cache from `path` and apply compaction filter on load.
// Useful when loading a pre-existing uncompacted static cache.
common_ngram_cache common_ngram_cache_load_compact(
    const char * path,
    common_ngram_cache_compaction_params params = {});
```

### Implementation of `common_ngram_cache_save_compact()`

The existing `common_ngram_cache_save()` iterates over the outer map
(n-gram → continuation map) and inner map (token → count) and writes everything.
The compact variant adds two filter passes:

**Pass 1: per-continuation filtering**
For each n-gram key, in its continuation map, drop any (token, count) pair where
`count < params.min_hits`. If the continuation map becomes empty after pruning,
drop the entire n-gram entry.

**Pass 2: per-n-gram length filtering**
For each n-gram key, determine its effective size (number of non-zero tokens in
the `common_ngram.tokens[]` array). If `prefer_short > 0` and the n-gram size
> prefer_short, apply a stricter threshold: only keep if the maximum continuation
count for that n-gram >= `min_hits * 3` (i.e., long n-grams need to be 3x
more popular to survive).

**Pass 3: hard cap**
If `max_entries > 0`, collect all (n-gram, max_continuation_count) pairs sorted
descending by count, and keep only the top `max_entries`.

```cpp
bool common_ngram_cache_save_compact(
    const common_ngram_cache & nc,
    const char * path,
    common_ngram_cache_compaction_params params)
{
    // Build filtered list
    struct entry {
        common_ngram          ngram;
        common_ngram_cache_part continuations;  // filtered
        int32_t               max_count;
        int                   ngram_size;
    };
    std::vector<entry> kept;

    for (auto & [ngram, cont_map] : nc) {
        // Determine n-gram size
        int ngram_size = 0;
        for (int i = 0; i < LLAMA_NGRAM_MAX; i++) {
            if (ngram.tokens[i] != 0) ngram_size = i+1;
        }

        // Per-continuation filter
        common_ngram_cache_part filtered_cont;
        int32_t max_count = 0;
        for (auto & [tok, count] : cont_map) {
            int threshold = params.min_hits;
            if (params.prefer_short > 0 && ngram_size > params.prefer_short)
                threshold = params.min_hits * 3;
            if (count >= threshold) {
                filtered_cont[tok] = count;
                max_count = std::max(max_count, count);
            }
        }
        if (filtered_cont.empty()) continue;

        kept.push_back({ngram, filtered_cont, max_count, ngram_size});
    }

    // Hard cap
    if (params.max_entries > 0 && (int)kept.size() > params.max_entries) {
        std::sort(kept.begin(), kept.end(),
            [](const entry & a, const entry & b) {
                return a.max_count > b.max_count;  // descending
            });
        kept.resize(params.max_entries);
    }

    // Write to file using existing binary format
    // (same format as common_ngram_cache_save, just with filtered entries)
    FILE * f = fopen(path, "wb");
    if (!f) return false;
    // ... write header, iterate kept, write each entry ...
    fclose(f);
    return true;
}
```

### Modified `common/ngram-cache.cpp`

The existing `common_ngram_cache_save()` can remain unchanged for compatibility.
`common_ngram_cache_save_compact()` is additive.

Alternatively, add a `compact` flag to `common_ngram_cache_save()` if
maintaining one code path is preferred:
```cpp
void common_ngram_cache_save(
    common_ngram_cache & ngram_cache,
    const char * path,
    bool compact = false,
    int min_hits = 2,
    int prefer_short = 2);
```

### `common/arg.cpp` — new CLI flags

For the lookup example and llama-server:
```
--lookup-cache-compact-on-save     apply compaction when writing dynamic cache
--lookup-cache-min-hits N          drop ngrams with total count < N (default: 2)
--lookup-cache-prefer-short N      apply stricter threshold to ngrams longer than N
                                   (default: 2, meaning bigrams+ are preferred)
--lookup-cache-max-entries N       hard cap on number of ngram entries in cache
                                   (default: 0 = no cap)
```

Wire these into `common_params` (add fields) and pass through to a
`common_ngram_cache_compaction_params` when the dynamic cache is saved on exit.

### `examples/lookup/lookup.cpp`

The existing save-on-exit code:
```cpp
if (!params.lookup_cache_dynamic.empty()) {
    common_ngram_cache_save(nc_dynamic, params.lookup_cache_dynamic.c_str());
}
```
Replace with:
```cpp
if (!params.lookup_cache_dynamic.empty()) {
    if (params.lookup_cache_compact_on_save) {
        common_ngram_cache_compaction_params cp;
        cp.min_hits    = params.lookup_cache_min_hits;
        cp.prefer_short = params.lookup_cache_prefer_short;
        cp.max_entries = params.lookup_cache_max_entries;
        common_ngram_cache_save_compact(nc_dynamic,
            params.lookup_cache_dynamic.c_str(), cp);
    } else {
        common_ngram_cache_save(nc_dynamic, params.lookup_cache_dynamic.c_str());
    }
}
```

Same change in `llama-server`'s shutdown path.

---

## Standalone offline compaction tool

Add `examples/ngram-compact/ngram-compact.cpp` — a small standalone tool that
reads an existing `.bin` cache, applies compaction, and writes a new file:

```
Usage: llama-ngram-compact \
    -i input-cache.bin \
    -o output-cache.bin \
    --min-hits 3 \
    --prefer-short 2 \
    --max-entries 500000
```

This is ~50 lines wrapping `common_ngram_cache_load()` +
`common_ngram_cache_save_compact()`. Very useful for compacting a large static
corpus cache built with `llama-lookup-create`.

---

## Integration sequence

1. Implement `common_ngram_cache_save_compact()` in `ngram-cache.cpp`.
   Unit test: build a synthetic cache with many low-count and high-count
   entries, call save_compact with min_hits=2, verify the output file is
   smaller and that high-count entries are preserved.
2. Add `common_ngram_cache_load_compact()` similarly.
3. Add params and CLI flags in `arg.cpp`.
4. Wire into `examples/lookup/lookup.cpp` save path.
5. Wire into `llama-server` shutdown path (wherever the dynamic cache is saved).
6. Build `llama-ngram-compact` standalone tool.
7. Measure: run a long session with `--lookup-cache-dynamic session.bin`, compare
   file size before and after, verify acceptance rate is equal or better.

---

## Expected impact for the user's setup

The 780M iGPU setup uses `ngram-mod` (no persistent cache) and `ngram-map-k`
(in-memory only). This change matters when using `--lookup-cache-dynamic` across
long SillyTavern sessions. After a multi-hour session the dynamic cache can grow
to tens of MB of mostly-singleton n-grams. Applying compaction at save:
- Reduces load time on next session startup (smaller file).
- Improves acceptance rate (noise removed, true patterns dominate).
- Allows using `--lookup-cache-static` with a large pre-built corpus
  (e.g., tokenized Hazbin Hotel scripts) that has been compacted to <10 MB
  of high-frequency n-grams from the full raw file.

Recommended initial settings for roleplay cache compaction:
```
--lookup-cache-compact-on-save
--lookup-cache-min-hits 3
--lookup-cache-prefer-short 2
--lookup-cache-max-entries 200000
```

---

## File format note

The existing `common_ngram_cache_save()` uses a simple sequential binary dump
with no header (just raw map contents). This is fragile — no version, no
vocab hash, no entry count. As part of this change, consider adding a minimal
header to the compact output format:

```
[magic u32: 0x4E47434D]   // "NGCM"
[version u16: 2]
[vocab_hash u64]           // first-2048-tokens hash for model compatibility check
[ngram_min u8]
[ngram_max u8]
[entry_count i32]
... existing entry format unchanged ...
```

This header is backward-incompatible with the old format; detect by checking
whether the first 4 bytes == magic. If not, fall back to the old load path.
New saves always use the new format.

---

## Relation to Suffix Trie spec (02-suffix-trie.md)

If the suffix trie is implemented, the compaction policy for its serialization
format should mirror this spec: evict leaf nodes with count < min_hits, prefer
shorter paths. The `evict_lru()` method in `suffix_trie` is essentially the
trie equivalent of Pass 1 + Pass 2 here.
