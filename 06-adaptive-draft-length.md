# Adaptive Draft Length Controller — Implementation Spec for llama.cpp

## Source papers / prior art
- "Parallel Speculative Decoding with Adaptive Draft Length" (PEARL)
  Liu et al., arXiv 2408.11850, 2024
- DEL: "Context-Aware Dynamic Exit Layer for Efficient Self-Speculative Decoding"
  arXiv 2504.05598, 2025
- General principle: Heming Xia et al. (2024 survey), Section 4.3 on
  adaptive draft lengths.

## What it does (plain English)
Every llama.cpp speculative decoding variant uses a **fixed** draft length:
`--draft-max N` tokens proposed every single step regardless of whether those
proposals are likely to be accepted. This is wasteful in two ways:

1. When the model is generating something highly predictable (e.g., the middle
   of a code block it has written before), drafts of length 4 are almost
   entirely accepted — you should draft 16-64 tokens instead.
2. When the model is generating something unpredictable (e.g., the first token
   after a user turn), even a draft of length 1 is often rejected — you waste
   compute on a pass that accepts nothing.

An **adaptive draft length controller** tracks the recent acceptance rate and
adjusts `n_draft` dynamically each step:
- High acceptance rate → increase draft length toward `--draft-max`.
- Low acceptance rate → decrease draft length toward `--draft-min`.

This is model-agnostic, spec-type-agnostic, and requires only ~20 lines in the
speculative decoding loop. No new data structures, no new files. Reported
speedup varies widely but PEARL reports **12-37% improvement over fixed-length**
speculative decoding on top of their base acceptance rate — the adaptive
controller is essentially free.

---

## Why this is the smallest change on the list
- Zero new files.
- ~30 lines of code added to `common/speculative.cpp`.
- Works on top of ANY existing `--spec-type` (ngram, SAM, trie, phrase bank,
  draft model, everything).
- The existing `--draft-max` and `--draft-min` flags become the ceiling and
  floor of the dynamic range; existing behavior is preserved when
  `--draft-min == --draft-max`.

---

## The algorithm

Use an **exponential moving average (EMA)** of the per-step acceptance fraction:

```
acceptance_rate_ema = decay * acceptance_rate_ema + (1 - decay) * step_acceptance
```

Where `step_acceptance = n_accepted / n_drafted` for the current step.

Then compute `n_draft` for the next step as a linear interpolation between
`draft_min` and `draft_max` based on `acceptance_rate_ema`:

```
n_draft_next = draft_min + round(acceptance_rate_ema * (draft_max - draft_min))
```

Clip to `[draft_min, draft_max]`.

Optional refinement (PEARL's approach): use a **multi-bucket EMA** — maintain
separate EMAs for "content continuation" tokens vs "turn boundary" tokens, and
switch between them based on whether the last token was a special marker (e.g.,
`<|eot_id|>`, `\n\n`, etc.). At a turn boundary, reset to `draft_min`
regardless of EMA. This avoids wasting a long draft on the first token of a new
response.

---

## Changes to existing files only

### `common/speculative.h`

Add fields to `common_speculative_params`:
```cpp
float spec_adaptive_decay;        // EMA decay 0..1 (default: 0.95)
bool  spec_adaptive_reset_at_eot; // reset draft len at end-of-turn tokens
                                   // (default: true)
```

Add fields to `common_speculative_state` (the internal state struct):
```cpp
float   adaptive_ema;       // current EMA of acceptance rate
int     adaptive_n_draft;   // current dynamic draft length
```

### `common/speculative.cpp`

**`common_speculative_init()`** — add initialization:
```cpp
state->adaptive_ema = 0.5f;   // start at 50% assumed acceptance
state->adaptive_n_draft = (params.n_draft_max + params.n_draft_min) / 2;
```

**Modify the main draft-verify loop** — currently in whichever calling example
invokes `common_speculative_draft()`. For `llama-server` this is in
`server.cpp`; for `llama-speculative` it is in `examples/speculative/`.

The cleanest place to put the controller is in a new helper:

```cpp
// Call AFTER common_sampler_sample_and_accept_n returns n_accepted.
// Updates adaptive_ema and adaptive_n_draft for the next draft step.
void common_speculative_update_adaptive(
    common_speculative_state * state,
    const common_speculative_params & params,
    int n_drafted,
    int n_accepted,
    llama_token last_accepted_token,  // for EOT detection
    const llama_vocab * vocab)
{
    if (params.n_draft_max <= params.n_draft_min) return; // fixed-length mode

    float step_acc = (n_drafted > 0) ? (float)n_accepted / n_drafted : 0.5f;
    state->adaptive_ema = params.spec_adaptive_decay * state->adaptive_ema
                         + (1.f - params.spec_adaptive_decay) * step_acc;

    // Check for end-of-turn token
    bool at_eot = false;
    if (params.spec_adaptive_reset_at_eot && vocab != nullptr) {
        // GGML vocab API: check if last_accepted_token is EOT or EOS
        at_eot = (last_accepted_token == llama_vocab_eot(vocab) ||
                  last_accepted_token == llama_vocab_eos(vocab));
    }

    if (at_eot) {
        // Reset to minimum; the next token is unpredictable
        state->adaptive_n_draft = params.n_draft_min;
        state->adaptive_ema = 0.2f;  // reset EMA too
    } else {
        int range = params.n_draft_max - params.n_draft_min;
        state->adaptive_n_draft = params.n_draft_min
            + (int)roundf(state->adaptive_ema * range);
        state->adaptive_n_draft = std::clamp(state->adaptive_n_draft,
                                             params.n_draft_min,
                                             params.n_draft_max);
    }
}
```

**Pass `adaptive_n_draft` to `common_speculative_draft()`** instead of the
fixed `params.n_draft_max`.  Modify the signature of `common_speculative_draft`:
```cpp
// existing:
std::vector<llama_token> common_speculative_draft(
    common_speculative_state * state,
    const common_speculative_params & params,
    const std::vector<llama_token> & prompt);

// add optional override param:
std::vector<llama_token> common_speculative_draft(
    common_speculative_state * state,
    const common_speculative_params & params,
    const std::vector<llama_token> & prompt,
    int n_draft_override = -1);  // -1 = use params.n_draft_max (existing behavior)
```

Inside `common_speculative_draft`, wherever `params.n_draft_max` is used as the
target draft length, replace with:
```cpp
int n_draft = (n_draft_override >= 0) ? n_draft_override : params.n_draft_max;
```

### Calling sites

In `examples/speculative/speculative.cpp` (and `speculative-simple.cpp`),
the loop after `common_sampler_sample_and_accept_n`:
```cpp
// existing code:
// ... use draft tokens, call decode, get n_accepted ...

// ADD after getting n_accepted:
common_speculative_update_adaptive(
    spec_state, spec_params,
    (int)draft.size(),
    n_accepted,
    draft.empty() ? LLAMA_TOKEN_NULL : draft[n_accepted > 0 ? n_accepted-1 : 0],
    llama_get_model(ctx) ? llama_model_get_vocab(llama_get_model(ctx)) : nullptr
);

// MODIFY next draft call:
auto draft = common_speculative_draft(spec_state, spec_params, inp,
    spec_state->adaptive_n_draft);
```

In `llama-server`: same pattern in the speculative generation callback.

### `common/arg.cpp`
```
--spec-adaptive-decay F       EMA decay for adaptive draft length (default: 0.95)
                              0 = instant adaptation, 1 = never adapt
--spec-no-adaptive-reset      don't reset draft length at EOT/EOS tokens
```

Note: `--draft-min` and `--draft-max` already exist and define the floor/ceiling
of the adaptive range. No new flags needed for the range itself.

---

## Integration sequence

1. Add `adaptive_ema` and `adaptive_n_draft` fields to the state struct. Init
   them in `common_speculative_init()`.
2. Write `common_speculative_update_adaptive()` as a new static function in
   `speculative.cpp`.
3. Modify `common_speculative_draft()` signature to accept `n_draft_override`.
4. Update `examples/speculative/speculative.cpp` calling site.
5. Update `examples/speculative-simple/speculative-simple.cpp` calling site.
6. Update `llama-server` if it calls `common_speculative_draft` directly.
7. Add CLI flags in `arg.cpp`.
8. Test: run with `--draft-min 1 --draft-max 64` and log `adaptive_n_draft`
   over time on a code editing task. Should see it climb to ~50 during
   repetitive output and drop to ~2 at turn boundaries.

---

## Statistics output
Add to the end-of-session stats dump:
```
adaptive draft length: min=1 max=64 final_ema=0.742 (n_draft ≈ 48)
```

---

## Edge cases

- **Fixed-length mode**: if `n_draft_min == n_draft_max`, skip all EMA updates
  and always return `n_draft_max`. Preserves existing behavior exactly.
- **Draft returns fewer tokens than requested**: if the drafting method (ngram,
  SAM, etc.) returns fewer tokens than `adaptive_n_draft`, `n_drafted` for the
  EMA update should be the number actually proposed, not the requested number.
  Treating a short draft as "low acceptance" would unfairly penalize sparse
  retrieval methods.
- **First step cold start**: EMA initialized to 0.5 → initial `n_draft` is the
  midpoint of [min, max]. This is fine.
- **Very low `decay`** (e.g., 0.1): the controller reacts aggressively to each
  step. Useful for models with high variance (reasoning/MoE). For MoE models
  where `ngram-mod` with `--draft-min 48` is currently recommended, set decay
  ~0.5 so it adapts fast to the long reasoning streaks.

---

## Tuning recommendations for specific use cases

**Roleplay (SillyTavern / Alastor character, 780M iGPU)**:
```
--draft-min 4
--draft-max 64
--spec-adaptive-decay 0.9
```
Expected: climbs to 40-64 during long character monologues, drops to 4-8 at
user-turn transitions.

**MoE models (Gemma 4 26B-A4B with ngram-mod)**:
```
--draft-min 24
--draft-max 96
--spec-adaptive-decay 0.95
```
The existing recommendation of `--draft-min 48` is a conservative fixed point;
the adaptive controller will find the same value dynamically and go higher when
conditions allow.

**Code editing (source file refactoring)**:
```
--draft-min 2
--draft-max 64
--spec-adaptive-decay 0.7
```
Fast adaptation: ramps up quickly on repetitive code blocks, resets fast at
function boundaries.
