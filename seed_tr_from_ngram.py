#!/usr/bin/env python3
"""
Seed a token-recycling (.cache) file from an ngram cache (.ngram) file.

Only unigram entries (1-token context) are useful for TR since TR's adjacency
matrix is indexed by a single context token.  Counts are normalized to
frequencies and used as initial scores.  Existing TR rows are left alone
unless --overwrite is given.

Usage:
    python seed_tr_from_ngram.py gemma4.ngram gemma4-token.cache
    python seed_tr_from_ngram.py gemma4.ngram gemma4-token.cache --min-count 3
    python seed_tr_from_ngram.py gemma4.ngram gemma4-token.cache --no-merge
    python seed_tr_from_ngram.py gemma4.ngram new.cache --vocab-size 262144
"""

import argparse
import struct
import sys
from pathlib import Path

# ── constants matching C++ sources ──────────────────────────────────────────

NGRAM_MAX        = 4
LLAMA_TOKEN_NULL = -1   # sentinel for unused ngram key slots

TR_MAGIC   = 0x54524341
TR_VERSION = 1

# Header layout (no C struct padding — written field-by-field in C++):
#   uint32  magic
#   uint16  version
#   int32   vocab_size
#   int32   k
#   float   decay
TR_HDR_FMT  = '<IHiif'
TR_HDR_SIZE = struct.calcsize(TR_HDR_FMT)   # 18 bytes

# Per-row: uint8 count, then count × (int32 token, float score)
TR_ENTRY_FMT  = '<if'
TR_ENTRY_SIZE = struct.calcsize(TR_ENTRY_FMT)  # 8 bytes

# Ngram cache record (no header, raw records until EOF):
#   common_ngram  = 4 × int32  (key)
#   int32         ntokens
#   ntokens × (int32 token, int32 count)
NGRAM_KEY_FMT  = '<4i'
NGRAM_KEY_SIZE = struct.calcsize(NGRAM_KEY_FMT)  # 16 bytes


# ── ngram reader ─────────────────────────────────────────────────────────────

def read_ngram_unigrams(path: str, min_count: int) -> dict[int, dict[int, int]]:
    """Return {context_token: {next_token: count}} for 1-gram entries only."""
    data = Path(path).read_bytes()
    total = len(data)
    offset = 0
    unigrams: dict[int, dict[int, int]] = {}
    n_skipped_multi = 0

    while offset < total:
        if offset + NGRAM_KEY_SIZE + 4 > total:
            break

        t0, t1, t2, t3 = struct.unpack_from(NGRAM_KEY_FMT, data, offset)
        offset += NGRAM_KEY_SIZE

        ntokens, = struct.unpack_from('<i', data, offset)
        offset += 4

        pairs_size = ntokens * 8
        if offset + pairs_size > total:
            break

        pairs = struct.unpack_from(f'<{ntokens * 2}i', data, offset)
        offset += pairs_size

        # Skip multi-gram entries (only unigrams map to TR's adj matrix)
        if t1 != LLAMA_TOKEN_NULL or t2 != LLAMA_TOKEN_NULL or t3 != LLAMA_TOKEN_NULL:
            n_skipped_multi += 1
            continue

        row = unigrams.setdefault(t0, {})
        for i in range(ntokens):
            tok, count = pairs[i * 2], pairs[i * 2 + 1]
            if count >= min_count:
                row[tok] = row.get(tok, 0) + count

    print(f"  {len(unigrams)} unigram context tokens read "
          f"({n_skipped_multi} multi-gram entries skipped)", file=sys.stderr)
    return unigrams


# ── TR cache reader / writer ──────────────────────────────────────────────────

def read_tr(path: str):
    """Return (vocab_size, k, decay, adj) where adj[t] = [(token, score), ...]."""
    data = Path(path).read_bytes()
    if len(data) < TR_HDR_SIZE:
        raise ValueError(f"File too short to be a TR cache: {path}")

    magic, version, vocab_size, k, decay = struct.unpack_from(TR_HDR_FMT, data, 0)
    if magic != TR_MAGIC or version != TR_VERSION:
        raise ValueError(f"Not a valid TR cache (magic={magic:#010x}, version={version})")

    adj: list[list[tuple[int, float]]] = []
    offset = TR_HDR_SIZE
    for _ in range(vocab_size):
        cnt, = struct.unpack_from('<B', data, offset)
        offset += 1
        row = []
        for _ in range(cnt):
            tok, score = struct.unpack_from(TR_ENTRY_FMT, data, offset)
            offset += TR_ENTRY_SIZE
            row.append((tok, score))
        adj.append(row)

    return vocab_size, k, decay, adj


def write_tr(path: str, vocab_size: int, k: int, decay: float,
             adj: list[list[tuple[int, float]]]) -> None:
    with open(path, 'wb') as f:
        f.write(struct.pack(TR_HDR_FMT, TR_MAGIC, TR_VERSION, vocab_size, k, decay))
        for row in adj:
            f.write(struct.pack('<B', len(row)))
            for tok, score in row:
                f.write(struct.pack(TR_ENTRY_FMT, tok, score))


# ── main logic ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ngram',  help='Input ngram cache file (.ngram)')
    ap.add_argument('tr',     help='TR cache file to update (or create if absent)')
    ap.add_argument('--out',  metavar='FILE',
                    help='Write result here instead of updating TR in-place')
    ap.add_argument('--vocab-size', type=int, metavar='N',
                    help='Required when creating a new TR file from scratch')
    ap.add_argument('-k', type=int, default=None,
                    help='Top-k entries per row (default: from TR file, or 5 for new files)')
    ap.add_argument('--decay', type=float, default=None,
                    help='EMA decay (default: from TR file, or 0.9 for new files)')
    ap.add_argument('--min-count', type=int, default=2, metavar='N',
                    help='Ignore ngram entries with count < N (default: 2)')
    ap.add_argument('--no-merge', action='store_true',
                    help='Skip rows that already have TR data instead of merging')
    args = ap.parse_args()

    output_path = args.out or args.tr

    # ── load ngram unigrams ──
    print(f"Reading ngram cache: {args.ngram}", file=sys.stderr)
    unigrams = read_ngram_unigrams(args.ngram, args.min_count)

    # ── load or create TR cache ──
    tr_path = Path(args.tr)
    if tr_path.exists():
        print(f"Reading TR cache: {args.tr}", file=sys.stderr)
        vocab_size, k, decay, adj = read_tr(args.tr)
        filled_before = sum(1 for r in adj if r)
        print(f"  vocab={vocab_size}, k={k}, decay={decay}, "
              f"filled={filled_before}/{vocab_size} "
              f"({100*filled_before/vocab_size:.2f}%)", file=sys.stderr)
        # Allow overriding k/decay from CLI
        if args.k     is not None: k     = args.k
        if args.decay is not None: decay = args.decay
        if args.vocab_size is not None and args.vocab_size != vocab_size:
            print(f"Warning: ignoring --vocab-size {args.vocab_size}, "
                  f"using {vocab_size} from TR file", file=sys.stderr)
    else:
        if args.vocab_size is None:
            ap.error("TR file does not exist; --vocab-size is required to create it")
        vocab_size = args.vocab_size
        k          = args.k     if args.k     is not None else 5
        decay      = args.decay if args.decay is not None else 0.9
        adj        = [[] for _ in range(vocab_size)]
        print(f"Creating new TR cache: vocab={vocab_size}, k={k}, decay={decay}",
              file=sys.stderr)

    # ── merge ngram frequencies into TR rows ──
    # For each token in a row: take max(tr_score, ngram_freq).
    # Tokens only in ngram are added to the pool; tokens only in TR are kept.
    # Pool is then sorted descending and capped at k.  Running the script
    # multiple times is idempotent since max(x, x) = x.
    n_merged  = 0
    n_seeded  = 0
    n_skipped = 0
    n_oob     = 0

    for ctx_tok, counts in unigrams.items():
        if ctx_tok < 0 or ctx_tok >= vocab_size:
            n_oob += 1
            continue

        total = sum(counts.values())
        if total == 0:
            continue

        existing = adj[ctx_tok]

        if not existing:
            # Row is empty — just fill it directly
            top = sorted(counts.items(), key=lambda x: -x[1])[:k]
            adj[ctx_tok] = [(tok, count / total) for tok, count in top]
            n_seeded += 1
        elif args.no_merge:
            n_skipped += 1
        else:
            # Merge: build combined pool keyed by token, taking max score
            pool: dict[int, float] = {tok: score for tok, score in existing}
            for tok, count in counts.items():
                freq = count / total
                if tok in pool:
                    pool[tok] = max(pool[tok], freq)
                else:
                    pool[tok] = freq
            top = sorted(pool.items(), key=lambda x: -x[1])[:k]
            adj[ctx_tok] = [(tok, score) for tok, score in top]
            n_merged += 1

    filled_after = sum(1 for r in adj if r)
    print(f"\nMerged {n_merged} rows  |  "
          f"seeded {n_seeded} new rows  |  "
          f"skipped {n_skipped} (--no-merge)  |  "
          f"{n_oob} out-of-range tokens ignored", file=sys.stderr)
    print(f"Filled: {filled_after}/{vocab_size} "
          f"({100*filled_after/vocab_size:.2f}%)", file=sys.stderr)

    # ── write output ──
    print(f"Writing: {output_path}", file=sys.stderr)
    write_tr(output_path, vocab_size, k, decay, adj)
    size_kb = Path(output_path).stat().st_size / 1024
    print(f"Done ({size_kb:.1f} KiB)", file=sys.stderr)


if __name__ == '__main__':
    main()
