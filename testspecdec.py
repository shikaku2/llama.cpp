#!/usr/bin/env python3
"""
testspecdec.py — benchmark every power-set combination of ngram speculative decoding types

Usage:
  ./testspecdec.py [llama-server args...] [--test-ngram all|none|type1,type2,...] [--repeats N] [--log FILE] [--lcd-wipe]

Custom flags (consumed before passthrough to llama-server):
  --test-ngram  comma-separated subset of ngram types, "all", "all-once", or "none" (baseline only)
                all-once: baseline + all types together (2 runs); all: full power-set (default: all)
  --repeats N   times to cycle through the question list per combo  (default: 50)
  --log FILE    append per-combo result lines here  (default: specdec_results.log)
  --lcd-wipe    delete the -lcd cache file before each combo so every run starts cold

All other args are forwarded verbatim to llama-server.

Type names: ngram-simple  ngram-map-k  ngram-map-k4v  ngram-mod  ngram-cache
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from itertools import combinations

# ---------------------------------------------------------------------------
# Questions — edit freely. Short deterministic answers maximise spec-dec benefit.
# ---------------------------------------------------------------------------
QUESTIONS = [
    "A train travels 60 km/h for 2.5 hours, then 90 km/h for 1.5 hours. What is the total distance traveled? Show your reasoning.",
    "If I have 3 boxes each containing 4 bags each containing 5 marbles, how many marbles do I have total? Explain step by step.",
    "A rectangle has a perimeter of 36 cm and its length is twice its width. What are the dimensions? Show your work.",
    "Alice is twice as old as Bob. In 5 years Alice will be 1.5 times as old as Bob. How old are they now? Show your reasoning.",
    "A store sells apples for $1.20 each and oranges for $0.80 each. I spend exactly $10.00 buying 10 pieces of fruit. How many of each? Show your reasoning.",
    "If a ball is dropped from 100 meters and bounces back to half its height each time, how high does it reach after the 4th bounce? Show each step.",
    "Three friends split a restaurant bill. The total is $87. One friend pays double what the second pays, and the third pays $3 more than the second. How much does each pay?",
    "A tank is filled by pipe A in 3 hours and drained by pipe B in 5 hours. If both are open, how long to fill an empty tank? Show your reasoning.",
    "Convert 0.142857142857... to a fraction. Explain why this repeating decimal is equal to that fraction.",
    "A farmer has chickens and cows. He counts 20 heads and 56 legs. How many chickens and how many cows? Show your reasoning.",
    "If you invest $1000 at 5% annual compound interest, how much do you have after 3 years? Show each year's calculation.",
    "A right triangle has legs of length 5 and 12. What is the hypotenuse? What are the angles? Show your reasoning.",
    "In a class of 30 students, 18 play soccer, 15 play basketball, and 8 play both. How many play neither? Use a Venn diagram approach.",
    "A car and a truck start 300 km apart driving toward each other. The car goes 80 km/h and the truck 70 km/h. When and where do they meet?",
    "What is the sum of all integers from 1 to 100? Explain Gauss's method and verify it.",
    "A recipe for 4 people needs 2.5 cups of flour. How much flour for 7 people? Show the proportional reasoning.",
    "If a pizza is cut into 8 equal slices and I eat 3, what percentage did I eat? What fraction remains? Show your working.",
    "A ladder 10 meters long leans against a wall. The base is 6 meters from the wall. How high up the wall does it reach? Show your reasoning.",
    "I think of a number, double it, add 15, divide by 5, and get 9. What was the original number? Work backwards step by step.",
    "A snail climbs 3 meters up a 10-meter pole each day but slides back 2 meters each night. How many days to reach the top? Explain carefully.",
]

ALL_NGRAM_TYPES = ["ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache"]

LLAMA_SERVER = "/home/aaron/llama.cpp-spec-dec/build/bin/llama-server"
SRV_LOG = f"/tmp/specdec_srv_{os.getpid()}.log"


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------
def parse_args(argv):
    test_ngram = "all"
    repeats = 50
    log_file = "specdec_results.log"
    lcd_wipe = False
    n_questions = len(QUESTIONS)
    pass_args = []

    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--test-ngram":
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                test_ngram = argv[i + 1]; i += 2
            else:
                test_ngram = "all"; i += 1
        elif a.startswith("--test-ngram="):
            test_ngram = a.split("=", 1)[1]; i += 1
        elif a == "--repeats":
            repeats = int(argv[i + 1]); i += 2
        elif a.startswith("--repeats="):
            repeats = int(a.split("=", 1)[1]); i += 1
        elif a == "--log":
            log_file = argv[i + 1]; i += 2
        elif a.startswith("--log="):
            log_file = a.split("=", 1)[1]; i += 1
        elif a == "--lcd-wipe":
            lcd_wipe = True; i += 1
        elif a == "--questions":
            n_questions = int(argv[i + 1]); i += 2
        elif a.startswith("--questions="):
            n_questions = int(a.split("=", 1)[1]); i += 1
        else:
            pass_args.append(a); i += 1

    return test_ngram, repeats, log_file, lcd_wipe, n_questions, pass_args


def extract_flag_value(args, *flags):
    """Return the value for a flag from a list of args, or None."""
    for i, a in enumerate(args):
        for flag in flags:
            if a == flag and i + 1 < len(args):
                return args[i + 1]
            if a.startswith(flag + "="):
                return a.split("=", 1)[1]
    return None


def build_types(test_ngram_arg):
    """Returns (types, all_once). all_once=True means only run baseline + all-together."""
    if test_ngram_arg == "all-once":
        return list(ALL_NGRAM_TYPES), True
    if test_ngram_arg == "all":
        return list(ALL_NGRAM_TYPES), False
    if test_ngram_arg == "none":
        return [], False
    types = [t.strip() for t in test_ngram_arg.split(",") if t.strip()]
    unknown = [t for t in types if t not in ALL_NGRAM_TYPES]
    if unknown:
        print(f"ERROR: unknown ngram type(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"Valid: {', '.join(ALL_NGRAM_TYPES)}", file=sys.stderr)
        sys.exit(1)
    return types, False


def power_set(types):
    """All non-empty subsets, ordered by size (singles first, then pairs, etc.)."""
    return [
        ",".join(combo)
        for r in range(1, len(types) + 1)
        for combo in combinations(types, r)
    ]


# ---------------------------------------------------------------------------
# Human-readable file size (1024-based)
# ---------------------------------------------------------------------------
def human_size(path):
    if not path or not os.path.isfile(path):
        return "(absent)"
    try:
        b = os.path.getsize(path)
    except OSError:
        return "(unknown)"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if b < 1024 or unit == "TiB":
            return f"{b:.2f} {unit}" if unit != "B" else f"{b} B"
        b /= 1024


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------
def start_server(pass_args, spec_arg):
    cmd = [LLAMA_SERVER] + pass_args + ["--spec-type", spec_arg]
    with open(SRV_LOG, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc


def wait_ready(proc, port, timeout=120):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, "server process exited unexpectedly"
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True, None
        except Exception:
            pass
        time.sleep(1)
    return False, f"server did not become ready within {timeout}s"


def stop_server(proc, grace=5):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def show_server_log(n=30):
    try:
        with open(SRV_LOG) as f:
            lines = f.readlines()
        if not lines:
            print("(no server output captured)", file=sys.stderr)
            return
        print("--- server output (last 30 lines) ---", file=sys.stderr)
        for line in lines[-n:]:
            print(line, end="", file=sys.stderr)
        print("-------------------------------------", file=sys.stderr)
    except OSError:
        print("(server log not found)", file=sys.stderr)


WARMUP_TYPES = {"ngram-cache", "ngram-map-k", "ngram-map-k4v"}

def combo_has_warmup(combo):
    return any(t in WARMUP_TYPES for t in combo.split(","))


def read_accept_vals():
    """Return all acceptance rate floats currently in the server log."""
    try:
        with open(SRV_LOG) as f:
            content = f.read()
    except OSError:
        return []
    return [float(m) for m in re.findall(r'draft acceptance rate\s*=\s*([0-9.]+)', content)]


# ---------------------------------------------------------------------------
# Run questions
# ---------------------------------------------------------------------------
def run_questions(proc, port, repeats, questions, show_warmup=False):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    errors = 0
    first_error = None
    crashed = False
    seen_accept = 0  # how many acceptance-rate log lines we've consumed

    total_qs = repeats * len(questions)
    done = 0
    q_start = time.monotonic()

    for rep in range(1, repeats + 1):
        for qi, q in enumerate(questions, 1):
            if proc.poll() is not None:
                print(f"\n  server crashed mid-run after {errors} error(s)", file=sys.stderr)
                show_server_log()
                crashed = True
                break

            elapsed = time.monotonic() - q_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_qs - done) / rate if rate > 0 else 0
            bar_filled = int(20 * done / total_qs) if total_qs > 0 else 0
            bar = "#" * bar_filled + "." * (20 - bar_filled)
            print(f"\r  [{bar}] rep {rep}/{repeats} q {qi}/{len(questions)} | "
                  f"{done}/{total_qs} | {rate:.2f} q/s | eta {eta:.0f}s | err {errors}  ",
                  end="", flush=True)

            body = json.dumps({
                "model": "test",
                "messages": [{"role": "user", "content": q}],
                "max_tokens": 2048,
            }).encode()
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=120) as r:
                    resp = json.loads(r.read())
                if "error" in resp:
                    errors += 1
                    if first_error is None:
                        first_error = str(resp["error"])
            except urllib.error.HTTPError as e:
                errors += 1
                if first_error is None:
                    try:
                        first_error = e.read().decode(errors="replace")
                    except Exception:
                        first_error = str(e)
            except Exception as e:
                errors += 1
                if first_error is None:
                    first_error = str(e)
            done += 1
        if crashed:
            break

        if show_warmup and repeats > 1:
            all_vals = read_accept_vals()
            new_vals = all_vals[seen_accept:]
            seen_accept = len(all_vals)
            if new_vals:
                avg = sum(new_vals) / len(new_vals)
                print(f"\n    repeat {rep}/{repeats}: accept={avg:.4f} ({len(new_vals)} samples)", flush=True)

    print()  # newline after progress bar

    if errors > 0:
        msg = first_error or "(empty response)"
        print(f"  {errors} request error(s); first: {msg}", file=sys.stderr)
        if not crashed:
            show_server_log()

    return errors


# ---------------------------------------------------------------------------
# Parse server log stats
# ---------------------------------------------------------------------------
def parse_tok_per_sec():
    try:
        with open(SRV_LOG) as f:
            lines = f.readlines()
    except OSError:
        return "N/A"
    vals = []
    for line in lines:
        if "eval time" in line:
            nums = re.findall(r'\d+\.\d+', line)
            if nums:
                vals.append(float(nums[-1]))
    if not vals:
        return "N/A"
    return f"{sum(vals)/len(vals):.1f}"


def parse_acceptance_rate():
    try:
        with open(SRV_LOG) as f:
            lines = f.readlines()
    except OSError:
        return "N/A"
    vals = []
    for line in lines:
        if "draft acceptance rate" in line:
            m = re.search(r'draft acceptance rate\s*=\s*([0-9.]+)', line)
            if m:
                vals.append(float(m.group(1)))
    if not vals:
        return "N/A"
    return f"{sum(vals)/len(vals):.4f}"


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def make_row(label, tok_ps, accept, wall_s, errors, lcd_before=None, lcd_after=None, use_lcd=False):
    if use_lcd:
        return f"{label:<40}  {tok_ps:>9}  {accept:>12}  {wall_s+'s':>10}  {str(errors):>7}  {lcd_before or '':<18}  {lcd_after or '':<18}"
    return f"{label:<40}  {tok_ps:>9}  {accept:>12}  {wall_s+'s':>10}  {str(errors):>7}"


def print_header(use_lcd):
    if use_lcd:
        print(f"\n{'COMBO':<40}  {'TOK/S(avg)':>9}  {'ACCEPT_RATE':>12}  {'WALL(s)':>10}  {'ERRORS':>7}  {'LCD_BEFORE':<18}  {'LCD_AFTER':<18}")
        print("-" * 120)
    else:
        print(f"\n{'COMBO':<40}  {'TOK/S(avg)':>9}  {'ACCEPT_RATE':>12}  {'WALL(s)':>10}  {'ERRORS':>7}")
        print("-" * 85)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    test_ngram_arg, repeats, log_file, lcd_wipe, n_questions, pass_args = parse_args(sys.argv[1:])

    questions = QUESTIONS[:n_questions]

    port = int(extract_flag_value(pass_args, "--port") or 8080)
    lcd_path = extract_flag_value(pass_args, "-lcd", "--lookup-cache-dynamic")
    use_lcd = bool(lcd_path)

    types, all_once = build_types(test_ngram_arg)
    if all_once:
        combos = ["none", ",".join(types)] if types else ["none"]
    else:
        combos = ["none"] + power_set(types)
    total = len(combos)

    print(f"testspecdec: {total} combo(s) × {repeats} repeat(s) × {len(questions)} question(s) = {total * repeats * len(questions)} total requests")
    print(f"testspecdec: port={port}, log={log_file}")
    if use_lcd:
        mode = "(wiped before each combo)" if lcd_wipe else "(accumulates across combos)"
        print(f"testspecdec: lcd={lcd_path} {mode}")
    print(f"testspecdec: server args: {' '.join(pass_args) or '<none>'}")

    # Write log header
    with open(log_file, "a") as f:
        f.write(f"# testspecdec run {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# args: {' '.join(pass_args) or '<none>'}  repeats={repeats}  questions={len(questions)}\n")
        if use_lcd:
            f.write(f"{'COMBO':<40}  {'TOK/S(avg)':>9}  {'ACCEPT_RATE':>12}  {'WALL(s)':>10}  {'ERRORS':>7}  {'LCD_BEFORE':<18}  {'LCD_AFTER':<18}\n")
        else:
            f.write(f"{'COMBO':<40}  {'TOK/S(avg)':>9}  {'ACCEPT_RATE':>12}  {'WALL(s)':>10}  {'ERRORS':>7}\n")

    print_header(use_lcd)

    results = []
    proc = None

    def cleanup():
        nonlocal proc
        if proc and proc.poll() is None:
            stop_server(proc)
        try:
            os.unlink(SRV_LOG)
        except OSError:
            pass

    # Handle Ctrl-C gracefully
    def sigint_handler(sig, frame):
        print("\ntestspecdec: interrupted, cleaning up...", file=sys.stderr)
        cleanup()
        sys.exit(130)

    signal.signal(signal.SIGINT, sigint_handler)

    try:
        for idx, combo in enumerate(combos, 1):
            if lcd_wipe and lcd_path:
                try:
                    os.unlink(lcd_path)
                except OSError:
                    pass

            lcd_before = human_size(lcd_path) if use_lcd else None

            print(f"[{idx}/{total}] Starting server with --spec-type {combo} ...", flush=True)

            with open(SRV_LOG, "w"):
                pass  # truncate

            proc = start_server(pass_args, combo)

            ok, err_msg = wait_ready(proc, port)
            if not ok:
                print(f"  ERROR: {err_msg}", file=sys.stderr)
                show_server_log()
                stop_server(proc)
                proc = None
                print(f"  SKIP: server failed to start", file=sys.stderr)
                continue

            t_start = time.monotonic()
            errors = run_questions(proc, port, repeats, questions, show_warmup=combo_has_warmup(combo))
            wall_s = f"{time.monotonic() - t_start:.1f}"

            stop_server(proc)
            proc = None

            lcd_after = human_size(lcd_path) if use_lcd else None
            tok_ps = parse_tok_per_sec()
            accept = parse_acceptance_rate()

            row = make_row(combo, tok_ps, accept, wall_s, errors, lcd_before, lcd_after, use_lcd)
            print(row)
            with open(log_file, "a") as f:
                f.write(row + "\n")

            results.append((combo, tok_ps, accept, wall_s, errors, lcd_before, lcd_after))

    finally:
        cleanup()

    # Final summary
    print("\n" + "=" * 30 + " SUMMARY " + "=" * 30)
    print_header(use_lcd)
    for combo, tok_ps, accept, wall_s, errors, lcd_before, lcd_after in results:
        print(make_row(combo, tok_ps, accept, wall_s, errors, lcd_before, lcd_after, use_lcd))
    print(f"\nResults appended to: {log_file}")


if __name__ == "__main__":
    main()
