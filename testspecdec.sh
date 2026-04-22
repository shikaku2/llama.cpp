#!/usr/bin/env bash
# testspecdec.sh — benchmark every power-set combination of ngram speculative decoding types
#
# Usage:
#   ./testspecdec.sh [llama-server args...] [--test-ngram all|none|type1,type2,...] [--repeats N] [--log FILE] [--lcd-wipe]
#
# Custom flags (consumed before passthrough to llama-server):
#   --test-ngram  comma-separated subset of ngram types, "all", or "none" (baseline only)
#                 default: all
#   --repeats N   times to cycle through the question list per combo  (default: 50)
#   --log FILE    append per-combo result lines here  (default: specdec_results.log)
#   --lcd-wipe    delete the -lcd cache file before each combo so every run starts cold
#
# All other args are forwarded verbatim to llama-server.
#
# Type names: ngram_simple  ngram_map_k  ngram_map_k4v  ngram_mod  ngram_cache

set -euo pipefail

# ---------------------------------------------------------------------------
# Questions — edit freely. Processed in order, then the list repeats $REPEATS times.
# Short, deterministic answers maximise spec-dec benefit and keep runs fast.
# ---------------------------------------------------------------------------
QUESTIONS=(
    "What is 25 times 25?"
    "What is 13 times 13?"
    "What is 7 times 8?"
    "What is 144 divided by 12?"
    "What is 2 to the power of 8?"
    "How many seconds are in a minute?"
    "How many inches are in a foot?"
    "How many days are in a week?"
    "What is the capital of France?"
    "What is the capital of Germany?"
    "What day comes after Friday?"
    "What month comes after March?"
    "What is the chemical symbol for gold?"
    "What is the square root of 64?"
    "How many sides does a hexagon have?"
)

# ---------------------------------------------------------------------------
# All known ngram spec types (no draft-model types — those need -md)
# ---------------------------------------------------------------------------
ALL_NGRAM_TYPES=(ngram_simple ngram_map_k ngram_map_k4v ngram_mod ngram_cache)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
REPEATS=50
LOG_FILE="specdec_results.log"
TEST_NGRAM_ARG="all"
LCD_WIPE=0
LLAMA_SERVER="/home/aaron/llama.cpp-spec-dec/build/bin/llama-server"
SRV_LOG="/tmp/specdec_srv_$$.log"

# ---------------------------------------------------------------------------
# Arg parsing — pull out our flags, forward the rest
# ---------------------------------------------------------------------------
PASS_ARGS=()
i=0
args=("$@")
while [[ $i -lt ${#args[@]} ]]; do
    arg="${args[$i]}"
    case "$arg" in
        --test-ngram)
            # optional value: if next token exists and doesn't start with - treat as value
            next_i=$(( i + 1 ))
            if [[ $next_i -lt ${#args[@]} && "${args[$next_i]}" != -* ]]; then
                TEST_NGRAM_ARG="${args[$next_i]}"
                i=$(( next_i + 1 ))
            else
                TEST_NGRAM_ARG="all"
                i=$(( i + 1 ))
            fi
            ;;
        --test-ngram=*)
            TEST_NGRAM_ARG="${arg#--test-ngram=}"
            i=$(( i + 1 ))
            ;;
        --repeats)
            REPEATS="${args[$(( i + 1 ))]}"
            i=$(( i + 2 ))
            ;;
        --repeats=*)
            REPEATS="${arg#--repeats=}"
            i=$(( i + 1 ))
            ;;
        --log)
            LOG_FILE="${args[$(( i + 1 ))]}"
            i=$(( i + 2 ))
            ;;
        --log=*)
            LOG_FILE="${arg#--log=}"
            i=$(( i + 1 ))
            ;;
        --lcd-wipe)
            LCD_WIPE=1
            i=$(( i + 1 ))
            ;;
        *)
            PASS_ARGS+=("$arg")
            i=$(( i + 1 ))
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Extract port from passthrough args (for curl)
# ---------------------------------------------------------------------------
PORT=8080
for (( j=0; j<${#PASS_ARGS[@]}; j++ )); do
    if [[ "${PASS_ARGS[$j]}" == "--port" && $(( j + 1 )) -lt ${#PASS_ARGS[@]} ]]; then
        PORT="${PASS_ARGS[$(( j + 1 ))]}"
    elif [[ "${PASS_ARGS[$j]}" =~ ^--port=(.+)$ ]]; then
        PORT="${BASH_REMATCH[1]}"
    fi
done

# ---------------------------------------------------------------------------
# Extract -lcd / --lookup-cache-dynamic path from passthrough args
# ---------------------------------------------------------------------------
LCD_PATH=""
for (( j=0; j<${#PASS_ARGS[@]}; j++ )); do
    p="${PASS_ARGS[$j]}"
    if [[ ( "$p" == "-lcd" || "$p" == "--lookup-cache-dynamic" ) && $(( j + 1 )) -lt ${#PASS_ARGS[@]} ]]; then
        LCD_PATH="${PASS_ARGS[$(( j + 1 ))]}"
    elif [[ "$p" =~ ^(-lcd|--lookup-cache-dynamic)=(.+)$ ]]; then
        LCD_PATH="${BASH_REMATCH[2]}"
    fi
done

# ---------------------------------------------------------------------------
# Human-readable SI file size (1024-based, like ls -lh)
# ---------------------------------------------------------------------------
human_size() {
    local path="$1"
    if [[ -z "$path" || ! -f "$path" ]]; then
        echo "(absent)"
        return
    fi
    local bytes
    bytes=$(stat -c %s "$path" 2>/dev/null) || { echo "(unknown)"; return; }
    awk -v b="$bytes" 'BEGIN {
        split("B KiB MiB GiB TiB", u)
        v = b; s = 1
        while (v >= 1024 && s < 5) { v /= 1024; s++ }
        if (s == 1) printf "%d %s", v, u[s]
        else        printf "%.2f %s", v, u[s]
    }'
}

# ---------------------------------------------------------------------------
# Build the set of types to include in the power set
# ---------------------------------------------------------------------------
if [[ "$TEST_NGRAM_ARG" == "all" ]]; then
    TYPES=("${ALL_NGRAM_TYPES[@]}")
elif [[ "$TEST_NGRAM_ARG" == "none" ]]; then
    TYPES=()
else
    IFS=',' read -r -a TYPES <<< "$TEST_NGRAM_ARG"
fi

# ---------------------------------------------------------------------------
# Validate requested types
# ---------------------------------------------------------------------------
for t in "${TYPES[@]}"; do
    valid=0
    for known in "${ALL_NGRAM_TYPES[@]}"; do
        [[ "$t" == "$known" ]] && valid=1 && break
    done
    if [[ $valid -eq 0 ]]; then
        echo "ERROR: unknown ngram type '$t'. Valid: ${ALL_NGRAM_TYPES[*]}" >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Power-set generator — emits every non-empty subset as a comma-separated line
# ---------------------------------------------------------------------------
generate_subsets() {
    local arr=("$@")
    local n=${#arr[@]}
    local total=$(( 1 << n ))
    for (( i=1; i<total; i++ )); do
        local combo=()
        for (( j=0; j<n; j++ )); do
            (( (i >> j) & 1 )) && combo+=("${arr[$j]}")
        done
        ( IFS=,; echo "${combo[*]}" )
    done
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    rm -f "$SRV_LOG"
}
trap cleanup EXIT

show_server_log() {
    if [[ -s "$SRV_LOG" ]]; then
        echo "--- server output (last 30 lines) ---" >&2
        tail -30 "$SRV_LOG" >&2
        echo "-------------------------------------" >&2
    else
        echo "(no server output captured)" >&2
    fi
}

wait_ready() {
    local tries=120
    for (( t=1; t<=tries; t++ )); do
        # bail early if the server process already died
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server process exited unexpectedly" >&2
            show_server_log
            return 1
        fi
        if curl -sf "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: server did not become ready within ${tries}s" >&2
    show_server_log
    return 1
}

# Escape a string for JSON (handles quotes and backslashes)
json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    printf '%s' "$s"
}

run_questions() {
    local errors=0
    local first_error=""
    for (( rep=1; rep<=REPEATS; rep++ )); do
        for Q in "${QUESTIONS[@]}"; do
            # abort remaining requests if the server died
            if ! kill -0 "$SERVER_PID" 2>/dev/null; then
                echo "  server crashed mid-run after $errors error(s)" >&2
                show_server_log
                echo "$errors"
                return
            fi
            local escaped
            escaped="$(json_escape "$Q")"
            local resp
            resp=$(curl -s --max-time 60 -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":\"${escaped}\"}],\"max_tokens\":64}" \
                2>/dev/null) || true
            if [[ -z "$resp" ]] || echo "$resp" | grep -q '"error"'; then
                (( errors++ )) || true
                # capture first error body for display
                if [[ -z "$first_error" && -n "$resp" ]]; then
                    first_error="$resp"
                fi
            fi
        done
    done
    if [[ $errors -gt 0 ]]; then
        echo "  $errors request error(s); first error: ${first_error:-(empty response)}" >&2
        show_server_log
    fi
    echo "$errors"
}

parse_acceptance_rate() {
    grep "draft acceptance rate" "$SRV_LOG" 2>/dev/null \
        | awk '{sum+=$5; n++} END {if(n>0) printf "%.4f", sum/n; else print "N/A"}'
}

parse_tok_per_sec() {
    # "eval time = ... tokens per second" — last numeric field on matching lines
    grep "eval time" "$SRV_LOG" 2>/dev/null \
        | awk '{for(i=1;i<=NF;i++) if($i~/^[0-9]+\.[0-9]+$/) last=$i; sum+=last; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}'
}

print_header() {
    if [[ -n "$LCD_PATH" ]]; then
        printf "\n%-40s  %9s  %12s  %10s  %7s  %-18s  %-18s\n" \
            "COMBO" "TOK/S(avg)" "ACCEPT_RATE" "WALL(s)" "ERRORS" "LCD_BEFORE" "LCD_AFTER"
        printf '%0.s-' {1..120}; echo
    else
        printf "\n%-40s  %9s  %12s  %10s  %7s\n" \
            "COMBO" "TOK/S(avg)" "ACCEPT_RATE" "WALL(s)" "ERRORS"
        printf '%0.s-' {1..85}; echo
    fi
}

# ---------------------------------------------------------------------------
# Build the list of combos to run: baseline first, then all ngram subsets
# ---------------------------------------------------------------------------
COMBOS=("none")
if [[ ${#TYPES[@]} -gt 0 ]]; then
    while IFS= read -r line; do
        COMBOS+=("$line")
    done < <(generate_subsets "${TYPES[@]}")
fi

total_combos=${#COMBOS[@]}
echo "testspecdec: ${total_combos} combo(s) × ${REPEATS} repeat(s) × ${#QUESTIONS[@]} question(s) = $(( total_combos * REPEATS * ${#QUESTIONS[@]} )) total requests"
echo "testspecdec: port=${PORT}, log=${LOG_FILE}"
[[ -n "$LCD_PATH" ]] && echo "testspecdec: lcd=${LCD_PATH}$([ $LCD_WIPE -eq 1 ] && echo ' (wiped before each combo)' || echo ' (accumulates across combos)')"
echo "testspecdec: server args: ${PASS_ARGS[*]:-<none>}"
echo ""

# Write log header
{
    echo "# testspecdec run $(date '+%Y-%m-%d %H:%M:%S')"
    echo "# args: ${PASS_ARGS[*]:-<none>}  repeats=${REPEATS}  questions=${#QUESTIONS[@]}"
    if [[ -n "$LCD_PATH" ]]; then
        printf "%-40s  %9s  %12s  %10s  %7s  %-18s  %-18s\n" \
            "COMBO" "TOK/S(avg)" "ACCEPT_RATE" "WALL(s)" "ERRORS" "LCD_BEFORE" "LCD_AFTER"
    else
        printf "%-40s  %9s  %12s  %10s  %7s\n" "COMBO" "TOK/S(avg)" "ACCEPT_RATE" "WALL(s)" "ERRORS"
    fi
} >> "$LOG_FILE"

# ---------------------------------------------------------------------------
# Summary accumulation
# ---------------------------------------------------------------------------
declare -a RES_COMBO=()
declare -a RES_TOKPS=()
declare -a RES_ACCEPT=()
declare -a RES_WALL=()
declare -a RES_ERRORS=()
declare -a RES_LCD_BEFORE=()
declare -a RES_LCD_AFTER=()

print_header

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
combo_num=0
for combo in "${COMBOS[@]}"; do
    (( combo_num++ )) || true
    label="$combo"
    spec_arg="$combo"

    # Wipe lcd cache if requested
    if [[ $LCD_WIPE -eq 1 && -n "$LCD_PATH" ]]; then
        rm -f "$LCD_PATH"
    fi

    # Record lcd size before
    lcd_before=""
    if [[ -n "$LCD_PATH" ]]; then
        lcd_before="$(human_size "$LCD_PATH")"
    fi

    printf "[%d/%d] Starting server with --spec-type %s ...\n" "$combo_num" "$total_combos" "$spec_arg"

    # Start server
    rm -f "$SRV_LOG"
    "${LLAMA_SERVER}" "${PASS_ARGS[@]}" --spec-type "$spec_arg" > "$SRV_LOG" 2>&1 &
    SERVER_PID=$!

    if ! wait_ready; then
        echo "  SKIP: server failed to start (check ${SRV_LOG})" >&2
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        continue
    fi

    # Time the question run
    t_start=$(date +%s%N)
    errors=$(run_questions)
    t_end=$(date +%s%N)
    wall_s=$(awk "BEGIN {printf \"%.1f\", ($t_end - $t_start)/1000000000}")

    # Stop server gracefully (SIGTERM → give it 5s → SIGKILL)
    kill "$SERVER_PID" 2>/dev/null || true
    for (( w=0; w<5; w++ )); do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    kill -0 "$SERVER_PID" 2>/dev/null && kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""

    # Record lcd size after (server destructor flushes on shutdown)
    lcd_after=""
    if [[ -n "$LCD_PATH" ]]; then
        lcd_after="$(human_size "$LCD_PATH")"
    fi

    # Parse stats from server log
    tok_ps=$(parse_tok_per_sec)
    accept=$(parse_acceptance_rate)

    # Store results
    RES_COMBO+=("$label")
    RES_TOKPS+=("$tok_ps")
    RES_ACCEPT+=("$accept")
    RES_WALL+=("$wall_s")
    RES_ERRORS+=("$errors")
    RES_LCD_BEFORE+=("$lcd_before")
    RES_LCD_AFTER+=("$lcd_after")

    # Print row
    if [[ -n "$LCD_PATH" ]]; then
        printf "%-40s  %9s  %12s  %10s  %7s  %-18s  %-18s\n" \
            "$label" "$tok_ps" "$accept" "${wall_s}s" "$errors" "$lcd_before" "$lcd_after"
    else
        printf "%-40s  %9s  %12s  %10s  %7s\n" "$label" "$tok_ps" "$accept" "${wall_s}s" "$errors"
    fi

    # Append to log file
    if [[ -n "$LCD_PATH" ]]; then
        printf "%-40s  %9s  %12s  %10s  %7s  %-18s  %-18s\n" \
            "$label" "$tok_ps" "$accept" "${wall_s}s" "$errors" "$lcd_before" "$lcd_after" >> "$LOG_FILE"
    else
        printf "%-40s  %9s  %12s  %10s  %7s\n" \
            "$label" "$tok_ps" "$accept" "${wall_s}s" "$errors" >> "$LOG_FILE"
    fi
done

# ---------------------------------------------------------------------------
# Final summary table
# ---------------------------------------------------------------------------
echo ""
echo "============================== SUMMARY =============================="
print_header
for (( i=0; i<${#RES_COMBO[@]}; i++ )); do
    if [[ -n "$LCD_PATH" ]]; then
        printf "%-40s  %9s  %12s  %10s  %7s  %-18s  %-18s\n" \
            "${RES_COMBO[$i]}" "${RES_TOKPS[$i]}" "${RES_ACCEPT[$i]}" "${RES_WALL[$i]}s" \
            "${RES_ERRORS[$i]}" "${RES_LCD_BEFORE[$i]}" "${RES_LCD_AFTER[$i]}"
    else
        printf "%-40s  %9s  %12s  %10s  %7s\n" \
            "${RES_COMBO[$i]}" "${RES_TOKPS[$i]}" "${RES_ACCEPT[$i]}" "${RES_WALL[$i]}s" "${RES_ERRORS[$i]}"
    fi
done
echo ""
echo "Results appended to: ${LOG_FILE}"
