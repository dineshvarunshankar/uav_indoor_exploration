#!/usr/bin/env bash
# Print active Isaac gate PIDs and elapsed times.
for pid in $(pgrep -f "scripts/check_|rsl_rl/train" 2>/dev/null || true); do
    [ "${pid}" = "$$" ] && continue
    [ -r "/proc/${pid}/cmdline" ] || continue
    case "$(tr '\0' '\n' < "/proc/${pid}/cmdline" | head -1)" in
        *python*) ;;
        *) continue ;;
    esac
    echo "${pid} $(ps -o etime= -p "${pid}" | tr -d ' ') $(ps -o cmd= -p "${pid}" | cut -c1-90)"
done
