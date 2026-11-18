#!/usr/bin/env bash
# Run one Isaac gate and check its artifact.
set -o pipefail

STATE_DIR=${ISAAC_GATE_STATE_DIR:-/tmp/isaac_gate_pids}
mkdir -p "${STATE_DIR}" || exit 4
exec 9>"${STATE_DIR}/launcher.lock" || exit 4
if ! flock -n 9; then
    echo "REFUSING TO START: another gate launcher is active"
    exit 4
fi

result_file=$1
log_file=$2
shift 2

# Check recorded process identities.
for pidfile in "${STATE_DIR}"/*.pid; do
    [ -e "${pidfile}" ] || continue
    read -r pid recorded_start < "${pidfile}" 2>/dev/null || true
    if [ -n "${pid:-}" ] && [ -r "/proc/${pid}/stat" ]; then
        live_start=$(awk '{print $22}' "/proc/${pid}/stat" 2>/dev/null || echo "")
        if [ -n "${live_start}" ] && [ "${live_start}" = "${recorded_start}" ]; then
            echo "REFUSING TO START: recorded gate ${pid} is still running"
            echo "stop it deliberately, then run this again"
            exit 4
        else
            echo "ignoring stale pid file for ${pid}: start time does not match, not ours"
        fi
    fi
    rm -f "${pidfile}"
done

# Reject active external gates.
for pid in $(pgrep -f "scripts/check_|scripts/diagnostics/|rsl_rl/train" 2>/dev/null || true); do
    [ "${pid}" = "$$" ] && continue
    [ -r "/proc/${pid}/cmdline" ] || continue
    argv0=$(tr '\0' '\n' < "/proc/${pid}/cmdline" | head -1)
    case "${argv0}" in
        *python*) ;;
        *) continue ;;
    esac
    if [ ! -e "${STATE_DIR}/${pid}.pid" ]; then
        echo "REFUSING TO START: an Isaac gate we did not start is running as ${pid}:"
        ps -o cmd= -p "${pid}" 2>/dev/null | cut -c1-100
        echo "stop it deliberately, then run this again"
        exit 4
    fi
done

# Check available GPU memory.
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "${free_mib}" -lt 12000 ]; then
    echo "REFUSING TO START: only ${free_mib} MiB of GPU memory free"
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
    exit 2
fi
rm -f /tmp/hub-ubuntu-*.config.json 2>/dev/null || true
# A result path of "-" disables artifact checks.
if [ "${result_file}" != "-" ]; then
    rm -f "${result_file}"
fi

# shellcheck disable=SC1091
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/ubuntu/uav_indoor
# Unbuffered Python output.
export PYTHONUNBUFFERED=1

if [ "${result_file}" = "-" ]; then
    echo "starting (no artifact expected): $*"
    "$@" > "${log_file}" 2>&1 &
else
    echo "starting: $* --result ${result_file}"
    "$@" --result "${result_file}" > "${log_file}" 2>&1 &
fi
child=$!
echo "${child} $(awk '{print $22}' "/proc/${child}/stat" 2>/dev/null || echo 0)" \
    > "${STATE_DIR}/${child}.pid"
wait "${child}"
raw_status=$?
rm -f "${STATE_DIR}/${child}.pid"
echo "process exit status ${raw_status} (not a verdict)"

if [ "${result_file}" = "-" ]; then
    echo "no artifact expected; process status ${raw_status} is the only signal"
    exit ${raw_status}
fi
if [ ! -f "${result_file}" ]; then
    echo "GATE DID NOT REPORT: no artifact at ${result_file}"
    exit 3
fi
python - "${result_file}" <<'PY'
import json
import sys

record = json.loads(open(sys.argv[1]).read())
print(f"verdict {record['verdict']} from {record['episodes']} of {record['requested_episodes']} episodes")
for failure in record.get("failures", []):
    print(f"  FAIL: {failure}")
if record["episodes"] < record["requested_episodes"]:
    print("  FAIL: fewer episodes completed than requested")
    sys.exit(1)
sys.exit(0 if record["verdict"] == "PASS" else 1)
PY
