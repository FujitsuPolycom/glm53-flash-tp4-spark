#!/usr/bin/env bash
set -euo pipefail

service_env="${1:-config/service.env}"
nodes_file="${2:-config/nodes.tsv}"
action="${ACTION:-start}"
remote_root="${REMOTE_ROOT:-/var/tmp/glm53-flash-sparkring}"

if [[ "${CONFIRM_REPLACE_GLM53:-0}" != 1 ]]; then
  echo "set CONFIRM_REPLACE_GLM53=1 to mutate the four serving hosts" >&2
  exit 2
fi
if [[ ! -f "${service_env}" || ! -f "${nodes_file}" ]]; then
  echo "missing service or topology file" >&2
  exit 2
fi
if [[ "${action}" != start && "${action}" != stop ]]; then
  echo "ACTION must be start or stop" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${service_env}"
set +a
: "${CONTAINER_PREFIX:?CONTAINER_PREFIX is required}"

declare -a ranks targets host_ips hcas
while IFS=$'\t' read -r rank target host_ip rank_hcas; do
  [[ -z "${rank}" || "${rank}" == \#* ]] && continue
  ranks+=("${rank}")
  targets+=("${target}")
  host_ips+=("${host_ip}")
  hcas+=("${rank_hcas}")
done < "${nodes_file}"

if [[ "${#ranks[@]}" != 4 ]]; then
  echo "topology must contain exactly four ranks" >&2
  exit 2
fi

for index in "${!ranks[@]}"; do
  target="${targets[index]}"
  ssh "${target}" "mkdir -p '${remote_root}'"
  scp -q "${service_env}" "${target}:${remote_root}/service.env"
  scp -q scripts/launch-rank.sh "${target}:${remote_root}/launch-rank.sh"
done

declare -a pids
for index in "${!ranks[@]}"; do
  rank="${ranks[index]}"
  target="${targets[index]}"
  ssh "${target}" \
    "docker stop -t 15 '${CONTAINER_PREFIX}-r${rank}' >/dev/null 2>&1 || true; docker rm '${CONTAINER_PREFIX}-r${rank}' >/dev/null 2>&1 || true" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

if [[ "${action}" == stop ]]; then
  echo "stopped ${CONTAINER_PREFIX} on all four ranks"
  exit 0
fi

pids=()
for index in "${!ranks[@]}"; do
  rank="${ranks[index]}"
  target="${targets[index]}"
  ssh "${target}" \
    "bash '${remote_root}/launch-rank.sh' '${rank}' '${host_ips[index]}' '${hcas[index]}' '${remote_root}/service.env'" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

echo "started ${CONTAINER_PREFIX}; follow rank 0 until API readiness"

