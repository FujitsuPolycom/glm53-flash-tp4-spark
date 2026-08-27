#!/usr/bin/env bash
set -euo pipefail

service_env="${1:-config/service.env}"
nodes_file="${2:-config/nodes.tsv}"

set -a
# shellcheck disable=SC1090
source "${service_env}"
set +a

failures=0
rank0_target=""
while IFS=$'\t' read -r rank target host_ip hcas; do
  [[ -z "${rank}" || "${rank}" == \#* ]] && continue
  [[ "${rank}" == 0 ]] && rank0_target="${target}"
  echo "rank ${rank} ${target}"
  if ! ssh "${target}" \
    "docker inspect -f 'state={{.State.Status}} oom={{.State.OOMKilled}} restart={{.RestartCount}} image={{.Config.Image}}' '${CONTAINER_PREFIX}-r${rank}'"; then
    failures=$((failures + 1))
    continue
  fi
  worker_rts="$(ssh "${target}" "rdma res show qp 2>/dev/null | grep 'state RTS' | grep -c 'comm VLLM::Worker' || true")"
  echo "worker_rts=${worker_rts}"
  [[ "${worker_rts}" == 24 ]] || failures=$((failures + 1))
  errors="$(ssh "${target}" "docker logs --since 30m '${CONTAINER_PREFIX}-r${rank}' 2>&1 | grep -Ec 'ERROR|CUDA error|NCCL WARN' || true")"
  echo "errors=${errors}"
  [[ "${errors}" == 0 ]] || failures=$((failures + 1))
done < "${nodes_file}"

if [[ -z "${rank0_target}" ]]; then
  echo "rank 0 is missing from topology" >&2
  exit 2
fi

health="$(ssh "${rank0_target}" "curl -sS -o /dev/null -w '%{http_code}' 'http://127.0.0.1:${PORT}/health'" || true)"
echo "health=${health}"
[[ "${health}" == 200 ]] || failures=$((failures + 1))

ssh "${rank0_target}" "curl -fsS 'http://127.0.0.1:${PORT}/v1/models'; echo"

metrics="$(ssh "${rank0_target}" "curl -fsS 'http://127.0.0.1:${PORT}/metrics'" || true)"
printf '%s\n' "${metrics}" | grep -E \
  'spec_decode_num_(drafts|draft_tokens|accepted_tokens)_total|num_requests_(running|waiting)|num_preemptions_total' || true

drafts="$(printf '%s\n' "${metrics}" | awk '/^vllm:spec_decode_num_drafts_total/{print $NF; exit}')"
draft_tokens="$(printf '%s\n' "${metrics}" | awk '/^vllm:spec_decode_num_draft_tokens_total/{print $NF; exit}')"
if [[ -n "${drafts}" && -n "${draft_tokens}" ]] && awk "BEGIN {exit !(${drafts} > 0)}"; then
  if ! awk "BEGIN {exit !(${draft_tokens} == 4 * ${drafts})}"; then
    echo "MTP4 invariant failed: drafts=${drafts} draft_tokens=${draft_tokens}" >&2
    failures=$((failures + 1))
  fi
fi

if [[ "${failures}" != 0 ]]; then
  echo "verification failed: ${failures} check(s)" >&2
  exit 1
fi
echo "verification passed"

