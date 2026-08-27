#!/usr/bin/env bash
set -euo pipefail

rank="${1:?rank is required}"
host_ip="${2:?host IP is required}"
hcas="${3:?comma-separated RDMA devices are required}"
service_env="${4:-/var/tmp/glm53-flash-sparkring/service.env}"

if [[ ! "${rank}" =~ ^[0-3]$ ]]; then
  echo "rank must be 0-3; got ${rank}" >&2
  exit 2
fi
if [[ ! -f "${service_env}" ]]; then
  echo "service environment is missing: ${service_env}" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "${service_env}"
set +a

required=(
  IMAGE MODEL_DIR SERVED_MODEL_NAME CONTAINER_PREFIX CACHE_ROOT
  MASTER_ADDR MASTER_PORT PORT SOCKET_IFNAME NCCL_IB_GID_INDEX
  MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
  KV_CACHE_MEMORY_BYTES GPU_MEMORY_UTILIZATION PATCHED_NCCL_SO
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "required setting is empty: ${name}" >&2
    exit 2
  fi
done

container="${CONTAINER_PREFIX}-r${rank}"
cache_dir="${CACHE_ROOT}/r${rank}"
headless=()
if [[ "${rank}" != 0 ]]; then
  headless=(--headless)
fi

if [[ ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
  echo "model is incomplete: ${MODEL_DIR}" >&2
  exit 3
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "serving image is missing: ${IMAGE}" >&2
  exit 4
fi
if ! docker run --rm --entrypoint /bin/sh "${IMAGE}" -c \
  "test -r '${PATCHED_NCCL_SO}'"; then
  echo "patched NCCL is missing from ${IMAGE}: ${PATCHED_NCCL_SO}" >&2
  exit 4
fi
if docker ps -a --format '{{.Names}}' | grep -Fxq "${container}"; then
  echo "container already exists: ${container}" >&2
  exit 5
fi

mkdir -p "${cache_dir}"

speculative_config='{"method":"mtp","num_speculative_tokens":4,"draft_tensor_parallel_size":4,"use_local_argmax_reduction":false,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}'
compilation_config='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[5,10,20,40,80,160],"custom_ops":["all"]}'

docker run -d \
  --name "${container}" \
  --init \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 16g \
  --cap-add IPC_LOCK \
  --ulimit memlock=-1:-1 \
  --device /dev/infiniband:/dev/infiniband \
  --security-opt label=disable \
  -v "${MODEL_DIR}:/models/glm53:ro" \
  -v "${cache_dir}:/cache/jit" \
  -e TORCH_USE_RTLD_GLOBAL=1 \
  -e GLOO_SOCKET_IFNAME="${SOCKET_IFNAME}" \
  -e NCCL_SOCKET_IFNAME="${SOCKET_IFNAME}" \
  -e NCCL_NET_PLUGIN=none \
  -e NCCL_NET=IB \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="${hcas}" \
  -e NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX}" \
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1 \
  -e NCCL_IB_SUBNET_PREFIX_LEN=24 \
  -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CROSS_NIC=1 \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e NCCL_MIN_NCHANNELS=4 \
  -e NCCL_MAX_NCHANNELS=4 \
  -e NCCL_ALGO=Ring \
  -e NCCL_SKIP_TREE_CONNECT=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e NCCL_DEBUG=WARN \
  -e VLLM_HOST_IP="${host_ip}" \
  -e VLLM_NCCL_SO_PATH="${PATCHED_NCCL_SO}" \
  -e LD_PRELOAD="${PATCHED_NCCL_SO}" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e SPARKRING_SKIP_MM_RENDERER_WARMUP=1 \
  -e VLLM_USE_AOT_COMPILE=1 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e TORCHINDUCTOR_COMPILE_THREADS=1 \
  -e OMP_NUM_THREADS=16 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1f \
  -e CUTE_DSL_ARCH=sm_121a \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e CMAKE_CUDA_ARCHITECTURES=121 \
  -e XDG_CACHE_HOME=/cache/jit \
  -e TRITON_CACHE_DIR=/cache/jit/triton \
  -e TORCH_EXTENSIONS_DIR=/cache/jit/torch_extensions \
  -e VLLM_CACHE_ROOT=/cache/jit/vllm \
  -e FLASHINFER_WORKSPACE_BASE=/cache/jit/flashinfer \
  -e INSTANTTENSOR_BACKEND=AIO_BUFFERED \
  -e INSTANTTENSOR_BUFFER_SIZE=1268776960 \
  -e INSTANTTENSOR_CONCURRENCY=1 \
  -e INSTANTTENSOR_IO_DEPTH=3 \
  -e INSTANTTENSOR_CHUNK_SIZE=2097152 \
  -e INSTANTTENSOR_MAX_FREE_MEM_USAGE=0.05 \
  "${IMAGE}" \
  /models/glm53 \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size 4 \
  --load-format instanttensor \
  --quantization modelopt_mixed \
  --linear-backend flashinfer_b12x \
  --moe-backend flashinfer_cutlass \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --enable-chunked-prefill \
  --kda-prefill-backend flashkda \
  --disable-custom-all-reduce \
  --kernel-config '{"enable_cutedsl_warmup":false,"enable_flashinfer_autotune":false}' \
  --speculative-config "${speculative_config}" \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --port "${PORT}" \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank "${rank}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --compilation-config "${compilation_config}" \
  --max-cudagraph-capture-size 160 \
  --async-scheduling \
  --enable-prefix-caching \
  --cudagraph-metrics \
  "${headless[@]}"

