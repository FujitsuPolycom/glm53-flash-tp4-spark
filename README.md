# GLM-5.3-Flash NVFP4 on four DGX Sparks

This repository serves
[`local-inference-lab/GLM-5.3-Flash-NVFP4`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4)
with vLLM across four NVIDIA DGX Sparks connected as a 200 Gb/s RoCE cycle.

## Status

| Component | Status |
|---|---|
| Four-rank TP4 serving | Implemented and live-smoke verified |
| Native RDMA transport | Implemented; 24 RTS vLLM worker QPs/rank verified |
| 524,288-token context | Implemented |
| Probabilistic MTP4 | Implemented and counter-verified |
| MTP4 performance matrix | Not yet qualified |
| Native B12x end-to-end backend | Separate research canary; not this profile |

The published service contract is deliberately specific. It is not a generic
vLLM recipe for arbitrary hosts or network topologies.

## Serving contract

- Model revision: `8627752b10b78c2b0f2fc69790a94ec9f1ddaa26`
- Four hosts, tensor parallel size 4
- 524,288 configured maximum context
  - The model architecture supports up to 1M tokens, but this profile has not
    qualified that boundary.
- 8,192 batched tokens and 32 sequences
- FP8 KV cache with an explicit 8 GiB slab per rank
  - There may be room to tune this after controlled memory and concurrency
    testing.
- InstantTensor rank-local loading
- FlashKDA prefill
- FlashInfer CUTLASS NVFP4 target experts
- Marlin MXFP8 MTP experts
- Probabilistic MTP4 with standard rejection sampling
- `FULL_AND_PIECEWISE` CUDA graphs
- Capture rows `5,10,20,40,80,160`
- Async scheduling and prefix caching
- SparkRing-patched NCCL 2.30.7 over native RoCE

MTP4 produces five target-query rows per live sequence: one target token plus
four draft tokens. The capture ladder therefore scales through `32 × 5 = 160`.

## Image

The private ARM64 image is published at:

```text
ghcr.io/fujitsupolycom/glm53-flash-sparkring:v14-arm64
```

Deployments pin its immutable digest:

```text
ghcr.io/fujitsupolycom/glm53-flash-sparkring@sha256:8ca89ea984ac8d1bcaed2a0d60141cd0d85abd4d9ad40d98de1c8458d215d524
```

Authenticate before pulling:

```bash
gh auth token | docker login ghcr.io -u FujitsuPolycom --password-stdin
```

The image is ARM64/SM121-specific. The repository does not redistribute model
weights.

## Prerequisites

Each Spark needs:

- NVIDIA DGX Spark software with a driver compatible with CUDA 13
- Docker with NVIDIA Container Toolkit
- two cycle-facing ConnectX/RoCE interfaces
- one 200 Gb/s DAC per ring edge, four DACs total for a four-Spark cycle
- the same RoCEv2 GID index on both cycle-facing interfaces
- enough local storage for the approximately 184 GiB NVFP4 checkpoint
- passwordless SSH from the controller for deployment orchestration

Clone this repository to the controller and copy the model to the same absolute
path on every rank.

## Download the model

Install the Hugging Face CLI and download the pinned revision on rank 0:

```bash
python3 -m pip install -U huggingface_hub hf_xet
hf download local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --revision 8627752b10b78c2b0f2fc69790a94ec9f1ddaa26 \
  --local-dir /var/tmp/models/GLM-5.3-Flash-NVFP4/8627752b10b78c2b0f2fc69790a94ec9f1ddaa26
```

Copy that directory to the remaining ranks over their high-speed addresses.
Every rank must have a complete `model.safetensors.index.json` and all shards
before launch.

## Configure the site

```bash
cp config/service.env.example config/service.env
cp config/nodes.example.tsv config/nodes.tsv
```

Edit both files. `nodes.tsv` has four tab-separated columns:

```text
rank    ssh_target    host_ip    hcas
```

`host_ip` is the address configured on the rendezvous interface. `hcas` is the
comma-separated pair of cycle-facing RDMA devices in local ring order.

The example files contain placeholders and are safe to commit. Resolved site
files should remain private.

## Deploy

The deployment command copies the rank launcher and site environment to every
host, removes only this profile's exact container names, and starts all four
ranks concurrently:

```bash
CONFIRM_REPLACE_GLM53=1 \
  bash scripts/deploy-cluster.sh config/service.env config/nodes.tsv
```

Follow rank 0:

```bash
ssh RANK0_SSH_TARGET \
  'docker logs -f --tail 100 glm53-nvfp4-tp4-r0'
```

First startup reads the checkpoint twice: once for the target and once for the
MTP runner. Graph capture follows. Do not treat the second 184 GiB progress bar
as a reload loop.

## Verify

```bash
bash scripts/verify-cluster.sh config/service.env config/nodes.tsv
```

The verifier requires:

- HTTP 200 from `/health`
- the expected served model and 524,288 context limit
- all four containers running with no OOM or restart
- exactly 24 RTS `VLLM::Worker` RDMA QPs per rank
- no CUDA, NCCL, or vLLM error lines after startup
- four-token drafting (`draft_tokens / drafts == 4` once requests have run)

Send a real request and rerun the verifier.

## Known limitations

- MTP4 has passed API and counter validation, but its throughput has not yet
  been compared against MTP3 under a matched matrix.
- Eight GiB of hybrid KV does not admit 32 independent 8K requests; C16 is the
  currently demonstrated high-concurrency region.
- This profile uses patched NCCL on a four-node cycle. Switch fabrics and
  two-node direct pairs require different transport settings.
- This setup has only been lightly validated on switchless four-Spark clusters.

