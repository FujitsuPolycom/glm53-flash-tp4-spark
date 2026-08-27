from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mtp4_and_graph_geometry_are_bound() -> None:
    launcher = (ROOT / "scripts" / "launch-rank.sh").read_text()
    assert '"num_speculative_tokens":4' in launcher
    assert "[5,10,20,40,80,160]" in launcher
    assert "--max-cudagraph-capture-size 160" in launcher


def test_serving_contract_is_complete() -> None:
    launcher = (ROOT / "scripts" / "launch-rank.sh").read_text()
    for value in (
        "--tensor-parallel-size 4",
        "--load-format instanttensor",
        "--moe-backend flashinfer_cutlass",
        "--linear-backend flashinfer_b12x",
        "--kv-cache-dtype fp8",
        "--kda-prefill-backend flashkda",
        "--async-scheduling",
        "--enable-prefix-caching",
        "NCCL_NET=IB",
        "NCCL_IB_DISABLE=0",
        "NCCL_ALGO=Ring",
    ):
        assert value in launcher


def test_unified_memory_profile_uses_direct_aio_and_ten_gib_kv() -> None:
    launcher = (ROOT / "scripts" / "launch-rank.sh").read_text()
    service = (ROOT / "config" / "service.env.example").read_text()
    assert "INSTANTTENSOR_BACKEND=AIO" in launcher
    assert "AIO_BUFFERED" not in launcher
    assert "KV_CACHE_MEMORY_BYTES=10737418240" in service


def test_site_specific_values_are_placeholders() -> None:
    examples = "\n".join(
        (ROOT / "config" / name).read_text()
        for name in ("service.env.example", "nodes.example.tsv")
    )
    assert "192.168." not in examples
    node_rows = [
        line
        for line in (ROOT / "config" / "nodes.example.tsv").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert len(node_rows) == 4
    assert all("user@RANK" in row and "_IP" in row for row in node_rows)


def test_mutating_orchestrator_requires_explicit_confirmation() -> None:
    deploy = (ROOT / "scripts" / "deploy-cluster.sh").read_text()
    assert 'CONFIRM_REPLACE_GLM53:-0' in deploy
    assert "docker stop -t 15" in deploy
    assert "docker rm" in deploy
