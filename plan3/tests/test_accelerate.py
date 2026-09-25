from pathlib import Path

from plan3.accelerate import stage_commands


def test_acceleration_preserves_search_contract_but_expands_model_cpu():
    retrieval = {"budgets": {"word": 200, "missing": 100}, "threads": 16,
                 "query_batch": 1000, "shard_size": 50000}
    commands = dict(stage_commands(Path("run"), 64, 64, retrieval))
    search = commands["retrieve"]
    assert search[search.index("--threads") + 1] == "16"
    assert search[search.index("--query-batch") + 1] == "1000"
    assert search[search.index("--channels") + 1] == "word,missing"
    for stage in ("train", "support", "control"):
        command = commands[stage]
        assert command[command.index("--threads") + 1] == "64"
    features = commands["features"]
    assert features[features.index("--workers") + 1] == "64"
    assert features[features.index("--worker-threads") + 1] == "1"
    assert commands["support"][commands["support"].index("--workers") + 1] == "64"
    assert list(commands).index("control") < list(commands).index("compare_control")
