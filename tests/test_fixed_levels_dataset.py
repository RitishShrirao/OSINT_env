import json
from collections import Counter
from pathlib import Path


def test_fixed_levels_seed_has_30_questions_and_target_node_spans():
    path = Path("datasets/fixed_levels/seed_fixed_levels.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload["seeding"]["seeded_questions"]

    counts = Counter(q["metadata"]["difficulty"] for q in questions)
    assert counts == {"easy": 10, "mid": 10, "high": 10}

    mid_support_nodes = [int(q["metadata"]["support_nodes"]) for q in questions if q["metadata"]["difficulty"] == "mid"]
    high_support_nodes = [int(q["metadata"]["support_nodes"]) for q in questions if q["metadata"]["difficulty"] == "high"]

    assert all(15 <= value <= 20 for value in mid_support_nodes)
    assert all(48 <= value <= 55 for value in high_support_nodes)
