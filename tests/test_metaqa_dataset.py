from pathlib import Path

from osint_env.data.generator import DatasetGenerator
from osint_env.domain.models import EnvironmentConfig


def _write_metaqa_fixture(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "kb.txt").write_text(
        "\n".join(
            [
                "Movie A|starred_actors|Actor X",
                "Movie B|starred_actors|Actor X",
                "Movie A|directed_by|Director D",
                "Movie C|directed_by|Director D",
                "Movie C|release_year|2002",
            ]
        ),
        encoding="utf-8",
    )

    rows = {
        "1-hop": ("what movies did [Actor X] act in\tMovie A|Movie B\n", "actor_to_movie\n"),
        "2-hop": ("which films share the director of [Movie A]\tMovie C\n", "movie_to_director_to_movie\n"),
        "3-hop": (
            "which release year corresponds to films with same director as [Movie A]\t2002\n",
            "movie_to_director_to_movie_to_year\n",
        ),
    }

    for hop, (qa_line, qtype_line) in rows.items():
        qa_dir = root / hop / "vanilla"
        qa_dir.mkdir(parents=True, exist_ok=True)
        (qa_dir / "qa_train.txt").write_text(qa_line, encoding="utf-8")
        (root / hop / "qa_train_qtype.txt").write_text(qtype_line, encoding="utf-8")


def test_metaqa_mode_builds_graph_and_hop_tasks(tmp_path: Path):
    metaqa_root = tmp_path / "metaQA"
    _write_metaqa_fixture(metaqa_root)

    cfg = EnvironmentConfig(
        seed=5,
        dataset_mode="metaqa",
        metaqa_root=str(metaqa_root),
        metaqa_variant="vanilla",
        metaqa_hops=["1-hop", "2-hop", "3-hop"],
        metaqa_splits=["train"],
    )

    gen = DatasetGenerator(cfg)
    graph = gen.build_canonical_graph()
    views = gen.build_platform_views(graph)
    tasks = gen.generate_tasks(graph, views, count=24)

    assert len(graph.nodes) >= 5
    assert any(edge.rel == "directed_by" for edge in graph.edges)
    assert any(post["post_id"].startswith("post_metaqa_") for post in views.microblog_posts)
    assert any(profile["user_id"] == "Actor X" for profile in views.profiles)

    hop_labels = {str(task.metadata.get("hop", "")) for task in tasks}
    difficulties = {str(task.metadata.get("difficulty", "")) for task in tasks}

    assert hop_labels == {"1-hop", "2-hop", "3-hop"}
    assert difficulties == {"easy", "medium", "hard"}
    assert all(task.supporting_edges for task in tasks)
