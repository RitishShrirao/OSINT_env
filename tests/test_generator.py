from osint_env.data.generator import DatasetGenerator
from osint_env.domain.models import EnvironmentConfig


def test_generator_outputs():
    gen = DatasetGenerator(EnvironmentConfig(n_users=20, seed=11))
    graph = gen.build_canonical_graph()
    views = gen.build_platform_views(graph)
    tasks = gen.generate_tasks(graph, views, count=5)
    assert len(graph.nodes) >= 20
    assert len(views.microblog_posts) == 20
    assert len(tasks) == 5
