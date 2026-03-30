from osint_env.data.generator import DatasetGenerator
from osint_env.domain.models import EnvironmentConfig
from osint_env.platforms.tools import ToolRegistry


def test_tools_basics():
    gen = DatasetGenerator(EnvironmentConfig(n_users=12, seed=3))
    g = gen.build_canonical_graph()
    views = gen.build_platform_views(g)
    tools = ToolRegistry(views)
    out = tools.search_posts(query="Update")
    assert out["count"] > 0
    profile_any = next(iter([p["user_id"] for p in views.profiles if p["user_id"].startswith("user_")]))
    profile = tools.get_profile(profile_any)
    assert profile["found"] is True
