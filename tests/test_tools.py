from osint_env.config import clone_environment_config, load_seeding_config, load_shared_config
from osint_env.data.generator import DatasetGenerator
from osint_env.domain.models import EnvironmentConfig
from osint_env.env.environment import OSINTEnvironment
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


def test_seeded_tools_expose_seed_question_entities():
    shared = load_shared_config("datasets/fixed_levels/shared_config_fixed_levels.json")
    env_cfg = clone_environment_config(shared.environment)
    env_cfg.seeding = load_seeding_config("datasets/fixed_levels/seed_fixed_levels.json")
    env_cfg.llm.provider = "mock"
    env = OSINTEnvironment(env_cfg)
    tools = env.tools

    post = tools.get_post("post_midnight_manifest")
    assert post["found"] is True
    assert "loc_dockyard17" in post["result"]["references"]

    people = tools.search_people(org="org_northbridge_logistics")
    user_ids = {row["user_id"] for row in people["results"]}
    assert "user_bharat" in user_ids
    assert "user_hiro" in user_ids

    alias_profile = tools.get_profile("alias_docksparrow")
    assert alias_profile["found"] is True
    assert alias_profile["result"]["user_id"] == "user_hiro"
