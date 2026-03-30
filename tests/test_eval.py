from osint_env.domain.models import EnvironmentConfig
from osint_env.env.environment import OSINTEnvironment
from osint_env.eval.runner import run_evaluation


def test_eval_runner():
    env = OSINTEnvironment(EnvironmentConfig(seed=17))
    result = run_evaluation(env, episodes=3)
    assert "task_success_rate" in result
    assert "deanonymization_accuracy" in result
