from __future__ import annotations

from osint_env.domain.models import Action, ActionType
from osint_env.env.environment import OSINTEnvironment
from osint_env.llm.interface import LLMClient, RuleBasedMockLLM


class SingleAgentRunner:
    def __init__(self, env: OSINTEnvironment, llm: LLMClient | None = None):
        self.env = env
        self.llm = llm or RuleBasedMockLLM()

    def run_episode(self) -> dict:
        obs = self.env.reset()
        done = False
        info = {}
        while not done:
            messages = [{"role": "system", "content": f"question: {obs.task['question']}"}]
            tools = []
            try:
                llm_resp = self.llm.generate(messages, tools)
                planned_calls = llm_resp.tool_calls[:2]
            except Exception:
                planned_calls = []

            for call in planned_calls:
                obs, _, done, info = self.env.step(Action(ActionType.CALL_TOOL, call))
                if done:
                    break
            if done:
                break
            answer_guess = self._heuristic_answer(obs.task["question"])
            obs, _, done, info = self.env.step(Action(ActionType.ANSWER, {"answer": answer_guess}))
        return info

    @staticmethod
    def _heuristic_answer(question: str) -> str:
        for token in question.replace("?", "").split():
            if token.startswith("alias_") or token.startswith("user_"):
                return token
        return "unknown"
