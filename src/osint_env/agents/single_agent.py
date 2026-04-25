from __future__ import annotations

import re

from osint_env.domain.models import Action, ActionType
from osint_env.env.environment import OSINTEnvironment
from osint_env.llm.interface import LLMClient, RuleBasedMockLLM
from osint_env.platforms.tool_schemas import build_lookup_tools


class SingleAgentRunner:
    def __init__(self, env: OSINTEnvironment, llm: LLMClient | None = None):
        self.env = env
        self.llm = llm or RuleBasedMockLLM()

    def run_episode(self) -> dict:
        obs = self.env.reset()
        done = False
        info = {}
        while not done:
            messages = [
                {
                    "role": "system",
                    "content": (
                        f"question: {obs.task['question']}\n"
                        f"shared_context_available: {bool(obs.task.get('shared_context_available', False))}\n"
                        "Use lookup tools to gather evidence before answering."
                    ),
                }
            ]
            tools = build_lookup_tools()
            try:
                llm_resp = self.llm.generate(messages, tools)
                planned_calls = llm_resp.tool_calls[:2]
            except Exception:
                planned_calls = []

            if not planned_calls and bool(obs.task.get("shared_context_available", False)):
                planned_calls = [
                    {
                        "tool_name": "search_shared_context",
                        "args": {"query": self._shared_context_query(obs.task["question"]), "k": 5},
                    }
                ]

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

    @staticmethod
    def _shared_context_query(question: str) -> str:
        id_match = re.search(r"\b(?:alias|user|post|thr|thread|org|loc|event)_[A-Za-z0-9_]+\b", question)
        if id_match:
            return id_match.group(0)
        path_match = re.search(r"relation path\s+(.+?),\s*which entity", question, flags=re.IGNORECASE)
        if path_match:
            first_relation = path_match.group(1).split("->", 1)[0].strip()
            if first_relation:
                return first_relation
        tokens = re.findall(r"[A-Za-z0-9_]+", question)
        return tokens[0] if tokens else question
