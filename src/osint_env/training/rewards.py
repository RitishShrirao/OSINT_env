from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from osint_env.domain.models import CanonicalGraph, Edge, TaskInstance
from osint_env.env.reward import build_reward_model, compute_answer_reward
from osint_env.training.config import GeneratorRewardWeights


def decode_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: list[str] = []
        for item in completion:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("content", "")))
        return "\n".join(part for part in parts if part)
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _extract_json_blob(text: str) -> Any:
    candidate = str(text or "").strip()
    if not candidate:
        return None
    left = candidate.find("{")
    right = candidate.rfind("}")
    if left >= 0 and right > left:
        snippet = candidate[left : right + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None
    return None


def normalize_answer(text: str) -> str:
    value = str(text or "").strip()
    value = value.strip('"').strip("'")
    value = re.sub(r"\s+", " ", value)
    value = value.rstrip(".\n ")
    return value


def extract_answer_from_completion(completion_text: str) -> str:
    blob = _extract_json_blob(completion_text)
    if isinstance(blob, dict):
        answer = str(blob.get("answer", "")).strip()
        if answer:
            return normalize_answer(answer)

    match = re.search(r"answer\s*[:=]\s*(.+)", completion_text, flags=re.IGNORECASE)
    if match:
        return normalize_answer(match.group(1))

    lines = [line.strip() for line in completion_text.splitlines() if line.strip()]
    if not lines:
        return ""
    return normalize_answer(lines[-1])


@dataclass(slots=True)
class GeneratedTaskCandidate:
    question: str
    answer: str
    supporting_edges: list[Edge]
    task_type: str
    is_valid: bool



def parse_generated_task_completion(completion_text: str, max_support_edges: int = 8) -> GeneratedTaskCandidate:
    blob = _extract_json_blob(completion_text)

    question = ""
    answer = ""
    task_type = "adversarial_trace"
    supporting_edges: list[Edge] = []

    if isinstance(blob, dict):
        question = str(blob.get("question", "")).strip()
        answer = normalize_answer(str(blob.get("answer", "")).strip())
        task_type = str(blob.get("task_type", "adversarial_trace")).strip() or "adversarial_trace"
        raw_edges = blob.get("supporting_edges", [])
        if isinstance(raw_edges, list):
            for row in raw_edges[:max_support_edges]:
                if not isinstance(row, dict):
                    continue
                src = str(row.get("src", "")).strip()
                rel = str(row.get("rel", "")).strip()
                dst = str(row.get("dst", "")).strip()
                if not src or not rel or not dst:
                    continue
                try:
                    confidence = float(row.get("confidence", 1.0))
                except (TypeError, ValueError):
                    confidence = 1.0
                supporting_edges.append(Edge(src=src, rel=rel, dst=dst, confidence=confidence))

    if not question:
        line_match = re.search(r"question\s*[:=]\s*(.+)", completion_text, flags=re.IGNORECASE)
        if line_match:
            question = line_match.group(1).strip()
    if not answer:
        answer = extract_answer_from_completion(completion_text)

    is_valid = bool(question and answer)
    return GeneratedTaskCandidate(
        question=question,
        answer=answer,
        supporting_edges=supporting_edges,
        task_type=task_type,
        is_valid=is_valid,
    )



def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", str(text).lower()))



def _jaccard_similarity(left: str, right: str) -> float:
    a = _token_set(left)
    b = _token_set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


class AnswererJudge:
    """Lightweight frozen answerer used to score adversarial hardness."""

    def __init__(self, model_name_or_path: str, max_new_tokens: int = 48):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, Any] = {}
        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.bfloat16

        model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **model_kwargs)
        model.eval()

        self._model = model
        self._tokenizer = tokenizer

    @lru_cache(maxsize=2048)
    def answer(self, question: str) -> str:
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        import torch

        prompt = (
            "You are an OSINT answering model. "
            "Answer with only the final entity string.\n"
            f"Question: {question}\n"
            "Answer:"
        )

        tokenizer = self._tokenizer
        model = self._model
        encoded = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max(1, int(self.max_new_tokens)),
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output[0][encoded["input_ids"].shape[1] :]
        completion = tokenizer.decode(generated, skip_special_tokens=True)
        return normalize_answer(extract_answer_from_completion(completion))


class GeneratorRewardFunction:
    """Reward for the graph/question generation swarm in adversarial self-play."""

    def __init__(
        self,
        graph: CanonicalGraph,
        answerer_judge: AnswererJudge,
        weights: GeneratorRewardWeights,
        max_support_edges: int = 8,
    ):
        self.answerer_judge = answerer_judge
        self.weights = weights
        self.max_support_edges = max_support_edges
        self.graph_nodes = set(graph.nodes.keys())
        self.graph_edges = {(edge.src, edge.rel, edge.dst) for edge in graph.edges}
        self._seen_questions: list[str] = []

    def _validity_score(self, candidate: GeneratedTaskCandidate) -> float:
        score = 0.0
        if candidate.question:
            score += 0.4
        if candidate.answer:
            score += 0.4
        if len(candidate.supporting_edges) <= self.max_support_edges:
            score += 0.2
        return min(1.0, score)

    def _consistency_score(self, candidate: GeneratedTaskCandidate) -> float:
        if not candidate.question or not candidate.answer:
            return 0.0

        edge_consistency = 0.0
        if candidate.supporting_edges:
            matches = sum(
                1
                for edge in candidate.supporting_edges
                if (edge.src, edge.rel, edge.dst) in self.graph_edges
            )
            edge_consistency = matches / max(1, len(candidate.supporting_edges))

        answer_in_graph = 1.0 if candidate.answer in self.graph_nodes else 0.0
        answer_in_edges = 1.0 if any(
            candidate.answer in {edge.src, edge.dst} for edge in candidate.supporting_edges
        ) else 0.0

        question_mentions_graph_symbol = 1.0 if any(
            node_id in candidate.question for node_id in self.graph_nodes
        ) else 0.0

        return (
            0.45 * edge_consistency
            + 0.30 * max(answer_in_graph, answer_in_edges)
            + 0.25 * question_mentions_graph_symbol
        )

    def _diversity_score(self, question: str) -> float:
        if not self._seen_questions:
            return 1.0
        max_similarity = max(_jaccard_similarity(question, prior) for prior in self._seen_questions)
        return max(0.0, 1.0 - max_similarity)

    def _hardness_score(self, candidate: GeneratedTaskCandidate) -> float:
        if not candidate.is_valid:
            return -1.0
        predicted_answer = normalize_answer(self.answerer_judge.answer(candidate.question))
        target_answer = normalize_answer(candidate.answer)
        return 1.0 if predicted_answer != target_answer else -0.4

    def __call__(
        self,
        prompts: list[Any] | None = None,
        completions: list[Any] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        del prompts
        if completions is None:
            completions = list(kwargs.get("completions", []))
        rewards: list[float] = []
        for completion in completions:
            text = decode_completion_text(completion)
            candidate = parse_generated_task_completion(text, max_support_edges=self.max_support_edges)

            validity = self._validity_score(candidate)
            consistency = self._consistency_score(candidate)
            diversity = self._diversity_score(candidate.question) if candidate.question else 0.0
            hardness = self._hardness_score(candidate)

            reward = (
                self.weights.validity * validity
                + self.weights.hardness * hardness
                + self.weights.diversity * diversity
                + self.weights.consistency * consistency
            )
            rewards.append(float(max(-1.5, min(1.5, reward))))

            if candidate.question:
                self._seen_questions.append(candidate.question)
                if len(self._seen_questions) > 4096:
                    self._seen_questions = self._seen_questions[-2048:]

        return rewards


class AnswererRewardFunction:
    """Answer-swarm reward wrapper that reuses the environment answer reward logic."""

    def __init__(self, graph: CanonicalGraph):
        self.reward_model = build_reward_model(graph)

    @staticmethod
    def _parse_support_edges(value: Any) -> list[Edge]:
        payload = value
        if isinstance(value, str):
            try:
                payload = json.loads(value)
            except json.JSONDecodeError:
                payload = []

        out: list[Edge] = []
        if not isinstance(payload, list):
            return out
        for row in payload:
            if not isinstance(row, dict):
                continue
            src = str(row.get("src", "")).strip()
            rel = str(row.get("rel", "")).strip()
            dst = str(row.get("dst", "")).strip()
            if not src or not rel or not dst:
                continue
            try:
                confidence = float(row.get("confidence", 1.0))
            except (TypeError, ValueError):
                confidence = 1.0
            out.append(Edge(src=src, rel=rel, dst=dst, confidence=confidence))
        return out

    @staticmethod
    def _value_at(column: Any, index: int, default: Any) -> Any:
        if isinstance(column, list) and index < len(column):
            return column[index]
        return default

    @staticmethod
    def _extract_predicted_edges(completion_text: str, support_edges: list[Edge]) -> list[Edge]:
        text = completion_text.lower()
        matched: list[Edge] = []
        for edge in support_edges:
            if edge.src.lower() in text and edge.rel.lower() in text and edge.dst.lower() in text:
                matched.append(edge)
        return matched

    def __call__(
        self,
        prompts: list[Any],
        completions: list[Any],
        answer: list[Any] | None = None,
        question: list[Any] | None = None,
        supporting_edges_json: list[Any] | None = None,
        difficulty: list[Any] | None = None,
        **kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []

        for idx, completion in enumerate(completions):
            completion_text = decode_completion_text(completion)
            predicted_answer = extract_answer_from_completion(completion_text)

            target_answer = normalize_answer(str(self._value_at(answer, idx, "")))
            question_text = str(self._value_at(question, idx, "")).strip()
            if not question_text:
                question_text = str(self._value_at(prompts, idx, "")).strip()

            support_payload = self._value_at(supporting_edges_json, idx, [])
            support_edges = self._parse_support_edges(support_payload)
            difficulty_level = str(self._value_at(difficulty, idx, "hard")).strip() or "hard"

            task = TaskInstance(
                task_id=f"train_task_{idx}",
                task_type="adversarial_trace",
                question=question_text,
                answer=target_answer,
                supporting_edges=support_edges,
                metadata={"difficulty": difficulty_level},
            )
            pred_edges = self._extract_predicted_edges(completion_text, support_edges)
            breakdown = compute_answer_reward(
                proposed_answer=predicted_answer,
                task=task,
                pred_edges=pred_edges,
                tool_outputs=[],
                step_count=1,
                model=self.reward_model,
                difficulty=difficulty_level,
            )
            rewards.append(float(breakdown.total))

        return rewards
