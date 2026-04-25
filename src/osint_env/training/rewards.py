from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from osint_env.data.generator import (
    emit_swarm_v2_question,
    enumerate_swarm_v2_neighbors,
    select_swarm_v2_answer,
    trace_swarm_v2_path,
)
from osint_env.domain.models import CanonicalGraph, Edge, TaskInstance
from osint_env.env.reward import build_reward_model, compute_answer_reward
from osint_env.env.spawn_reward_hooks import parl_reward_breakdown
from osint_env.training.config import (
    GeneratorRewardWeights,
    SwarmV2SharedContextConfig,
    SwarmV2ValidationConfig,
)


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
class SwarmReplayToolCall:
    tool_name: str
    args: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SwarmOrchestratorTelemetry:
    spawn_count: int = 0
    finished_subtasks: int = 0
    critical_steps: int = 1
    breadth: int = 0
    depth: int = 0


@dataclass(slots=True)
class ReplayValidationResult:
    is_valid: bool
    reasons: list[str] = field(default_factory=list)
    duplicate_similarity: float = 0.0
    context_nodes: int = 0
    context_edges: int = 0
    unique_path_count: int = 0
    replayed_question: str = ""
    replayed_answer: str = ""
    replayed_edges: list[Edge] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "reasons": list(self.reasons),
            "duplicate_similarity": float(self.duplicate_similarity),
            "context_nodes": int(self.context_nodes),
            "context_edges": int(self.context_edges),
            "unique_path_count": int(self.unique_path_count),
            "replayed_question": self.replayed_question,
            "replayed_answer": self.replayed_answer,
            "replayed_edges": [
                {
                    "src": edge.src,
                    "rel": edge.rel,
                    "dst": edge.dst,
                    "confidence": float(edge.confidence),
                }
                for edge in self.replayed_edges
            ],
        }


def _parse_edge_rows(value: Any, max_support_edges: int) -> list[Edge]:
    if not isinstance(value, list):
        return []
    out: list[Edge] = []
    for row in value[:max_support_edges]:
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


def _parse_tool_trace(value: Any) -> list[SwarmReplayToolCall]:
    if not isinstance(value, list):
        return []
    out: list[SwarmReplayToolCall] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        tool_name = str(row.get("tool_name", row.get("tool", ""))).strip()
        args = row.get("args", {})
        output = row.get("output", {})
        if not tool_name:
            continue
        out.append(
            SwarmReplayToolCall(
                tool_name=tool_name,
                args=dict(args) if isinstance(args, dict) else {},
                output=dict(output) if isinstance(output, dict) else {},
            )
        )
    return out


def _parse_subagent_outputs(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for row in value:
        if isinstance(row, str):
            token = row.strip()
        elif isinstance(row, dict):
            token = str(row.get("content", row.get("summary", ""))).strip()
        else:
            token = str(row).strip()
        if token:
            out.append(token)
    return out


def _parse_orchestrator(value: Any) -> SwarmOrchestratorTelemetry:
    if not isinstance(value, dict):
        return SwarmOrchestratorTelemetry()
    return SwarmOrchestratorTelemetry(
        spawn_count=max(0, int(value.get("spawn_count", 0) or 0)),
        finished_subtasks=max(0, int(value.get("finished_subtasks", 0) or 0)),
        critical_steps=max(1, int(value.get("critical_steps", 1) or 1)),
        breadth=max(0, int(value.get("breadth", 0) or 0)),
        depth=max(0, int(value.get("depth", 0) or 0)),
    )


@dataclass(slots=True)
class GeneratedTaskCandidate:
    question: str
    answer: str
    supporting_edges: list[Edge]
    task_type: str
    is_valid: bool
    tool_trace: list[SwarmReplayToolCall] = field(default_factory=list)
    subagent_outputs: list[str] = field(default_factory=list)
    canonical_edges: list[Edge] = field(default_factory=list)
    canonical_nodes: list[str] = field(default_factory=list)
    orchestrator: SwarmOrchestratorTelemetry = field(default_factory=SwarmOrchestratorTelemetry)
    validation: dict[str, Any] = field(default_factory=dict)



def parse_generated_task_completion(completion_text: str, max_support_edges: int = 8) -> GeneratedTaskCandidate:
    blob = _extract_json_blob(completion_text)

    question = ""
    answer = ""
    task_type = "adversarial_trace"
    supporting_edges: list[Edge] = []
    tool_trace: list[SwarmReplayToolCall] = []
    subagent_outputs: list[str] = []
    canonical_edges: list[Edge] = []
    canonical_nodes: list[str] = []
    orchestrator = SwarmOrchestratorTelemetry()
    validation: dict[str, Any] = {}

    if isinstance(blob, dict):
        question = str(blob.get("question", "")).strip()
        answer = normalize_answer(str(blob.get("answer", "")).strip())
        task_type = str(blob.get("task_type", "adversarial_trace")).strip() or "adversarial_trace"
        supporting_edges = _parse_edge_rows(blob.get("supporting_edges", []), max_support_edges=max_support_edges)
        tool_trace = _parse_tool_trace(blob.get("tool_trace", []))
        subagent_outputs = _parse_subagent_outputs(blob.get("subagent_outputs", []))
        orchestrator = _parse_orchestrator(blob.get("orchestrator"))
        validation = dict(blob.get("validation", {})) if isinstance(blob.get("validation"), dict) else {}
        canonical_graph = blob.get("canonical_graph", {})
        if isinstance(canonical_graph, dict):
            canonical_nodes = [
                str(node_id).strip()
                for node_id in canonical_graph.get("nodes", [])
                if str(node_id).strip()
            ]
            canonical_edges = _parse_edge_rows(
                canonical_graph.get("edges", []),
                max_support_edges=max(1, max_support_edges * 4),
            )

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
        tool_trace=tool_trace,
        subagent_outputs=subagent_outputs,
        canonical_edges=canonical_edges,
        canonical_nodes=canonical_nodes,
        orchestrator=orchestrator,
        validation=validation,
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


def _distinct_ngram_ratio(texts: list[str], n: int = 2) -> float:
    tokens: list[str] = []
    for text in texts:
        tokens.extend(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
    if len(tokens) < n:
        return 0.0 if texts else 1.0
    ngrams = [tuple(tokens[idx : idx + n]) for idx in range(0, len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / max(1, len(ngrams))


class SwarmV2ReplayValidator:
    """Hard-gated replay validator for deterministic swarm_v2 generation."""

    def __init__(
        self,
        graph: CanonicalGraph,
        validation: SwarmV2ValidationConfig,
        shared_context: SwarmV2SharedContextConfig,
        seen_questions: list[str] | None = None,
    ):
        self.graph = graph
        self.validation = validation
        self.shared_context = shared_context
        self.seen_questions = list(seen_questions or [])
        self.graph_nodes = set(graph.nodes.keys())
        self.graph_edges = {(edge.src, edge.rel, edge.dst) for edge in graph.edges}
        self.outgoing: dict[str, list[Edge]] = {}
        for edge in graph.edges:
            self.outgoing.setdefault(edge.src, []).append(edge)

    def remember(self, question: str) -> None:
        token = str(question).strip()
        if not token:
            return
        self.seen_questions.append(token)
        if len(self.seen_questions) > 4096:
            self.seen_questions = self.seen_questions[-2048:]

    def _count_matching_paths(self, start: str, relations: list[str], answer: str, limit: int = 4) -> int:
        if not start or not relations:
            return 0

        count = 0
        stack: list[tuple[str, int, tuple[str, ...]]] = [(start, 0, (start,))]
        while stack:
            node_id, rel_idx, seen_nodes = stack.pop()
            if rel_idx >= len(relations):
                if node_id == answer:
                    count += 1
                    if count >= limit:
                        return count
                continue

            relation = relations[rel_idx]
            for edge in self.outgoing.get(node_id, []):
                if edge.rel != relation:
                    continue
                if edge.dst in seen_nodes:
                    continue
                stack.append((edge.dst, rel_idx + 1, seen_nodes + (edge.dst,)))
        return count

    def _replay_tool_trace(self, candidate: GeneratedTaskCandidate) -> tuple[list[str], list[Edge], str, str]:
        reasons: list[str] = []
        replayed_edges: list[Edge] = []
        replayed_answer = ""
        replayed_question = ""

        if not candidate.tool_trace:
            return ["non_replayable_tool_calls"], replayed_edges, replayed_answer, replayed_question

        for call in candidate.tool_trace:
            if call.tool_name == "enumerate_neighbors":
                node_id = str(call.args.get("node_id", "")).strip()
                expected_edge = call.args.get("expected_edge", {})
                if not node_id:
                    reasons.append("non_replayable_tool_calls")
                    continue
                neighbors = enumerate_swarm_v2_neighbors(self.graph, node_id)
                if not neighbors:
                    reasons.append("non_replayable_tool_calls")
                if isinstance(expected_edge, dict):
                    expected_key = (
                        str(expected_edge.get("src", "")).strip(),
                        str(expected_edge.get("rel", "")).strip(),
                        str(expected_edge.get("dst", "")).strip(),
                    )
                    if expected_key not in {(edge.src, edge.rel, edge.dst) for edge in neighbors}:
                        reasons.append("non_replayable_tool_calls")
            elif call.tool_name == "trace_path":
                candidate_path = call.args.get("path", candidate.supporting_edges)
                replayed_edges = trace_swarm_v2_path(self.graph, candidate_path)
                if not replayed_edges:
                    reasons.append("non_replayable_tool_calls")
            elif call.tool_name == "select_answer":
                replayed_answer = select_swarm_v2_answer(replayed_edges)
                if not replayed_answer:
                    reasons.append("non_replayable_tool_calls")
            elif call.tool_name == "emit_question":
                replayed_question = emit_swarm_v2_question(replayed_edges)
                if not replayed_question:
                    reasons.append("non_replayable_tool_calls")
            else:
                reasons.append("non_replayable_tool_calls")

        return reasons, replayed_edges, replayed_answer, replayed_question

    def validate(self, candidate: GeneratedTaskCandidate) -> ReplayValidationResult:
        reasons: list[str] = []

        if not candidate.question or not candidate.answer:
            reasons.append("missing_question_or_answer")

        if not candidate.supporting_edges:
            reasons.append("malformed_support_edges")

        if len(candidate.supporting_edges) > self.validation.max_support_edges:
            reasons.append("context_or_support_budget_overflow")

        edge_keys = [(edge.src, edge.rel, edge.dst) for edge in candidate.supporting_edges]
        if len(set(edge_keys)) != len(edge_keys):
            reasons.append("malformed_support_edges")

        for edge in candidate.supporting_edges:
            if edge.src not in self.graph_nodes or edge.dst not in self.graph_nodes:
                reasons.append("unseen_nodes_or_edges")
                break
            if (edge.src, edge.rel, edge.dst) not in self.graph_edges:
                reasons.append("unseen_nodes_or_edges")
                break

        replay_reasons, replayed_edges, replayed_answer, replayed_question = self._replay_tool_trace(candidate)
        reasons.extend(replay_reasons)

        if replayed_edges:
            expected_keys = [(edge.src, edge.rel, edge.dst) for edge in replayed_edges]
            if expected_keys != edge_keys:
                reasons.append("non_replayable_tool_calls")
            relations = [edge.rel for edge in replayed_edges]
            unique_path_count = self._count_matching_paths(
                start=replayed_edges[0].src,
                relations=relations,
                answer=replayed_answer or candidate.answer,
            )
        else:
            unique_path_count = 0

        if unique_path_count != 1:
            reasons.append("non_unique_derivation_path")

        if replayed_answer and normalize_answer(replayed_answer) != normalize_answer(candidate.answer):
            reasons.append("non_replayable_tool_calls")

        if replayed_question and replayed_question != candidate.question:
            reasons.append("non_replayable_tool_calls")

        if candidate.answer and normalize_answer(candidate.answer).lower() in candidate.question.lower():
            reasons.append("answer_leakage")

        duplicate_similarity = 0.0
        if candidate.question and self.seen_questions:
            duplicate_similarity = max(
                _jaccard_similarity(candidate.question, seen_question)
                for seen_question in self.seen_questions
            )
            if duplicate_similarity >= self.validation.duplicate_similarity_threshold:
                reasons.append("duplicate_or_near_duplicate")

        context_nodes = len({edge.src for edge in candidate.supporting_edges} | {edge.dst for edge in candidate.supporting_edges})
        context_edges = len(candidate.supporting_edges)
        max_context_nodes = min(self.validation.max_context_nodes, self.shared_context.max_nodes)
        max_context_edges = min(self.validation.max_context_edges, self.shared_context.max_edges)
        if context_nodes > max_context_nodes or context_edges > max_context_edges:
            reasons.append("context_or_support_budget_overflow")

        if len(candidate.supporting_edges) > self.validation.max_path_hops:
            reasons.append("context_or_support_budget_overflow")

        return ReplayValidationResult(
            is_valid=not reasons,
            reasons=sorted(set(reasons)),
            duplicate_similarity=duplicate_similarity,
            context_nodes=context_nodes,
            context_edges=context_edges,
            unique_path_count=unique_path_count,
            replayed_question=replayed_question,
            replayed_answer=replayed_answer,
            replayed_edges=replayed_edges,
        )


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
        pipeline_mode: str = "legacy",
        swarm_v2_validation: SwarmV2ValidationConfig | None = None,
        swarm_v2_shared_context: SwarmV2SharedContextConfig | None = None,
        parl_max_parallel_hint: int = 0,
    ):
        self.graph = graph
        self.answerer_judge = answerer_judge
        self.weights = weights
        self.max_support_edges = max_support_edges
        self.pipeline_mode = str(pipeline_mode).strip().lower() or "legacy"
        self.graph_nodes = set(graph.nodes.keys())
        self.graph_edges = {(edge.src, edge.rel, edge.dst) for edge in graph.edges}
        self._seen_questions: list[str] = []
        self.swarm_v2_validation = swarm_v2_validation or SwarmV2ValidationConfig(
            max_support_edges=max_support_edges
        )
        self.swarm_v2_shared_context = swarm_v2_shared_context or SwarmV2SharedContextConfig()
        self.parl_max_parallel_hint = max(0, int(parl_max_parallel_hint or 0))
        self._swarm_v2_validator = SwarmV2ReplayValidator(
            graph=graph,
            validation=self.swarm_v2_validation,
            shared_context=self.swarm_v2_shared_context,
            seen_questions=self._seen_questions,
        )
        self._debug_batches_seen = 0
        self._debug_reason_counter: Counter[str] = Counter()
        self._debug_reward_window: list[float] = []
        self._debug_last_batch: dict[str, Any] = {}

    @staticmethod
    def _std(values: list[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return variance ** 0.5

    def _invalid_swarm_v2_reward(
        self,
        candidate: GeneratedTaskCandidate,
        validation_result: ReplayValidationResult,
    ) -> float:
        # Avoid a constant hard penalty. Keep invalid samples negative but
        # graded so GRPO still gets reward variance/advantages when quality
        # differs. Scale is intentionally wider than the original [-1.35]
        # constant path:
        #   malformed/no JSON ~= -2.0
        #   partial structured JSON ~= -1.2 .. -0.4
        #   replayable but imperfect candidates are handled by valid path.
        reason_penalty = {
            "missing_question_or_answer": 0.55,
            "malformed_support_edges": 0.40,
            "non_replayable_tool_calls": 0.55,
            "non_unique_derivation_path": 0.30,
            "unseen_nodes_or_edges": 0.35,
            "answer_leakage": 0.45,
            "duplicate_or_near_duplicate": 0.20,
            "context_or_support_budget_overflow": 0.25,
        }
        penalty = 0.35
        for reason in validation_result.reasons:
            penalty += reason_penalty.get(reason, 0.10)

        # Partial credit for parseable structure to reduce flat rewards.
        partial_credit = 0.0
        if candidate.question:
            partial_credit += 0.25
        if candidate.answer:
            partial_credit += 0.25
        if candidate.supporting_edges:
            partial_credit += min(0.35, 0.08 * len(candidate.supporting_edges))
        if candidate.tool_trace:
            partial_credit += min(0.30, 0.06 * len(candidate.tool_trace))
        if candidate.subagent_outputs:
            partial_credit += 0.10
        if candidate.canonical_edges or candidate.canonical_nodes:
            partial_credit += 0.10

        reward = partial_credit - penalty
        return float(max(-2.0, min(-0.05, reward)))

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

    @staticmethod
    def _support_path_coverage(candidate: GeneratedTaskCandidate) -> float:
        if not candidate.supporting_edges:
            return 0.0
        keys = {(edge.src, edge.rel, edge.dst) for edge in candidate.supporting_edges}
        return len(keys) / max(1, len(candidate.supporting_edges))

    def _swarm_diversity_score(self, candidate: GeneratedTaskCandidate) -> float:
        if not candidate.subagent_outputs:
            return 0.0
        distinct_ratio = _distinct_ngram_ratio(candidate.subagent_outputs, n=2)
        path_coverage = self._support_path_coverage(candidate)
        return max(0.0, min(1.0, (0.7 * distinct_ratio) + (0.3 * path_coverage)))

    def _context_pressure_score(self, validation_result: ReplayValidationResult) -> float:
        if not validation_result.is_valid:
            return 0.0

        node_util = validation_result.context_nodes / max(1, self.swarm_v2_shared_context.max_nodes)
        edge_util = validation_result.context_edges / max(1, self.swarm_v2_shared_context.max_edges)
        utilization = max(node_util, edge_util)
        target = max(0.05, float(self.swarm_v2_shared_context.target_pressure))
        if utilization > 1.0:
            return 0.0
        gap = abs(utilization - target)
        return max(0.0, 1.0 - (gap / max(target, 1.0 - target)))

    def _parl_scores(self, candidate: GeneratedTaskCandidate) -> tuple[float, float]:
        breakdown = parl_reward_breakdown(
            task_outcome_reward=0.0,
            spawn_count=candidate.orchestrator.spawn_count,
            finished_subtasks=candidate.orchestrator.finished_subtasks,
            critical_steps=candidate.orchestrator.critical_steps,
            lambda_parallel=0.15,
            lambda_finish=0.20,
            anneal=1.0,
            breadth=candidate.orchestrator.breadth,
            depth=candidate.orchestrator.depth,
            max_parallel_hint=self.parl_max_parallel_hint,
        )
        return breakdown.parallel, breakdown.finish

    def _swarm_v2_reward(self, candidate: GeneratedTaskCandidate) -> tuple[float, ReplayValidationResult]:
        validator = self._swarm_v2_validator
        validator.seen_questions = list(self._seen_questions)
        validation_result = validator.validate(candidate)
        if not validation_result.is_valid:
            return self._invalid_swarm_v2_reward(candidate, validation_result), validation_result

        hardness = self._hardness_score(candidate)
        swarm_diversity = self._swarm_diversity_score(candidate)
        context_pressure = self._context_pressure_score(validation_result)
        parl_parallel, parl_finish = self._parl_scores(candidate)

        reward = (
            0.25  # valid JSON/schema
            + 0.30  # replayable derivation
            + (0.30 * hardness)
            + (0.15 * swarm_diversity)
            + (0.10 * context_pressure)
            + (0.025 * parl_parallel)
            + (0.025 * parl_finish)
        )
        return reward, validation_result

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
        batch_reasons: Counter[str] = Counter()
        valid_count = 0
        for completion in completions:
            text = decode_completion_text(completion)
            candidate = parse_generated_task_completion(text, max_support_edges=self.max_support_edges)

            if self.pipeline_mode == "swarm_v2":
                reward, validation_result = self._swarm_v2_reward(candidate)
                rewards.append(float(max(-2.0, min(1.2, reward))))
                if validation_result.is_valid and candidate.question:
                    valid_count += 1
                    self._seen_questions.append(candidate.question)
                    if len(self._seen_questions) > 4096:
                        self._seen_questions = self._seen_questions[-2048:]
                else:
                    for reason in validation_result.reasons:
                        batch_reasons[reason] += 1
            else:
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
                rewards.append(float(max(-2.0, min(1.2, reward))))

            if self.pipeline_mode != "swarm_v2" and candidate.question:
                self._seen_questions.append(candidate.question)
                if len(self._seen_questions) > 4096:
                    self._seen_questions = self._seen_questions[-2048:]

        self._debug_batches_seen += 1
        self._debug_reward_window.extend(rewards)
        self._debug_reward_window = self._debug_reward_window[-512:]
        self._debug_reason_counter.update(batch_reasons)
        batch_mean = float(sum(rewards) / max(1, len(rewards)))
        batch_std = float(self._std(rewards))
        advantages = [float(value - batch_mean) for value in rewards]
        self._debug_last_batch = {
            "batch_rewards": list(rewards),
            "batch_reward_mean": batch_mean,
            "batch_reward_std": batch_std,
            "advantage_proxy_min": min(advantages) if advantages else 0.0,
            "advantage_proxy_max": max(advantages) if advantages else 0.0,
            "advantage_proxy_std": float(self._std(advantages)),
            "valid_count": int(valid_count),
            "invalid_count": int(max(0, len(rewards) - valid_count)),
            "valid_output_ratio": float(valid_count / max(1, len(rewards))),
            "top_invalid_reasons": batch_reasons.most_common(5),
        }
        if self.pipeline_mode == "swarm_v2" and (self._debug_batches_seen % 10 == 0):
            window_std = self._std(self._debug_reward_window)
            print(
                "[reward_debug][generator] "
                f"batches={self._debug_batches_seen} "
                f"window_reward_std={window_std:.6f} "
                f"last_batch_valid={valid_count}/{len(rewards)} "
                f"top_invalid_reasons={batch_reasons.most_common(3)}"
            )

        return rewards


class AnswererRewardFunction:
    """Answer-swarm reward wrapper that reuses the environment answer reward logic."""

    def __init__(
        self,
        graph: CanonicalGraph,
        pipeline_mode: str = "legacy",
        parl_max_parallel_hint: int = 0,
    ):
        self.reward_model = build_reward_model(graph)
        self.pipeline_mode = str(pipeline_mode).strip().lower() or "legacy"
        self.parl_max_parallel_hint = max(0, int(parl_max_parallel_hint or 0))

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
        blob = _extract_json_blob(completion_text)
        if isinstance(blob, dict):
            structured_edges = _parse_edge_rows(blob.get("supporting_edges", []), max_support_edges=len(support_edges))
            if structured_edges:
                return structured_edges
        text = completion_text.lower()
        matched: list[Edge] = []
        for edge in support_edges:
            if edge.src.lower() in text and edge.rel.lower() in text and edge.dst.lower() in text:
                matched.append(edge)
        return matched

    def _extract_orchestrator_reward(self, completion_text: str, base_reward: float) -> float:
        if self.pipeline_mode != "swarm_v2":
            return float(base_reward)
        blob = _extract_json_blob(completion_text)
        orchestrator = _parse_orchestrator(blob.get("orchestrator")) if isinstance(blob, dict) else SwarmOrchestratorTelemetry()
        breakdown = parl_reward_breakdown(
            task_outcome_reward=base_reward,
            spawn_count=orchestrator.spawn_count,
            finished_subtasks=orchestrator.finished_subtasks,
            critical_steps=orchestrator.critical_steps,
            lambda_parallel=0.15,
            lambda_finish=0.20,
            anneal=1.0,
            breadth=orchestrator.breadth,
            depth=orchestrator.depth,
            max_parallel_hint=self.parl_max_parallel_hint,
        )
        return float(breakdown.total)

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
            rewards.append(self._extract_orchestrator_reward(completion_text, breakdown.total))

        return rewards
