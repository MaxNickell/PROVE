"""
ProbLog executor for PROVE pipeline.
Executes ProbLog queries using LLM-generated rules.
"""

from typing import List, Dict, Tuple
import re

from problog.program import PrologString
from problog import get_evaluatable

from src.core.model_manager import ModelManager
from src.core.types import BinarySubquestion, ProbLogFact, SubquestionResult, ModeResult
from src.pipeline.problog_builder import ProbLogFactBuilder


class ProbLogExecutor:
    """
    Execute ProbLog reasoning for subquestions.

    Flow:
    1. Build facts from evidence
    2. LLM generates rules + query
    3. Execute ProbLog program
    4. Return probability
    """

    def __init__(self):
        self.model_manager = ModelManager()
        self.fact_builder = ProbLogFactBuilder()

    def execute_dual(
        self,
        subquestions: List[BinarySubquestion],
        evidence_collections: List['EvidenceCollection'],
        images: Dict[str, 'ImageData'],
        ultimate_question: str,
        threshold: float = 0.5
    ) -> Tuple[ModeResult, ModeResult]:
        """
        Execute all subquestions in both probabilistic and deterministic modes.

        Args:
            subquestions: List of binary subquestions
            evidence_collections: Evidence for each subquestion (1:1)
            images: ImageData for entity metadata
            ultimate_question: Original question
            threshold: Threshold for deterministic mode

        Returns:
            (probabilistic_result, deterministic_result)
        """
        if not subquestions:
            empty = ModeResult(subquestion_results=[], final_answer="False", problog_program="")
            return empty, empty

        if len(evidence_collections) != len(subquestions):
            raise RuntimeError(f"Evidence mismatch: {len(evidence_collections)} vs {len(subquestions)}")

        llm = self.model_manager.get_llm_client()
        prob_results, det_results = [], []

        print(f"\n{'='*60}")
        print(f"ProbLog Execution ({len(subquestions)} subquestions, threshold={threshold})")
        print(f"{'='*60}")

        for i, (sq, evidence) in enumerate(zip(subquestions, evidence_collections), 1):
            print(f"\n[{i}/{len(subquestions)}] {sq.question}")

            # Build probabilistic facts
            prob_facts = self.fact_builder.build_facts(evidence, images)
            print(f"  Facts: {len(prob_facts)}")

            # Generate rules + query (once, reuse for both modes)
            rules, query = self._generate_query(sq, prob_facts, llm)

            # Build deterministic facts
            det_facts = ProbLogFactBuilder.threshold_facts(prob_facts, threshold)

            # Execute both
            prob_prob = self._execute_program(prob_facts, rules, query)
            det_prob = self._execute_program(det_facts, rules, query)

            print(f"  Probabilistic: {prob_prob:.4f}")
            print(f"  Deterministic: {det_prob:.4f}")

            # Store results
            prob_results.append(SubquestionResult(
                subquestion=sq.question,
                probability=prob_prob,
                supporting_facts=prob_facts,
                problog_program=self._build_program_string(prob_facts, rules, query),
                evidence_trail=[f"prob: {prob_prob:.4f}"]
            ))

            det_results.append(SubquestionResult(
                subquestion=sq.question,
                probability=det_prob,
                supporting_facts=det_facts,
                problog_program=self._build_program_string(det_facts, rules, query),
                evidence_trail=[f"det: {det_prob:.4f}"]
            ))

        # Compose ultimate answers
        print(f"\n{'='*60}")
        print(f"Ultimate Composition: {ultimate_question}")
        print(f"{'='*60}")

        prob_answer = self._compose_ultimate(prob_results, ultimate_question, llm)
        det_answer = self._compose_ultimate(det_results, ultimate_question, llm)

        print(f"  Probabilistic: {prob_answer}")
        print(f"  Deterministic: {det_answer}")

        return (
            ModeResult(
                subquestion_results=prob_results,
                final_answer=prob_answer,
                problog_program=self._build_unified_program(prob_results)
            ),
            ModeResult(
                subquestion_results=det_results,
                final_answer=det_answer,
                problog_program=self._build_unified_program(det_results)
            )
        )

    def _generate_query(
        self,
        subquestion: BinarySubquestion,
        facts: List[ProbLogFact],
        llm
    ) -> Tuple[str, str]:
        """LLM generates ProbLog rules and query for a subquestion."""

        facts_str = ProbLogFactBuilder.facts_to_string(facts)

        prompt = f"""Generate ProbLog rules and query to answer this subquestion.

PREDICATES:
- entity(image_id, entity_id, category)
- attribute(image_id, entity_id, value)
- relation(image_id, subject_id, object_id, relation_type)
- count(image_id, category, count_value)

SUGAR RULES (available):
- has_attribute(I,E,A) :- attribute(I,E,A).
- is_category(I,E,C) :- entity(I,E,C).
- has_relationship(I,A,B,R) :- relation(I,A,B,R).

AVAILABLE FACTS:
{facts_str}

SUBQUESTION: {subquestion.question}

Generate ONLY:
1. Rule definition(s) using :- syntax
2. query(...) statement

Example output:
dog_is_orange(I) :-
    is_category(I, E, dog),
    has_attribute(I, E, orange).
query(dog_is_orange(image_a)).

Output rules and query only, no explanation:"""

        messages = [
            {"role": "system", "content": "Generate valid ProbLog syntax only. No markdown, no explanations."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = llm.chat(messages, temperature=0.2)
            rules, query = self._parse_response(response)

            if not query.startswith("query("):
                return self._fallback_query(subquestion)

            return rules, query

        except Exception as e:
            print(f"  Warning: Query generation failed: {e}")
            return self._fallback_query(subquestion)

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """Parse LLM response into rules and query."""
        # Remove markdown
        response = re.sub(r'```(?:prolog|problog)?\n?', '', response)
        response = response.replace('`', '').strip()

        rules_lines, query_lines = [], []

        for line in response.split('\n'):
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.startswith('query('):
                query_lines.append(line)
            else:
                rules_lines.append(line)

        return '\n'.join(rules_lines), '\n'.join(query_lines)

    def _fallback_query(self, subquestion: BinarySubquestion) -> Tuple[str, str]:
        """Generate fallback query when LLM fails."""
        # Extract image from question
        match = re.search(r'image[_ ]([ab])', subquestion.question.lower())
        image_id = f"image_{match.group(1)}" if match else "image_a"

        rules = "fallback(I) :- entity(I, _, _)."
        query = f"query(fallback({image_id}))."
        return rules, query

    def _execute_program(
        self,
        facts: List[ProbLogFact],
        rules: str,
        query: str
    ) -> float:
        """Execute ProbLog program and return query probability."""
        program = self._build_program_string(facts, rules, query)

        try:
            result = get_evaluatable().create_from(PrologString(program)).evaluate()

            # Extract query result
            query_match = re.search(r'query\(([^)]+\([^)]*\))\)', query)
            if query_match:
                query_term = query_match.group(1)
                for key in result.keys():
                    if str(key) == query_term or query_term in str(key):
                        return float(result[key])

            return 0.0

        except Exception as e:
            print(f"  Warning: ProbLog execution failed: {e}")
            return 0.5

    def _build_program_string(
        self,
        facts: List[ProbLogFact],
        rules: str,
        query: str
    ) -> str:
        """Build complete ProbLog program."""
        facts_str = ProbLogFactBuilder.facts_to_string(facts)
        return f"{ProbLogFactBuilder.SUGAR_RULES}\n\n{facts_str}\n\n{rules}\n\n{query}"

    def _build_unified_program(self, results: List[SubquestionResult]) -> str:
        """Build unified program for logging."""
        parts = [ProbLogFactBuilder.SUGAR_RULES, ""]
        for i, r in enumerate(results, 1):
            parts.append(f"% [{i}] {r.subquestion} → {r.probability:.4f}")
            parts.append(r.problog_program)
            parts.append("")
        return "\n".join(parts)

    def _compose_ultimate(
        self,
        results: List[SubquestionResult],
        ultimate_question: str,
        llm
    ) -> str:
        """LLM composes final answer from subquestion results."""

        # Build summary of subquestion answers
        answers = []
        for r in results:
            answer = "TRUE" if r.probability >= 0.5 else "FALSE"
            answers.append(f"- {r.subquestion} → {answer} (p={r.probability:.3f})")

        prompt = f"""Given these subquestion answers:

{chr(10).join(answers)}

Answer this question: {ultimate_question}

Reply with ONLY 'True' or 'False'."""

        messages = [
            {"role": "system", "content": "Answer with only 'True' or 'False'. No explanation."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = llm.chat(messages, temperature=0.0).strip().lower()
            return "True" if "true" in response else "False"
        except Exception:
            return "False"
