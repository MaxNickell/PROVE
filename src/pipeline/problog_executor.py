"""
ProbLog executor for PROVE pipeline.
Executes ProbLog queries using LLM-generated rules.
"""

from typing import List, Dict, Tuple
import re

from problog.program import PrologString
from problog import get_evaluatable

from src.core.model_manager import ModelManager
from src.core.types import ProbLogFact, ModeResult
from src.pipeline.problog_builder import ProbLogFactBuilder


class ProbLogExecutor:
    """
    Execute ProbLog reasoning for a question.

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
        question: str,
        evidence: 'EvidenceCollection',
        images: Dict[str, 'ImageData'],
        threshold: float = 0.5
    ) -> Tuple[ModeResult, ModeResult]:
        """
        Execute question in both probabilistic and deterministic modes.

        Args:
            question: The question to answer
            evidence: Evidence collection for the question
            images: ImageData for entity metadata
            threshold: Threshold for deterministic mode

        Returns:
            (probabilistic_result, deterministic_result)
        """
        llm = self.model_manager.get_llm_client()

        print(f"\n{'='*60}")
        print(f"ProbLog Execution (threshold={threshold})")
        print(f"Question: {question}")
        print(f"{'='*60}")

        # Build probabilistic facts
        prob_facts = self.fact_builder.build_facts(evidence, images)
        print(f"  Facts: {len(prob_facts)}")

        # Generate rules + query (once, reuse for both modes)
        rules, query = self._generate_query(question, prob_facts, llm)

        # Build deterministic facts
        det_facts = ProbLogFactBuilder.threshold_facts(prob_facts, threshold)

        # Execute both
        prob_prob = self._execute_program(prob_facts, rules, query, threshold)
        det_prob = self._execute_program(det_facts, rules, query, threshold)

        print(f"  Probabilistic: {prob_prob:.4f}")
        print(f"  Deterministic: {det_prob:.4f}")

        # Convert probability to answer
        prob_answer = "True" if prob_prob >= 0.5 else "False"
        det_answer = "True" if det_prob >= 0.5 else "False"

        print(f"  Probabilistic answer: {prob_answer}")
        print(f"  Deterministic answer: {det_answer}")

        return (
            ModeResult(
                probability=prob_prob,
                final_answer=prob_answer,
                problog_program=self._build_program_string(prob_facts, rules, query)
            ),
            ModeResult(
                probability=det_prob,
                final_answer=det_answer,
                problog_program=self._build_program_string(det_facts, rules, query)
            )
        )

    def _generate_query(
        self,
        question: str,
        facts: List[ProbLogFact],
        llm
    ) -> Tuple[str, str]:
        """LLM generates ProbLog rules and query for the question."""

        facts_str = ProbLogFactBuilder.facts_to_string(facts)

        prompt = f"""Generate ProbLog rules and query to answer this question.

PREDICATES:
- entity(image_id, entity_id, category)
- attribute(image_id, entity_id, value)
- relation(image_id, subject_id, object_id, relation_type)

COUNT PREDICATES (use directly in rules):
- count_at_least(image_id, class, N) - at least N objects in image
- count_at_most(image_id, class, N) - at most N objects in image
- count_exactly(image_id, class, N) - exactly N objects in image
- count_more(image_id_a, image_id_b, class) - more in A than B
- count_fewer(image_id_a, image_id_b, class) - fewer in A than B
- count_equal(image_id_a, image_id_b, class) - same count in both
- count_total_exactly(image_id_a, image_id_b, class, N) - exactly N total
- count_total_at_least(image_id_a, image_id_b, class, N) - at least N total
- count_total_at_most(image_id_a, image_id_b, class, N) - at most N total

AVAILABLE FACTS:
{facts_str}

IMPORTANT RULES:
- CLOSED WORLD: You may ONLY reference predicates that appear in the AVAILABLE FACTS above. Do NOT generate rules that use predicates not present in the facts.
- SELECTIVE USAGE: You do NOT need to use all facts. Only use facts that are relevant to answering the question.
- INCOMPLETE EVIDENCE: If the evidence is incomplete, write rules using only the facts that ARE available. Do not reference facts that do not exist.

QUESTION: {question}

Generate ONLY:
1. Rule definition(s) using :- syntax
2. query(...) statement

NOTE: Multi-word values in facts are quoted with single quotes (e.g., 'showing teeth'). You must use the same quoting in your rules.

Example output:
dog_is_orange(I) :-
    entity(I, E, dog),
    attribute(I, E, orange).
query(dog_is_orange(image_a)).

Example with quoted atoms:
bird_on_branch(I) :-
    entity(I, E1, bird),
    entity(I, E2, branch),
    relation(I, E1, E2, 'on top of').
query(bird_on_branch(image_a)).

Output rules and query only, no explanation:"""

        messages = [
            {"role": "system", "content": "Generate valid ProbLog syntax only. No markdown, no explanations."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = llm.chat(messages, temperature=0.0)
            rules, query = self._parse_response(response)

            if not query.startswith("query("):
                return self._fallback_query(question)

            return rules, query

        except Exception as e:
            print(f"  Warning: Query generation failed: {e}")
            return self._fallback_query(question)

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

    def _fallback_query(self, question: str) -> Tuple[str, str]:
        """Generate fallback query when LLM fails."""
        # Extract image from question
        match = re.search(r'image[_ ]([ab])', question.lower())
        image_id = f"image_{match.group(1)}" if match else "image_a"

        rules = "fallback(I) :- entity(I, _, _)."
        query = f"query(fallback({image_id}))."
        return rules, query

    def _execute_program(
        self,
        facts: List[ProbLogFact],
        rules: str,
        query: str,
        threshold: float = 0.5
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
            return threshold

    def _build_program_string(
        self,
        facts: List[ProbLogFact],
        rules: str,
        query: str
    ) -> str:
        """Build complete ProbLog program."""
        facts_str = ProbLogFactBuilder.facts_to_string(facts)
        return f"{facts_str}\n\n{rules}\n\n{query}"
