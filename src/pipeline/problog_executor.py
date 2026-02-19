"""
ProbLog executor for PROVE pipeline.
Executes ProbLog queries using LLM-generated rules.
Matches sweep's robustness: subprocess isolation, timeouts.
"""

from typing import List, Dict, Tuple
import multiprocessing as mp
import re

from problog.program import PrologString
from problog import get_evaluatable

from src.core.model_manager import ModelManager
from src.core.types import ProbLogFact, ModeResult
from src.pipeline.problog_builder import ProbLogFactBuilder

# ProbLog built-in predicates that LLMs sometimes redefine as helpers
_RESERVED_PREDICATES = ['condition']

# Default timeout for ProbLog evaluation (seconds)
_PROBLOG_TIMEOUT = 30

# Default timeout for LLM calls (seconds)
_LLM_TIMEOUT = 120


def _problog_worker(program, queue):
    """Evaluate a ProbLog program in a subprocess (memory-isolated)."""
    try:
        result = get_evaluatable().create_from(PrologString(program)).evaluate()
        for k, v in result.items():
            queue.put(float(v))
            return
        queue.put(0.0)
    except Exception as e:
        queue.put(('error', str(e)))


def sanitize_program(program):
    """Rename LLM-generated predicates that clash with ProbLog built-ins."""
    for pred in _RESERVED_PREDICATES:
        program = re.sub(rf'\b{pred}\b', f'{pred}_', program)
    return program


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
        threshold: float = 0.5,
        ices: list = None
    ) -> Tuple[ModeResult, ModeResult]:
        """
        Execute question in both probabilistic and deterministic modes.

        Args:
            question: The question to answer
            evidence: Evidence collection for the question
            threshold: Threshold for final answer (prob >= threshold -> "True")
            images: ImageData for entity metadata

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
        rules, query = self._generate_query(question, prob_facts, llm, ices=ices)

        # Build deterministic facts (always threshold at 0.5)
        det_facts = ProbLogFactBuilder.threshold_facts(prob_facts, 0.5)

        # Execute both in subprocesses with timeout
        prob_prob, prob_err = self._execute_program(prob_facts, rules, query, threshold)
        det_prob, det_err = self._execute_program(det_facts, rules, query, threshold)

        if prob_err:
            print(f"  Warning: Probabilistic ProbLog failed: {prob_err}")
        if det_err:
            print(f"  Warning: Deterministic ProbLog failed: {det_err}")

        print(f"  Probabilistic: {prob_prob:.4f}")
        print(f"  Deterministic: {det_prob:.4f}")

        # Convert probability to answer using configurable threshold
        prob_answer = "True" if prob_prob >= threshold else "False"
        det_answer = "True" if det_prob >= threshold else "False"

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

    @staticmethod
    def _build_prompt(question: str, facts_str: str, ices: list = None) -> str:
        """Build the ProbLog generation prompt.

        Dynamically detects which count predicates appear in the facts
        and only advertises those, preventing the LLM from inventing
        predicates that don't exist. Also adapts examples based on
        how many images are present in the facts.
        """
        # Detect which image IDs appear in the facts
        import re
        image_ids = sorted(set(re.findall(r'image_[a-z]', facts_str)))
        if not image_ids:
            image_ids = ['image_a']  # fallback
        multi_image = len(image_ids) > 1

        # Detect which count predicates actually appear in the facts
        count_pred_descriptions = {
            'count_at_least': 'count_at_least(image_id, class, N) - at least N objects in image',
            'count_at_most': 'count_at_most(image_id, class, N) - at most N objects in image',
            'count_exactly': 'count_exactly(image_id, class, N) - exactly N objects in image',
            'count_more': 'count_more(image_id_a, image_id_b, class) - more in A than B',
            'count_fewer': 'count_fewer(image_id_a, image_id_b, class) - fewer in A than B',
            'count_equal': 'count_equal(image_id_a, image_id_b, class) - same count in both',
            'count_total_exactly': 'count_total_exactly(image_id_a, image_id_b, class, N) - exactly N total',
            'count_total_at_least': 'count_total_at_least(image_id_a, image_id_b, class, N) - at least N total',
            'count_total_at_most': 'count_total_at_most(image_id_a, image_id_b, class, N) - at most N total',
        }
        available_counts = []
        for pred, desc in count_pred_descriptions.items():
            if pred + '(' in facts_str:
                available_counts.append(f'- {desc}')

        if available_counts:
            count_section = "COUNT PREDICATES (available in the facts above):\n" + '\n'.join(available_counts) + "\n\n"
        else:
            count_section = ""

        # Detect if facts are count-only (no entity/attribute/relation facts)
        has_entity = 'entity(' in facts_str
        has_attribute = 'attribute(' in facts_str
        has_relation = 'relation(' in facts_str
        count_only = available_counts and not has_entity and not has_attribute and not has_relation

        if count_only:
            predicate_section = """WARNING: The facts below contain ONLY count predicates. Do NOT use entity(), attribute(), or relation() — they are not available. Build your answer rule using ONLY the count predicates listed below."""
        else:
            # Only list predicates that actually appear in the facts
            pred_lines = []
            missing_preds = []
            if has_entity:
                pred_lines.append('- entity(image_id, entity_id, category)          — ALWAYS 3 arguments')
            else:
                missing_preds.append('entity')
            if has_attribute:
                pred_lines.append('- attribute(image_id, entity_id, value)           — ALWAYS 3 arguments (NOT 2)')
            else:
                missing_preds.append('attribute')
            if has_relation:
                pred_lines.append('- relation(image_id, subject_id, object_id, relation_type) — ALWAYS 4 arguments (NOT 3)')
            else:
                missing_preds.append('relation')

            predicate_section = "PREDICATES (with EXACT arities — you MUST match these):\n" + '\n'.join(pred_lines)
            if missing_preds:
                predicate_section += f"\n\nWARNING: The following predicates are NOT available in the facts: {', '.join(missing_preds)}. Do NOT use them."

        # Build answer pattern guidance based on image count
        if multi_image:
            answer_pattern = "- ANSWER PATTERN: Always define an `answer` rule and end with `query(answer).`. For per-image conditions, write helper rule(s) with variable I, then combine in `answer :- ...`. For cross-image count predicates (count_total_*, count_more, count_fewer, count_equal), use them DIRECTLY in the answer rule — they already span both images. Never generate more than one query statement."
            connectives = """- LOGICAL CONNECTIVES: Choose connectives based on the question's logical meaning:
  - `,` (AND/conjunction): ALL conditions must hold. Use when the question requires EVERY image to satisfy the condition (e.g., "all images show X", "both images have X", "each image contains X").
  - `;` (OR/disjunction): AT LEAST ONE condition must hold. Use when the question requires ANY image to satisfy the condition (e.g., "at least one X is Y", "there is an X that is Y", "either image shows X").
  - Parenthesized mix: Use `(cond_a ; cond_b)` within a larger conjunction when part of the question is universal and part is existential (e.g., "same count AND at least one does X").
  Analyze what the question actually requires — do not default to AND or OR without reasoning about the meaning."""
        else:
            img = image_ids[0]
            answer_pattern = f"- ANSWER PATTERN: Always define an `answer` rule and end with `query(answer).`. Write helper rule(s) with variable I, then use `answer :- helper({img}).`. Never generate more than one query statement."
            connectives = """- LOGICAL CONNECTIVES: Use `,` (AND) to combine multiple conditions that must all hold. Use `;` (OR) when at least one condition suffices."""

        # Build example patterns based on image count
        # (All examples are real NLVR2 training programs from the ICE pool)
        img_a = image_ids[0]
        if multi_image:
            img_b = image_ids[1]
            examples = f"""EXAMPLE PATTERNS:

Single image condition:
school_bus_facing_right(I) :-
    entity(I, E, 'school bus'),
    attribute(I, E, 'facing right').
answer :- school_bus_facing_right({img_b}).
query(answer).

Both images must satisfy condition → AND (,):
full_round_pizza(I) :-
    entity(I, E, pizza),
    attribute(I, E, full),
    attribute(I, E, round).
answer :- full_round_pizza({img_a}), full_round_pizza({img_b}).
query(answer).

At least one image satisfies condition → OR (;):
panda_playing_bubble(I) :-
    entity(I, E1, panda),
    entity(I, E2, bubble),
    relation(I, E1, E2, 'playing with').
answer :- panda_playing_bubble({img_a}) ; panda_playing_bubble({img_b}).
query(answer).

Cross-image count (use directly, do NOT wrap in per-image helper):
answer :- count_total_exactly({img_a}, {img_b}, flute, 2).
query(answer).

Mixed per-image + cross-image count:
adult_leopard(I) :-
    entity(I, E, leopard),
    attribute(I, E, adult).
answer :-
    count_total_at_least({img_a}, {img_b}, cub, 3),
    (adult_leopard({img_a}) ; adult_leopard({img_b})).
query(answer)."""
        else:
            examples = f"""EXAMPLE PATTERNS:

Simple attribute check:
bird_flying(I) :-
    entity(I, E, bird),
    attribute(I, E, flying).
answer :- bird_flying({img_a}).
query(answer).

Relationship check:
power_lines_above_train(I) :-
    entity(I, E1, 'power lines'),
    entity(I, E2, train),
    relation(I, E1, E2, above).
answer :- power_lines_above_train({img_a}).
query(answer).

Multiple conditions:
human_standing_front_of_car(I) :-
    entity(I, E1, human),
    attribute(I, E1, standing),
    entity(I, E2, car),
    relation(I, E1, E2, 'in front of').
answer :- human_standing_front_of_car({img_a}).
query(answer)."""

        # When ICE is provided, replace hardcoded examples with ICE examples
        if ices:
            ice_parts = ["REFERENCE EXAMPLES (real questions with correct ProbLog programs — use these as style/pattern guides, but write rules using YOUR facts above, not theirs):"]
            for i, ice in enumerate(ices):
                ice_parts.append(f"\n--- Example {i+1} ---")
                ice_parts.append(f"Question: {ice['question']}")
                ice_parts.append(f"Program:\n{ice['program']}")
            examples = "\n".join(ice_parts)

        prompt = f"""Generate ProbLog rules and query to answer this question.

{predicate_section}

{count_section}AVAILABLE FACTS:
{facts_str}

IMPORTANT RULES:
- CLOSED WORLD: You may ONLY use predicates that appear in the AVAILABLE FACTS above. If a predicate (like count_exactly, count_more, etc.) does NOT appear in the facts, you MUST NOT use it. Using undefined predicates causes a fatal error.
- ARITY MUST MATCH: entity always has 3 args (image_id, entity_id, category). attribute always has 3 args (image_id, entity_id, value). relation always has 4 args (image_id, subject_id, object_id, relation_type). The first argument is ALWAYS the image_id. Never omit it.
- SELECTIVE USAGE: You do NOT need to use all facts. Only use facts relevant to answering the question.
- INCOMPLETE EVIDENCE: If the evidence is incomplete, write rules using only the facts that ARE available. Do not reference facts that do not exist.
- PURE LOGIC ONLY: Do NOT embed numeric literals, probability values, or arithmetic comparisons (like 0.8, <=, >=) in rule bodies. ProbLog rules must contain only predicate calls and logical connectives (, ; \\+). Probabilities are handled by the facts, not the rules.
{answer_pattern}
{connectives}

QUESTION: {question}

Generate ONLY:
1. Rule(s) using :- syntax
2. An answer rule combining the conditions
3. query(answer).

{examples}

Output rules and query only, no explanation:"""

        return prompt

    def _generate_query(
        self,
        question: str,
        facts: List[ProbLogFact],
        llm,
        ices: list = None
    ) -> Tuple[str, str]:
        """LLM generates ProbLog rules and query for the question."""

        facts_str = ProbLogFactBuilder.facts_to_string(facts)

        prompt = self._build_prompt(question, facts_str, ices=ices)

        messages = [
            {"role": "system", "content": "Generate valid ProbLog syntax only. No markdown, no explanations."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = llm.chat(messages, temperature=0.0)
            rules, query = self._parse_response(response)

            if not query.startswith("query("):
                return self._fallback_query(question, facts)

            return rules, query

        except Exception as e:
            print(f"  Warning: Query generation failed: {e}")
            return self._fallback_query(question, facts)

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

    def _fallback_query(self, question: str, facts: list = None) -> Tuple[str, str]:
        """Generate fallback query when LLM fails."""
        # Try to extract image from question
        match = re.search(r'image[_ ]([a-z])', question.lower())
        if match:
            image_id = f"image_{match.group(1)}"
        elif facts:
            # Use first image_id found in facts
            for f in facts:
                for arg in f.arguments:
                    if isinstance(arg, str) and arg.startswith('image_'):
                        image_id = arg
                        break
                else:
                    continue
                break
            else:
                image_id = "image_a"
        else:
            image_id = "image_a"

        rules = "fallback(I) :- entity(I, _, _)."
        query = f"query(fallback({image_id}))."
        return rules, query

    def _execute_program(
        self,
        facts: List[ProbLogFact],
        rules: str,
        query: str,
        threshold: float = 0.5
    ) -> Tuple[float, str]:
        """Execute ProbLog program in a subprocess with timeout.

        Returns:
            (probability, error_message) — error_message is None on success
        """
        program = self._build_program_string(facts, rules, query)
        program = sanitize_program(program)

        queue = mp.Queue()
        proc = mp.Process(target=_problog_worker, args=(program, queue))
        proc.start()
        proc.join(timeout=_PROBLOG_TIMEOUT)

        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)
            return threshold, f"ProbLog timed out after {_PROBLOG_TIMEOUT}s"

        try:
            result = queue.get_nowait()
        except Exception:
            return threshold, "ProbLog subprocess returned no result"

        if isinstance(result, tuple) and result[0] == 'error':
            return threshold, result[1]

        return result, None

    def _build_program_string(
        self,
        facts: List[ProbLogFact],
        rules: str,
        query: str
    ) -> str:
        """Build complete ProbLog program."""
        facts_str = ProbLogFactBuilder.facts_to_string(facts)
        return f"{facts_str}\n\n{rules}\n\n{query}"
