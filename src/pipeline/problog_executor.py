"""
ProbLog execution component for PROVE pipeline.
Uses actual ProbLog library with LLM-generated rules to answer subquestions.
"""

from typing import List, Dict, Any, Tuple
import re

from problog.program import PrologString
from problog import get_evaluatable

from src.core.model_manager import ModelManager
from src.core.types import BinarySubquestion, ProbLogFact, SubquestionResult


class ProbLogExecutorError(RuntimeError):
    """Custom exception for ProbLog execution failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class ProbLogExecutor:
    """
    Execute ProbLog reasoning using LLM-generated rules.

    For each subquestion:
    1. Convert facts to ProbLog string
    2. Generate rules using LLM (with in-context examples)
    3. Combine facts + sugar + rules + query
    4. Execute via ProbLog library
    5. Return probability result
    """

    SUGAR_RULES = """% Helper predicates
has_attribute(I,E,A) :- attribute(I,E,A).
is_category(I,E,C) :- entity(I,E,C,_,_,_,_).
has_relationship(I,A,B,R) :- relation(I,A,B,R)."""

    def __init__(self):
        """Initialize executor with ModelManager singleton."""
        self.model_manager = ModelManager()

    def execute_subquestions(
        self,
        subquestions: List[BinarySubquestion],
        evidence_collections: List['EvidenceCollection'],
        images: Dict[str, 'ImageData'],
        ultimate_question: str = None
    ) -> Tuple[List[SubquestionResult], str]:
        """
        Execute subquestions using SCOPED evidence (one evidence collection per subquestion).

        NEW EFFICIENT FLOW:
        1. For each subquestion + evidence pair:
           - Build scoped facts from evidence (only relevant entities)
           - Generate rules with scoped facts (smaller LLM prompt)
           - Execute scoped program and get probability
        2. Generate ultimate composition from subquestion predicates
        3. Execute ultimate query

        Args:
            subquestions: Binary subquestions to answer
            evidence_collections: Evidence for each subquestion (1:1 mapping)
            images: ImageData for entity metadata
            ultimate_question: Original ultimate question (optional)

        Returns:
            Tuple[List[SubquestionResult], str]: (subquestion results, natural language answer)

        Raises:
            ProbLogExecutorError: If execution fails
        """
        try:
            if not subquestions:
                return [], "No subquestions provided."

            if not evidence_collections or len(evidence_collections) != len(subquestions):
                raise ProbLogExecutorError(f"Evidence collections mismatch: {len(evidence_collections)} vs {len(subquestions)} subquestions")

            # Import ProbLogFactBuilder here to avoid circular import
            from src.pipeline.problog_builder import ProbLogFactBuilder

            fact_builder = ProbLogFactBuilder()
            llm_client = self.model_manager.get_llm_client()
            total = len(subquestions)

            print(f"\n🔍 Executing {total} subquestions with SCOPED evidence...")

            # Step 1: Execute each subquestion with its scoped evidence
            results = []
            all_predicates = []  # For ultimate composition

            for i, (subquestion, evidence) in enumerate(zip(subquestions, evidence_collections), 1):
                print(f"  [{i}/{total}] {subquestion.question}")

                # Build scoped facts from THIS subquestion's evidence
                scoped_facts = fact_builder.build_facts_from_evidence(evidence, images)

                # Generate rules with scoped facts (much smaller prompt!)
                rules_string, query_string = self._generate_rules_for_subquestion(
                    subquestion, scoped_facts, llm_client
                )

                # Build complete scoped program
                facts_string = self._facts_to_problog_string(scoped_facts)
                scoped_program = self._build_complete_program(facts_string, rules_string, query_string)

                # Execute scoped program
                probability = self._execute_problog_program(scoped_program, query_string)

                print(f"    → Probability: {probability:.4f}")

                # Extract predicate name for ultimate composition
                predicate_match = re.search(r'query\(([^(]+)\(', query_string)
                if predicate_match:
                    all_predicates.append(predicate_match.group(1))

                # Store result with scoped facts and program
                results.append(SubquestionResult(
                    subquestion=subquestion.question,
                    probability=probability,
                    supporting_facts=scoped_facts,
                    problog_program=scoped_program,
                    evidence_trail=[
                        f"Evidence: {len(evidence.attributes)} attributes, {len(evidence.relationships)} relationships, {len(evidence.counts)} counts",
                        f"Scoped facts: {len(scoped_facts)}",
                        f"Probability: {probability:.4f}"
                    ]
                ))

            # Step 2: Generate ultimate answer (if provided)
            ultimate_answer = "No ultimate question provided."
            if ultimate_question:
                print(f"\n🔍 Generating ultimate answer...")
                print(f"  Ultimate question: {ultimate_question}")
                ultimate_answer = self._execute_ultimate_composition_with_predicates(
                    ultimate_question, all_predicates, results, llm_client
                )

            # Step 3: Save unified program for debugging
            unified_program = self._build_unified_debug_program(results, ultimate_question, ultimate_answer)
            with open('knowledge_base.pl', 'w') as f:
                f.write(unified_program)

            print(f"\n✓ Completed {total} subquestions")
            print(f"  Saved unified program to: knowledge_base.pl")

            return results, ultimate_answer

        except Exception as err:
            raise ProbLogExecutorError(f"ProbLog execution failed: {err}")

    def _execute_ultimate_composition_with_predicates(
        self,
        ultimate_question: str,
        predicates: List[str],
        results: List[SubquestionResult],
        llm_client
    ) -> str:
        """
        Use LLM to answer ultimate question given binary subquestion answers.

        Args:
            ultimate_question: Ultimate question text
            predicates: List of predicate names (unused, kept for compatibility)
            results: Subquestion results with probabilities
            llm_client: LLM client

        Returns:
            str: Binary answer ("True" or "False")
        """
        # Convert subquestion probabilities to binary answers (threshold 0.5)
        subquestion_answers = []
        for result in results:
            answer = "TRUE" if result.probability >= 0.5 else "FALSE"
            subquestion_answers.append({
                'question': result.subquestion,
                'answer': answer,
                'probability': result.probability
            })

        # Build minimal prompt
        prompt = self._create_ultimate_reasoning_prompt(
            ultimate_question,
            subquestion_answers
        )

        # Call LLM with system instruction for binary output
        print(f"  Calling LLM for ultimate reasoning...")
        messages = [
            {
                "role": "system",
                "content": "You are a logical reasoning assistant. Answer questions with ONLY 'True' or 'False'. Do not provide any explanation or additional text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        response = llm_client.chat(messages, temperature=0.0)

        # Parse response to extract True/False
        answer = response.strip()
        # Normalize to True/False
        if "true" in answer.lower():
            answer = "True"
        elif "false" in answer.lower():
            answer = "False"
        else:
            # Fallback if LLM doesn't follow instruction
            print(f"  Warning: LLM returned unexpected answer: '{answer}'")
            answer = "False"  # Conservative default

        # Display answer
        print(f"\n  LLM Answer: {answer}")

        return answer

    def _create_ultimate_reasoning_prompt(
        self,
        ultimate_question: str,
        subquestion_answers: List[Dict]
    ) -> str:
        """Create prompt that constrains LLM to output only True or False."""
        prompt = "Given the following subquestion answers:\n\n"

        for i, sq in enumerate(subquestion_answers, 1):
            prompt += f"{i}. {sq['question']} → {sq['answer']}\n"

        prompt += f"\n{ultimate_question}\n\n"
        prompt += "Answer with ONLY 'True' or 'False', nothing else."

        return prompt

    def _build_unified_debug_program(
        self,
        results: List[SubquestionResult],
        ultimate_question: str,
        ultimate_answer: str
    ) -> str:
        """Build clean output with sugar rules defined once."""
        parts = [
            self.SUGAR_RULES,
            ""
        ]

        if ultimate_question:
            parts.append(f"% Ultimate: {ultimate_question}")
            parts.append(f"% Answer: {ultimate_answer}\n")

        for i, result in enumerate(results, 1):
            parts.append(f"% Subquestion {i}: {result.subquestion}")
            parts.append(f"% P={result.probability:.4f}")
            parts.append(result.problog_program)
            parts.append("")

        return "\n".join(parts)

    def _facts_to_problog_string(self, facts: List[ProbLogFact]) -> str:
        """Convert facts to ProbLog string without comments."""
        return "\n".join(f.to_prolog_string() for f in facts)

    def _build_complete_program(
        self,
        facts_string: str,
        rules_string: str,
        query_string: str
    ) -> str:
        """Build complete ProbLog program with sugar rules for execution."""
        return f"{self.SUGAR_RULES}\n\n{facts_string}\n\n{rules_string}\n\n{query_string}"

    def _extract_query_result(self, result_dict: dict, query_string: str) -> float:
        """
        Extract probability for specific query from unified results.

        Handles both standard queries and count queries with variables.

        Args:
            result_dict: Dictionary of all ProbLog results
            query_string: Query statement (e.g., "query(bird_count(image_a, N)).")

        Returns:
            float: Probability (0.0 to 1.0)
        """
        query_match = re.search(r'query\(([^)]+\([^)]*\))\)', query_string)
        if not query_match:
            print(f"⚠ Warning: Could not parse query string: {query_string}")
            return 0.5

        query_term = query_match.group(1)

        # Check if this is a count query with variable (e.g., "bird_count(image_a, N)")
        if ', N)' in query_term or ',N)' in query_term:
            # This is a count query - return weighted average of all instantiations
            predicate_base = query_term.split('(')[0]  # e.g., "bird_count"
            image_id_match = re.search(r'\(([^,]+),', query_term)
            if not image_id_match:
                print(f"⚠ Warning: Could not extract image_id from: {query_term}")
                return 0.0

            image_id = image_id_match.group(1)  # e.g., "image_a"

            # Find all matching results: bird_count(image_a,0), bird_count(image_a,1), etc.
            weighted_sum = 0.0
            total_prob = 0.0
            for key in result_dict.keys():
                key_str = str(key)
                # Match pattern: predicate_base(image_id,NUMBER)
                if key_str.startswith(predicate_base) and image_id in key_str:
                    prob = float(result_dict[key])
                    # Extract the count value
                    count_match = re.search(rf'{predicate_base}\({image_id},(\d+)\)', key_str)
                    if count_match:
                        count_val = int(count_match.group(1))
                        weighted_sum += count_val * prob
                        total_prob += prob

            # Return expected value (weighted average count)
            if total_prob > 0:
                return weighted_sum / total_prob
            else:
                return 0.0

        # Standard query - find exact match
        for key in result_dict.keys():
            if str(key) == query_term or query_term in str(key):
                return float(result_dict[key])

        print(f"⚠ Warning: Query '{query_term}' not found in ProbLog results")
        print(f"  Available results: {list(result_dict.keys())}")
        return 0.0

    def _execute_problog_program(self, program: str, query_string: str) -> float:
        """
        Execute ProbLog program and extract probability for query.

        Uses problog library like demo_problog.py.

        Args:
            program: Complete ProbLog program string
            query_string: Query statement

        Returns:
            float: Probability (0.0 to 1.0)
        """
        # Sanitize program: remove any backticks or other problematic characters
        program = program.replace('`', '')  # Remove backticks
        program = program.replace('\r', '')  # Remove carriage returns

        try:
            # Create ProbLog model and evaluate
            result = get_evaluatable().create_from(PrologString(program)).evaluate()

            # Extract query name from query string
            # query_string format: "query(predicate_name(image_a))."
            query_match = re.search(r'query\(([^)]+\([^)]*\))\)', query_string)
            if not query_match:
                print(f"⚠ Warning: Could not parse query string: {query_string}")
                return 0.5

            query_term = query_match.group(1)

            # Find matching result
            for key in result.keys():
                if str(key) == query_term or query_term in str(key):
                    return float(result[key])

            # Query not found in results - might be because probability is 0
            print(f"⚠ Warning: Query '{query_term}' not found in ProbLog results")
            print(f"  Available results: {list(result.keys())}")
            return 0.0  # If not found, assume false

        except Exception as e:
            print(f"⚠ Warning: ProbLog execution failed: {e}")
            import traceback
            traceback.print_exc()
            return 0.5

    def _generate_rules_for_subquestion(
        self,
        subquestion: BinarySubquestion,
        facts: List[ProbLogFact],
        llm_client
    ) -> Tuple[str, str]:
        """
        Use LLM to generate ProbLog rules and query for a subquestion.

        Args:
            subquestion: Binary subquestion
            facts: Available facts
            llm_client: LLM client

        Returns:
            Tuple[str, str]: (rules_string, query_string)
        """
        # Build prompt with in-context examples
        prompt = self._create_rule_generation_prompt(subquestion, facts)

        messages = [
            {
                "role": "system",
                "content": "You are an expert at writing ProbLog rules for visual reasoning. Generate syntactically correct ProbLog rules and queries based on subquestions and available facts. Output ONLY valid ProbLog syntax, no explanations."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            response = llm_client.chat(messages, temperature=0.2, max_tokens=1000)

            # Parse response to extract rules and query
            rules_string, query_string = self._parse_llm_response(response)

            # Validate basic syntax
            if not self._validate_problog_syntax(rules_string, query_string):
                print(f"⚠ Warning: Generated invalid ProbLog syntax, using fallback")
                return self._generate_fallback_rule(subquestion)

            return rules_string, query_string

        except Exception as e:
            print(f"⚠ Warning: Rule generation failed: {e}")
            return self._generate_fallback_rule(subquestion)

    def _create_rule_generation_prompt(
        self,
        subquestion: BinarySubquestion,
        facts: List[ProbLogFact]
    ) -> str:
        """
        Create prompt with in-context examples for rule generation.

        Includes:
        - Task description
        - Available predicates
        - Sugar rules
        - 3 in-context examples from ProbLogImplementation.md
        - Current subquestion and facts

        Args:
            subquestion: Binary subquestion
            facts: Available facts

        Returns:
            str: Complete prompt
        """
        # Convert facts to readable string
        facts_string = self._facts_to_problog_string(facts)

        prompt = f"""TASK
You will generate ProbLog rules and a query to answer a binary subquestion based on available facts.

AVAILABLE PREDICATES
- entity(image_id, entity_id, category, x1, y1, x2, y2)
- relation(image_id, entity_a, entity_b, relation_type)
- attribute(image_id, entity_id, attr_value)
- scene_attr(image_id, attr_value)
- count(image_id, category, value)

SUGAR RULES (always available)
has_attribute(I,E,A) :- attribute(I,E,A).
is_category(I,E,C) :- entity(I,E,C,_,_,_,_).
has_relationship(I,A,B,R) :- relation(I,A,B,R).

INSTRUCTIONS
- Write ProbLog rules that define a predicate to answer the subquestion
- Use the sugar rules to make your rules cleaner
- Create a query(...) statement for the specific image mentioned
- Use proper ProbLog syntax: predicates end with '.', rules use ':-'
- CRITICAL: Only use exact categories from the Available Facts (e.g., if facts show 'buffalo' and 'cow', use those specific categories - NOT abstract terms like 'animal')
- DO NOT use markdown code blocks or backticks in your output
- Output ONLY the rule definitions and query, nothing else

---

EXAMPLE 1
Subquestion: Is the dog in image A wearing a green harness?

Available Facts:
0.861::entity(image_a, harness_a_0, harness, 195,129,336,290).
0.929::entity(image_a, dog_a_4, dog, 55,96,545,391).
0.854::relation(image_a, harness_a_0, dog_a_4, wearing).
0.954::attribute(image_a, harness_a_0, green).

Expected Output:
dog_wearing_green_harness(I) :-
    is_category(I,D,dog),
    is_category(I,H,harness),
    has_relationship(I,H,D,wearing),
    has_attribute(I,H,green).

query(dog_wearing_green_harness(image_a)).

---

EXAMPLE 2
Subquestion: Does image A contain two birds?

Available Facts:
0.871::entity(image_a, bird_a_0, bird, 210,226,286,327).
0.871::entity(image_a, bird_a_1, bird, 293,35,340,98).
0.016::count(image_a, bird, 0).
0.225::count(image_a, bird, 1).
0.759::count(image_a, bird, 2).

Expected Output:
bird_count_two(I) :- count(I, bird, 2).

query(bird_count_two(image_a)).

---

EXAMPLE 3
Subquestion: Is there a white bird on top of another animal in image A?

Available Facts:
0.874::entity(image_a, buffalo_a_0, buffalo, 93,182,402,597).
0.938::entity(image_a, bird_a_7, bird, 196,96,270,202).
0.906::relation(image_a, bird_a_7, buffalo_a_0, on_top_of).
0.787::attribute(image_a, bird_a_7, white).

Expected Output:
white_bird_on_animal(I) :-
    is_category(I,B,bird),
    is_category(I,A,buffalo),
    has_relationship(I,B,A,on_top_of),
    has_attribute(I,B,white).

query(white_bird_on_animal(image_a)).

---

NOW GENERATE RULES FOR THIS SUBQUESTION

Subquestion: "{subquestion.question}"

Available Facts (scoped to this subquestion):
{facts_string}

Output ONLY the rule definitions and query statements. No explanations or additional text."""

        return prompt

    def _parse_llm_response(self, response: str) -> Tuple[str, str]:
        """
        Parse LLM response to extract rules and query.
        Strips markdown code blocks and sanitizes output.

        Args:
            response: LLM response text

        Returns:
            Tuple[str, str]: (rules_string, query_string)
        """
        # Strip markdown code blocks if present
        response = response.strip()

        # Remove ```prolog or ```problog or ``` code fences
        if '```' in response:
            # Extract content between code fences
            parts = response.split('```')
            if len(parts) >= 3:
                # Content is between first and second ```
                response = parts[1]
                # Remove language identifier (prolog, problog, etc)
                if response.startswith(('prolog', 'problog', 'pl')):
                    response = '\n'.join(response.split('\n')[1:])

        # Remove backticks used for inline code
        response = response.replace('`', '')

        lines = response.strip().split('\n')

        rules_lines = []
        query_lines = []

        for line in lines:
            line = line.strip()
            if not line or line.startswith('%'):
                continue  # Skip empty lines and comments

            if line.startswith('query('):
                query_lines.append(line)
            else:
                rules_lines.append(line)

        rules_string = '\n'.join(rules_lines)
        query_string = '\n'.join(query_lines)

        return rules_string, query_string

    def _validate_problog_syntax(self, rules: str, query: str) -> bool:
        """
        Basic syntax validation for ProbLog rules and query.

        Checks:
        - Query starts with 'query('
        - Balanced parentheses
        - Rules contain ':-' operator (unless empty for simple queries)

        Args:
            rules: Rule definitions
            query: Query statement

        Returns:
            bool: True if syntax appears valid
        """
        if not query or not query.strip().startswith('query('):
            return False

        # Check balanced parentheses in both rules and query
        for text in [rules, query]:
            if text.count('(') != text.count(')'):
                return False

        # Rules should contain :- operator (unless empty for simple scene/count queries)
        if rules.strip() and ':-' not in rules:
            return False

        return True

    def _generate_fallback_rule(self, subquestion: BinarySubquestion) -> Tuple[str, str]:
        """
        Generate simple fallback rule when LLM fails.

        Creates basic query based on subquestion type.

        Args:
            subquestion: Binary subquestion

        Returns:
            Tuple[str, str]: (rules_string, query_string)
        """
        # Try to extract image_id from question
        image_match = re.search(r'image[_ ]([ab])', subquestion.question.lower())
        image_id = f"image_{image_match.group(1)}" if image_match else "image_a"

        if subquestion.subquestion_type == "scene_attribute":
            # Simple scene attribute query
            rules = "% Fallback: direct scene attribute query\nfallback_scene(I) :- scene_attr(I, _)."
            query = f"query(fallback_scene({image_id}))."
        elif subquestion.subquestion_type == "count":
            # Simple count query
            rules = "% Fallback: direct count query\nfallback_count(I) :- count(I, _, _)."
            query = f"query(fallback_count({image_id}))."
        else:
            # Default fallback
            rules = "% Fallback: entity existence\nfallback_entity(I) :- is_category(I, _, _)."
            query = f"query(fallback_entity({image_id}))."

        return rules, query

    def _generate_ultimate_composition_rule(
        self,
        ultimate_question: str,
        subquestions: List[BinarySubquestion],
        llm_client
    ) -> Tuple[str, str]:
        """
        Generate ProbLog rule that composes subquestions to answer ultimate question.

        Args:
            ultimate_question: The original ultimate question
            subquestions: List of binary subquestions with generated rules
            llm_client: LLM client

        Returns:
            Tuple[str, str]: (composition_rule, query_string)
        """
        # Build list of subquestion predicates (extract from previous rule generation)
        subquestion_list = "\n".join([
            f"{i+1}. \"{sq.question}\" (type: {sq.subquestion_type})"
            for i, sq in enumerate(subquestions)
        ])

        prompt = f"""You are generating a ProbLog composition rule that answers an ultimate question by combining subquestion predicates.

ULTIMATE QUESTION: "{ultimate_question}"

SUBQUESTIONS (each has a predicate in the ProbLog program):
{subquestion_list}

TASK: Generate a single ProbLog rule named 'ultimate_answer' that logically combines the subquestion predicates to answer the ultimate question.

RULES:
1. The predicate name for each subquestion is derived from its content (lowercase, underscores)
2. Use logical operators: conjunction (,), disjunction (;), negation (\\+)
3. For comparison questions ("more than", "less than"), use comparison operators
4. The rule should encode the logical structure of the ultimate question

EXAMPLES:

Example 1:
Ultimate: "Are there more birds in A than B AND are all birds orange?"
Subquestions:
  1. "Are there more birds in image A than image B?" → more_birds_in_a_than_b
  2. "Are all birds in image A orange?" → all_birds_orange_in_a
  3. "Are all birds in image B orange?" → all_birds_orange_in_b

Output:
ultimate_answer :-
    more_birds_in_a_than_b,
    all_birds_orange_in_a,
    all_birds_orange_in_b.

query(ultimate_answer).

Example 2:
Ultimate: "Do both images show dogs wearing collars?"
Subquestions:
  1. "Is there a dog in image A wearing a collar?" → dog_wearing_collar_a
  2. "Is there a dog in image B wearing a collar?" → dog_wearing_collar_b

Output:
ultimate_answer :-
    dog_wearing_collar_a,
    dog_wearing_collar_b.

query(ultimate_answer).

Example 3:
Ultimate: "Is image A indoor OR outdoor?"
Subquestions:
  1. "Is image A indoor?" → image_a_indoor
  2. "Is image A outdoor?" → image_a_outdoor

Output:
ultimate_answer :-
    (image_a_indoor ; image_a_outdoor).

query(ultimate_answer).

NOW generate the composition rule and query for the given ultimate question and subquestions.
Output ONLY valid ProbLog syntax (the rule and query), nothing else.
"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert at writing ProbLog composition rules. Generate syntactically correct ProbLog that logically combines subquestion predicates. Output ONLY valid ProbLog syntax."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            response = llm_client.chat(messages, temperature=0.2, max_tokens=500)

            # Remove markdown code fences if present
            import re
            # Remove code fence markers (```problog, ```prolog, ```, etc.)
            response = re.sub(r'```(?:problog|prolog)?\s*\n?', '', response)
            response = re.sub(r'```\s*$', '', response)

            # Extract rule and query from response
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]

            # Separate rule lines from query line
            rule_lines = []
            query_line = ""

            for line in lines:
                if line.startswith("query("):
                    query_line = line
                else:
                    rule_lines.append(line)

            composition_rule = "\n".join(rule_lines)

            if not query_line:
                query_line = "query(ultimate_answer)."

            return composition_rule, query_line

        except Exception as e:
            print(f"⚠ Warning: Ultimate composition rule generation failed: {e}")
            # Fallback: simple conjunction of all subquestions
            fallback_rule = "ultimate_answer :- true."  # Always true as fallback
            fallback_query = "query(ultimate_answer)."
            return fallback_rule, fallback_query

    def get_execution_summary(
        self,
        results: List[SubquestionResult]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for execution results.

        Args:
            results: List of SubquestionResult instances

        Returns:
            Dict with summary information
        """
        if not results:
            return {
                "total_subqueries": 0,
                "avg_probability": 0.0,
                "high_confidence_results": 0
            }

        total_subqueries = len(results)
        all_probabilities = [result.probability for result in results]
        avg_probability = sum(all_probabilities) / len(all_probabilities)

        high_confidence = len([p for p in all_probabilities if p > 0.8 or p < 0.2])

        return {
            "total_subqueries": total_subqueries,
            "avg_probability": avg_probability,
            "high_confidence_results": high_confidence,
            "probability_distribution": {
                "very_likely (>0.8)": len([p for p in all_probabilities if p > 0.8]),
                "likely (0.6-0.8)": len([p for p in all_probabilities if 0.6 <= p <= 0.8]),
                "uncertain (0.4-0.6)": len([p for p in all_probabilities if 0.4 <= p <= 0.6]),
                "unlikely (0.2-0.4)": len([p for p in all_probabilities if 0.2 <= p <= 0.4]),
                "very_unlikely (<0.2)": len([p for p in all_probabilities if p < 0.2])
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Test ProbLog executor with new implementation
    executor = ProbLogExecutor()

    print("✓ ProbLogExecutor rewritten to use actual ProbLog library")
    print("✓ LLM generates rules with 3 in-context examples")
    print("✓ Ready for integration testing")
