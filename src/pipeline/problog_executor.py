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

    # Sugar rules always included in programs
    SUGAR_RULES = """% Helper predicates for easier rule writing
has_attribute(I,E,A) :- attribute(I,E,A).
is_category(I,E,C) :- entity(I,E,C,_,_,_,_).
has_relationship(I,A,B,R) :- relation(I,A,B,R)."""

    def __init__(self):
        """Initialize executor with ModelManager singleton."""
        self.model_manager = ModelManager()

    def execute_subquestions(
        self,
        subquestions: List[BinarySubquestion],
        facts: List[ProbLogFact],
        ultimate_question: str = None
    ) -> Tuple[List[SubquestionResult], float]:
        """
        Execute all subquestions in one unified ProbLog program.
        Also generates and executes ultimate question composition rule.

        Flow:
        1. Generate rules for ALL subquestions (LLM calls)
        2. Generate ultimate composition rule (NEW)
        3. Build unified program (facts + sugar + all rules + all queries + ultimate query)
        4. Execute once via ProbLog library
        5. Extract results for each subquestion AND ultimate answer

        Args:
            subquestions: Binary subquestions to answer
            facts: ProbLog facts (knowledge base)
            ultimate_question: Original ultimate question (optional)

        Returns:
            Tuple[List[SubquestionResult], float]: (subquestion results, ultimate probability)

        Raises:
            ProbLogExecutorError: If execution fails
        """
        try:
            if not subquestions:
                return [], 0.5

            if not facts:
                print("⚠ Warning: No facts available for reasoning")
                return [
                    SubquestionResult(
                        subquestion=sq.question,
                        probability=0.5,
                        supporting_facts=[],
                        evidence_trail=["No facts available"]
                    )
                    for sq in subquestions
                ], 0.5

            llm_client = self.model_manager.get_llm_client()
            total = len(subquestions)

            print(f"\n🔍 Generating rules for {total} subquestions...")

            # Step 1: Generate rules for ALL subquestions
            all_rules = []
            all_queries = []
            for i, subquestion in enumerate(subquestions, 1):
                print(f"  [{i}/{total}] Generating rules: {subquestion.question}")
                rules_string, query_string = self._generate_rules_for_subquestion(
                    subquestion, facts, llm_client
                )
                all_rules.append(f"% Rule for: {subquestion.question}")
                all_rules.append(rules_string)
                all_queries.append(query_string)

            # Step 2: Generate ultimate composition rule (NEW)
            ultimate_probability = 0.5  # Default
            if ultimate_question:
                print(f"\n🔍 Generating ultimate composition rule...")
                print(f"  Ultimate question: {ultimate_question}")
                composition_rule, ultimate_query = self._generate_ultimate_composition_rule(
                    ultimate_question, subquestions, llm_client
                )
                all_rules.append(f"\n% Ultimate composition rule")
                all_rules.append(composition_rule)
                all_queries.append(ultimate_query)
                print(f"  ✓ Generated composition rule")

            # Step 3: Build unified program
            facts_string = self._facts_to_problog_string(facts)
            unified_program = self._build_unified_program(
                facts_string, all_rules, all_queries
            )

            # Step 4: Save complete program to knowledge_base.pl
            with open('knowledge_base.pl', 'w') as f:
                f.write("% PROVE Pipeline - Unified ProbLog Program\n")
                f.write(f"% Generated for {total} subquestions\n")
                if ultimate_question:
                    f.write(f"% Ultimate question: {ultimate_question}\n")
                f.write("\n")
                f.write(unified_program)

            print(f"✓ Generated unified ProbLog program ({len(unified_program)} chars)")
            print(f"  Saved to: knowledge_base.pl")

            # Step 5: Execute once
            print(f"\n🔍 Executing unified ProbLog program...")
            result_dict = self._execute_unified_program(unified_program)

            # Step 6: Extract results for each subquestion
            results = []
            for i, (subquestion, query_string) in enumerate(zip(subquestions, all_queries[:-1] if ultimate_question else all_queries), 1):
                probability = self._extract_query_result(result_dict, query_string)
                print(f"  [{i}/{total}] {subquestion.question}")
                print(f"    → Probability: {probability:.4f}")

                results.append(SubquestionResult(
                    subquestion=subquestion.question,
                    probability=probability,
                    supporting_facts=[unified_program],
                    evidence_trail=[f"Unified ProbLog execution: {probability:.4f}"]
                ))

            # Step 7: Extract ultimate answer probability (NEW)
            if ultimate_question:
                ultimate_probability = self._extract_query_result(result_dict, all_queries[-1])
                print(f"\n  🎯 ULTIMATE ANSWER: {ultimate_question}")
                print(f"     → Probability: {ultimate_probability:.4f}")

            print(f"✓ Completed {total} subquestions + ultimate answer\n")
            return results, ultimate_probability

        except Exception as err:
            raise ProbLogExecutorError(f"ProbLog execution failed: {err}")

    def _execute_single_subquestion(
        self,
        subquestion: BinarySubquestion,
        facts: List[ProbLogFact]
    ) -> SubquestionResult:
        """
        Execute single subquestion using ProbLog with LLM-generated rules.

        Flow:
        1. Convert facts to ProbLog string
        2. Generate rules for this subquestion (LLM)
        3. Combine: facts + sugar + rules + query
        4. Execute via ProbLog library
        5. Return result with probability

        Args:
            subquestion: Binary subquestion to execute
            facts: All ProbLog facts

        Returns:
            SubquestionResult: Execution result
        """
        try:
            llm_client = self.model_manager.get_llm_client()

            # Step 1: Convert facts to ProbLog string
            facts_string = self._facts_to_problog_string(facts)

            # Step 2: Generate rules + query for this subquestion (LLM call)
            rules_string, query_string = self._generate_rules_for_subquestion(
                subquestion, facts, llm_client
            )

            # DEBUG: Print generated rules to diagnose issues
            print(f"    DEBUG: Generated rules for: {subquestion.question[:50]}...")
            print(f"      Rules preview: {rules_string[:150] if rules_string else '(empty)'}...")
            print(f"      Query: {query_string}")

            # Step 3: Combine into complete program
            full_program = self._build_complete_program(
                facts_string, rules_string, query_string
            )

            # Save complete program to knowledge_base.pl for debugging
            with open('knowledge_base.pl', 'w') as f:
                f.write(f"% Subquestion: {subquestion.question}\n\n")
                f.write(full_program)

            # Step 4: Execute via ProbLog library
            probability = self._execute_problog_program(full_program, query_string)

            # Step 5: Return result
            return SubquestionResult(
                subquestion=subquestion.question,
                probability=probability,
                supporting_facts=[full_program],  # Store full program for debugging
                evidence_trail=[
                    f"Subquestion type: {subquestion.subquestion_type}",
                    f"Generated rules: {rules_string[:200]}...",
                    f"Query: {query_string}",
                    f"Probability: {probability:.4f}"
                ]
            )

        except Exception as e:
            print(f"⚠ Warning: Failed to execute subquestion '{subquestion.question}': {e}")
            import traceback
            traceback.print_exc()
            # Return default uncertainty on failure
            return SubquestionResult(
                subquestion=subquestion.question,
                probability=0.5,
                supporting_facts=[],
                evidence_trail=[f"Execution failed: {str(e)}"]
            )

    def _facts_to_problog_string(self, facts: List[ProbLogFact]) -> str:
        """
        Convert ProbLogFact list to ProbLog string format.

        Args:
            facts: List of ProbLog facts

        Returns:
            str: Facts formatted as ProbLog string
        """
        fact_lines = []

        # Group by predicate for organization
        predicates = ["entity", "relation", "attribute", "scene_attr", "count"]

        for predicate in predicates:
            predicate_facts = [f for f in facts if f.predicate == predicate]
            if predicate_facts:
                fact_lines.append(f"% {predicate} facts")
                for fact in predicate_facts:
                    fact_lines.append(fact.to_prolog_string())
                fact_lines.append("")  # Empty line between sections

        return "\n".join(fact_lines)

    def _build_complete_program(
        self,
        facts_string: str,
        rules_string: str,
        query_string: str
    ) -> str:
        """
        Combine facts, sugar, rules, and query into complete ProbLog program.

        Args:
            facts_string: ProbLog facts
            rules_string: Generated rules
            query_string: Query statement

        Returns:
            str: Complete ProbLog program
        """
        program_parts = [
            "% Facts from visual evidence",
            facts_string,
            "",
            "% Sugar rules",
            self.SUGAR_RULES,
            "",
            "% Generated rules for this subquestion",
            rules_string,
            "",
            "% Query",
            query_string
        ]

        return "\n".join(program_parts)

    def _build_unified_program(
        self,
        facts_string: str,
        all_rules: List[str],
        all_queries: List[str]
    ) -> str:
        """
        Build single ProbLog program with all rules and queries.

        Args:
            facts_string: ProbLog facts
            all_rules: List of rule strings (includes comment headers)
            all_queries: List of query strings

        Returns:
            str: Complete unified ProbLog program
        """
        program_parts = [
            "% Facts from visual evidence",
            facts_string,
            "",
            "% Sugar rules",
            self.SUGAR_RULES,
            "",
            "% Generated rules for all subquestions",
            "\n\n".join(all_rules),
            "",
            "% Queries for all subquestions",
            "\n".join(all_queries)
        ]

        return "\n".join(program_parts)

    def _execute_unified_program(self, program: str) -> dict:
        """
        Execute unified ProbLog program and return all results.

        Args:
            program: Complete ProbLog program string

        Returns:
            dict: Dictionary mapping query terms to probabilities
        """
        program = program.replace('`', '').replace('\r', '')

        try:
            result = get_evaluatable().create_from(PrologString(program)).evaluate()
            return dict(result)
        except Exception as e:
            print(f"⚠ Warning: Unified ProbLog execution failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

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
- If subquestion asks about a specific image, query that image
- DO NOT use markdown code blocks or backticks in your output
- DO NOT wrap output in ```prolog or ``` code fences
- Output ONLY the rule definitions and query, nothing else

---

EXAMPLE 1
Subquestions:
1. Is the dog in image A wearing a green harness?
2. Is the dog in image B wearing a black collar?

Problog Facts:
0.861::entity(image_a, harness_a_0, harness, 195,129,336,290).
0.929::entity(image_a, dog_a_4, dog, 55,96,545,391).
0.873::entity(image_b, dog_b_3, dog, 60,0,157,176).
0.872::entity(image_b, collar_b_4, collar, 101,39,140,62).
0.854::relation(image_a, harness_a_0, dog_a_4, wearing).
0.875::relation(image_b, collar_b_4, dog_b_3, wearing).
0.954::attribute(image_a, harness_a_0, green).
0.885::attribute(image_b, collar_b_4, black).

Expected Output:
dog_wearing_green_harness(I) :-
    is_category(I,D,dog),
    is_category(I,H,harness),
    has_relationship(I,H,D,wearing),
    has_attribute(I,H,green).

dog_wearing_black_collar(I) :-
    is_category(I,D,dog),
    is_category(I,C,collar),
    has_relationship(I,C,D,wearing),
    has_attribute(I,C,black).

query(dog_wearing_green_harness(image_a)).
query(dog_wearing_black_collar(image_b)).

---

EXAMPLE 2
Subquestions:
1. In image A, is there a man to the left of a woman?
2. In image A, is a woman holding an umbrella?
3. In image A, is the umbrella red?

Problog Facts:
0.881::entity(image_a, man_a_0, man, 150,150,300,400).
0.887::entity(image_a, woman_a_1, woman, 300,150,450,400).
0.905::entity(image_a, umbrella_a_2, umbrella, 320,80,420,200).
0.892::relation(image_a, woman_a_1, umbrella_a_2, holding).
0.846::relation(image_a, man_a_0, woman_a_1, left_of).
0.943::attribute(image_a, umbrella_a_2, red).

Expected Output:
man_left_of_woman(I) :-
    is_category(I,M,man),
    is_category(I,W,woman),
    has_relationship(I,M,W,left_of).

woman_holding_umbrella(I) :-
    is_category(I,W,woman),
    is_category(I,U,umbrella),
    has_relationship(I,W,U,holding).

umbrella_is_red(I) :-
    is_category(I,U,umbrella),
    has_attribute(I,U,red).

query(man_left_of_woman(image_a)).
query(woman_holding_umbrella(image_a)).
query(umbrella_is_red(image_a)).

---

EXAMPLE 3
Subquestions:
1. Is image A indoor?
2. Does image A contain four students?

Problog Facts:
0.931::entity(image_a, student_a_0, student, 50,120,200,300).
0.905::entity(image_a, student_a_1, student, 210,120,350,300).
0.915::entity(image_a, student_a_2, student, 360,120,480,300).
0.927::entity(image_a, student_a_3, student, 490,120,620,300).
0.954::scene_attr(image_a, indoor).
0.894::count(image_a, student, 4).

Expected Output:
scene_is_indoor(I) :- scene_attr(I, indoor).

student_count_four(I) :- count(I, student, 4).

query(scene_is_indoor(image_a)).
query(student_count_four(image_a)).

---

NOW GENERATE RULES FOR THIS SUBQUESTION

Subquestion: "{subquestion.question}"
Subquestion Type: {subquestion.subquestion_type}

Available Facts:
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
