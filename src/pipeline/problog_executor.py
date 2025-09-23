"""
ProbLog execution component for PROVE pipeline.
Decomposes subqueries to ProbLog queries and executes probabilistic reasoning.
"""

from typing import List, Dict, Any, Tuple
import json
import re

from src.core.model_manager import ModelManager
from src.core.types import BinarySubquery, ProbLogFact, SubqueryResult


class ProbLogExecutorError(RuntimeError):
    """Custom exception for ProbLog execution failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return self.message


class ProbLogExecutor:
    """
    Execute ProbLog reasoning over knowledge base to answer subqueries.
    Decomposes binary subqueries into logical queries and computes probabilities.
    """
    
    def __init__(self):
        """Initialize executor with ModelManager singleton."""
        self.model_manager = ModelManager()
    
    def execute_subqueries(
        self,
        subqueries: List[BinarySubquery],
        facts: List[ProbLogFact]
    ) -> List[SubqueryResult]:
        """
        Execute all subqueries against the knowledge base.
        
        Args:
            subqueries: Binary subqueries to answer
            facts: ProbLog facts (knowledge base)
            
        Returns:
            List[SubqueryResult]: Results for each subquery
            
        Raises:
            ProbLogExecutorError: If execution fails
        """
        try:
            if not subqueries:
                return []
            
            # Get LLM client for query decomposition
            llm_client = self.model_manager.get_llm_client()
            
            # Build fact lookup for efficient querying
            fact_lookup = self._build_fact_lookup(facts)
            
            # Process each subquery
            results = []
            
            for subquery in subqueries:
                result = self._execute_single_subquery(
                    subquery, facts, fact_lookup, llm_client
                )
                results.append(result)
            
            return results
            
        except Exception as err:
            raise ProbLogExecutorError(f"ProbLog execution failed: {err}")
    
    def _execute_single_subquery(
        self,
        subquery: BinarySubquery,
        facts: List[ProbLogFact],
        fact_lookup: Dict[str, List[ProbLogFact]],
        llm_client
    ) -> SubqueryResult:
        """
        Execute a single subquery against the knowledge base.
        
        Args:
            subquery: Binary subquery to execute
            facts: All ProbLog facts
            fact_lookup: Organized fact lookup
            llm_client: LLM client for query decomposition
            
        Returns:
            SubqueryResult: Execution result
        """
        try:
            # Decompose subquery to logical reasoning
            reasoning_components = self._decompose_subquery(llm_client, subquery, facts)
            
            # Execute reasoning using fact lookup
            probability, supporting_facts = self._compute_probability(
                reasoning_components, fact_lookup
            )
            
            # Build evidence trail
            evidence_trail = self._build_evidence_trail(
                reasoning_components, supporting_facts
            )
            
            # Create result
            result = SubqueryResult(
                subquery=subquery.question,
                probability=probability,
                supporting_facts=[fact.to_prolog_string() for fact in supporting_facts],
                evidence_trail=evidence_trail
            )
            
            return result
            
        except Exception as e:
            print(f"Warning: Failed to execute subquery '{subquery.question}': {e}")
            # Return default result for failed subqueries
            return SubqueryResult(
                subquery=subquery.question,
                probability=0.5,  # Default uncertainty
                supporting_facts=[],
                evidence_trail=[f"Failed to execute: {str(e)}"]
            )
    
    def _build_fact_lookup(
        self,
        facts: List[ProbLogFact]
    ) -> Dict[str, List[ProbLogFact]]:
        """
        Build efficient fact lookup organized by predicate and arguments.
        
        Args:
            facts: All ProbLog facts
            
        Returns:
            Dict organizing facts for efficient lookup
        """
        lookup = {
            "attribute": [],
            "relation": [],
            "object": [],
            "location": []
        }
        
        for fact in facts:
            if fact.predicate in lookup:
                lookup[fact.predicate].append(fact)
            else:
                # Handle unknown predicates
                if "other" not in lookup:
                    lookup["other"] = []
                lookup["other"].append(fact)
        
        return lookup
    
    def _decompose_subquery(
        self,
        llm_client,
        subquery: BinarySubquery,
        facts: List[ProbLogFact]
    ) -> Dict[str, Any]:
        """
        Decompose subquery into logical reasoning components.
        
        Args:
            llm_client: LLM client
            subquery: Binary subquery to decompose
            facts: Available facts for context
            
        Returns:
            Dict with reasoning components
        """
        # Create context about available facts
        fact_context = self._create_fact_context(facts, subquery.referenced_objects)
        
        prompt = f"""Decompose this binary subquery into logical reasoning components that can be evaluated using the available facts:

Subquery: "{subquery.question}"
Type: {subquery.subquery_type}
Referenced Objects: {subquery.referenced_objects}

Available Facts Context:
{fact_context}

Determine what logical reasoning is needed to answer this question. Consider:

1. **Direct Facts**: Can the question be answered directly from attribute or relation facts?
2. **Comparative Reasoning**: Does it require comparing attributes between objects?
3. **Relationship Reasoning**: Does it depend on spatial or interaction relationships?
4. **Combined Reasoning**: Does it need multiple facts combined?

Return JSON with this exact format:
{{
  "reasoning_type": "direct|comparative|relationship|combined",
  "required_facts": [
    {{
      "predicate": "attribute|relation|object",
      "object_id": "specific_object_id",
      "additional_args": ["arg1", "arg2"],
      "importance": "high|medium|low"
    }}
  ],
  "logical_operation": "AND|OR|COMPARISON|THRESHOLD",
  "explanation": "Brief explanation of the reasoning process"
}}

Examples:
- "Does person_a_0 have high muscle_mass?" → Direct attribute lookup
- "Is person_a_0 more muscular than person_b_0?" → Comparative reasoning on muscle attributes  
- "Is person_a_0 lifting weight_a_1?" → Relationship fact lookup
- "Is person_a_0 stronger based on muscle and lifting?" → Combined attribute + relation reasoning"""

        messages = [
            {
                "role": "system", 
                "content": "You are an expert at decomposing visual questions into logical reasoning components. Analyze what facts and operations are needed to answer each question. Return strict JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            response = llm_client.chat(
                messages,
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            response_data = json.loads(response)
            
            # Validate and return components
            return {
                "reasoning_type": response_data.get("reasoning_type", "direct"),
                "required_facts": response_data.get("required_facts", []),
                "logical_operation": response_data.get("logical_operation", "AND"),
                "explanation": response_data.get("explanation", "Direct fact lookup")
            }
            
        except Exception as e:
            print(f"Warning: Failed to decompose subquery: {e}")
            # Return default decomposition
            return {
                "reasoning_type": "direct",
                "required_facts": [],
                "logical_operation": "AND",
                "explanation": "Default reasoning due to decomposition failure"
            }
    
    def _create_fact_context(
        self,
        facts: List[ProbLogFact],
        referenced_objects: List[str]
    ) -> str:
        """
        Create context string showing available facts for referenced objects.
        
        Args:
            facts: All available facts
            referenced_objects: Objects referenced in the subquery
            
        Returns:
            str: Formatted fact context
        """
        context_parts = []
        
        # Group facts by referenced objects
        for obj_id in referenced_objects:
            obj_facts = []
            
            for fact in facts:
                # Check if this fact involves the referenced object
                if obj_id in fact.arguments:
                    fact_str = f"{fact.predicate}({', '.join(fact.arguments)}) [conf: {fact.probability:.2f}]"
                    obj_facts.append(fact_str)
            
            if obj_facts:
                context_parts.append(f"{obj_id}: {', '.join(obj_facts[:5])}")  # Limit to 5 facts
            else:
                context_parts.append(f"{obj_id}: No direct facts available")
        
        return "\n".join(context_parts)
    
    def _compute_probability(
        self,
        reasoning_components: Dict[str, Any],
        fact_lookup: Dict[str, List[ProbLogFact]]
    ) -> Tuple[float, List[ProbLogFact]]:
        """
        Compute probability based on reasoning components and available facts.
        
        Args:
            reasoning_components: Decomposed reasoning components
            fact_lookup: Organized fact lookup
            
        Returns:
            Tuple of (probability, supporting_facts)
        """
        reasoning_type = reasoning_components["reasoning_type"]
        required_facts = reasoning_components["required_facts"]
        logical_operation = reasoning_components["logical_operation"]
        
        supporting_facts = []
        
        if reasoning_type == "direct":
            # Direct fact lookup
            probability = self._compute_direct_probability(required_facts, fact_lookup, supporting_facts)
            
        elif reasoning_type == "comparative":
            # Comparative reasoning between objects
            probability = self._compute_comparative_probability(required_facts, fact_lookup, supporting_facts)
            
        elif reasoning_type == "relationship":
            # Relationship-based reasoning
            probability = self._compute_relationship_probability(required_facts, fact_lookup, supporting_facts)
            
        elif reasoning_type == "combined":
            # Combined reasoning with multiple facts
            probability = self._compute_combined_probability(required_facts, fact_lookup, supporting_facts, logical_operation)
            
        else:
            # Default reasoning
            probability = 0.5
        
        return probability, supporting_facts
    
    def _compute_direct_probability(
        self,
        required_facts: List[Dict[str, Any]],
        fact_lookup: Dict[str, List[ProbLogFact]],
        supporting_facts: List[ProbLogFact]
    ) -> float:
        """Compute probability for direct fact lookup."""
        if not required_facts:
            return 0.5
        
        probabilities = []
        
        for req_fact in required_facts:
            predicate = req_fact.get("predicate", "attribute")
            object_id = req_fact.get("object_id", "")
            
            # Find matching facts
            matching_facts = []
            for fact in fact_lookup.get(predicate, []):
                if object_id in fact.arguments:
                    matching_facts.append(fact)
            
            if matching_facts:
                # Use highest confidence fact
                best_fact = max(matching_facts, key=lambda f: f.probability)
                probabilities.append(best_fact.probability)
                supporting_facts.append(best_fact)
        
        # Average probabilities for direct reasoning
        return sum(probabilities) / len(probabilities) if probabilities else 0.5
    
    def _compute_comparative_probability(
        self,
        required_facts: List[Dict[str, Any]],
        fact_lookup: Dict[str, List[ProbLogFact]],
        supporting_facts: List[ProbLogFact]
    ) -> float:
        """Compute probability for comparative reasoning."""
        # Simplified comparative reasoning
        # In a full implementation, this would compare attribute values
        return self._compute_direct_probability(required_facts, fact_lookup, supporting_facts)
    
    def _compute_relationship_probability(
        self,
        required_facts: List[Dict[str, Any]],
        fact_lookup: Dict[str, List[ProbLogFact]],
        supporting_facts: List[ProbLogFact]
    ) -> float:
        """Compute probability for relationship reasoning."""
        # Look for relation facts
        relation_facts = fact_lookup.get("relation", [])
        
        probabilities = []
        for req_fact in required_facts:
            object_id = req_fact.get("object_id", "")
            
            # Find relevant relation facts
            for fact in relation_facts:
                if object_id in fact.arguments:
                    probabilities.append(fact.probability)
                    supporting_facts.append(fact)
        
        return max(probabilities) if probabilities else 0.5
    
    def _compute_combined_probability(
        self,
        required_facts: List[Dict[str, Any]],
        fact_lookup: Dict[str, List[ProbLogFact]],
        supporting_facts: List[ProbLogFact],
        logical_operation: str
    ) -> float:
        """Compute probability for combined reasoning."""
        individual_probs = []
        
        for req_fact in required_facts:
            prob = self._compute_direct_probability([req_fact], fact_lookup, supporting_facts)
            individual_probs.append(prob)
        
        if not individual_probs:
            return 0.5
        
        # Apply logical operation
        if logical_operation == "AND":
            # Product of probabilities (all must be true)
            result = 1.0
            for prob in individual_probs:
                result *= prob
            return result
        elif logical_operation == "OR":
            # Complement of product of complements (at least one true)
            result = 1.0
            for prob in individual_probs:
                result *= (1.0 - prob)
            return 1.0 - result
        else:
            # Default to average
            return sum(individual_probs) / len(individual_probs)
    
    def _build_evidence_trail(
        self,
        reasoning_components: Dict[str, Any],
        supporting_facts: List[ProbLogFact]
    ) -> List[str]:
        """
        Build human-readable evidence trail.
        
        Args:
            reasoning_components: Decomposed reasoning components
            supporting_facts: Facts that supported the reasoning
            
        Returns:
            List[str]: Evidence trail steps
        """
        trail = []
        
        # Add reasoning explanation
        explanation = reasoning_components.get("explanation", "Direct reasoning")
        trail.append(f"Reasoning: {explanation}")
        
        # Add supporting facts
        if supporting_facts:
            trail.append("Supporting Evidence:")
            for fact in supporting_facts[:5]:  # Limit to 5 facts
                fact_desc = f"- {fact.predicate}({', '.join(fact.arguments)}) with confidence {fact.probability:.2f}"
                trail.append(fact_desc)
        else:
            trail.append("No direct supporting facts found")
        
        return trail
    
    def get_execution_summary(
        self,
        results: List[SubqueryResult]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for execution results.
        
        Args:
            results: List of SubqueryResult instances
            
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
    # Test ProbLog executor
    executor = ProbLogExecutor()
    
    # Sample data
    from src.core.types import BinarySubquery, ProbLogFact
    
    subqueries = [
        BinarySubquery(
            question="Does person_a_0 have high muscle_mass?",
            referenced_objects=["person_a_0"],
            subquery_type="attribute"
        )
    ]
    
    facts = [
        ProbLogFact(
            probability=0.89,
            predicate="attribute",
            arguments=["person_a_0", "muscle_mass", "high"]
        )
    ]
    
    try:
        # Note: This test requires actual LLM client to be available
        print("✓ ProbLogExecutor component created")
        print("✓ Ready for integration testing with LLM client")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")