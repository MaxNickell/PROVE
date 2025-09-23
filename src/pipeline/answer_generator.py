"""
Answer generation component for PROVE pipeline.
Synthesizes final answers using subquery results and evidence trails.
"""

from typing import List, Dict, Any
import json

from src.core.model_manager import ModelManager
from src.core.types import SubqueryResult, AnswerResult


class AnswerGeneratorError(RuntimeError):
    """Custom exception for answer generation failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return self.message


class AnswerGenerator:
    """
    Generate final answers using subquery results and evidence trails.
    Synthesizes comprehensive responses with confidence and explanations.
    """
    
    def __init__(self):
        """Initialize generator with ModelManager singleton."""
        self.model_manager = ModelManager()
    
    def generate_final_answer(
        self,
        ultimate_question: str,
        subquery_results: List[SubqueryResult],
        image_contexts: Dict[str, str] = None
    ) -> AnswerResult:
        """
        Generate final answer using subquery results and evidence.
        
        Args:
            ultimate_question: Original comparative question
            subquery_results: Results from ProbLog execution
            image_contexts: Optional image context for reference
            
        Returns:
            AnswerResult: Final answer with explanation
            
        Raises:
            AnswerGeneratorError: If generation fails
        """
        try:
            if not subquery_results:
                return AnswerResult(
                    text="Unable to answer due to insufficient evidence.",
                    explanation="No subquery results available for analysis."
                )
            
            # Get LLM client
            llm_client = self.model_manager.get_llm_client()
            
            # Organize subquery results
            organized_results = self._organize_subquery_results(subquery_results)
            
            # Generate final answer using LLM synthesis
            answer_result = self._synthesize_answer(
                llm_client, ultimate_question, organized_results, image_contexts
            )
            
            return answer_result
            
        except Exception as err:
            raise AnswerGeneratorError(f"Answer generation failed: {err}")
    
    def _organize_subquery_results(
        self,
        subquery_results: List[SubqueryResult]
    ) -> Dict[str, Any]:
        """
        Organize subquery results for LLM synthesis.
        
        Args:
            subquery_results: Results from ProbLog execution
            
        Returns:
            Dict with organized results
        """
        # Categorize results by confidence
        high_confidence = []  # > 0.8 or < 0.2
        medium_confidence = []  # 0.2 - 0.8
        low_confidence = []  # 0.4 - 0.6 (uncertain)
        
        for result in subquery_results:
            if result.probability > 0.8 or result.probability < 0.2:
                high_confidence.append(result)
            elif result.probability >= 0.6 or result.probability <= 0.4:
                medium_confidence.append(result)
            else:
                low_confidence.append(result)
        
        # Calculate overall confidence
        all_probabilities = [r.probability for r in subquery_results]
        avg_probability = sum(all_probabilities) / len(all_probabilities)
        
        # Identify key findings
        positive_findings = [r for r in subquery_results if r.probability > 0.6]
        negative_findings = [r for r in subquery_results if r.probability < 0.4]
        
        return {
            "high_confidence": high_confidence,
            "medium_confidence": medium_confidence,
            "low_confidence": low_confidence,
            "avg_probability": avg_probability,
            "positive_findings": positive_findings,
            "negative_findings": negative_findings,
            "total_subqueries": len(subquery_results)
        }
    
    def _synthesize_answer(
        self,
        llm_client,
        ultimate_question: str,
        organized_results: Dict[str, Any],
        image_contexts: Dict[str, str] = None
    ) -> AnswerResult:
        """
        Synthesize final answer using LLM reasoning over organized results.
        
        Args:
            llm_client: LLM client
            ultimate_question: Original question
            organized_results: Organized subquery results
            image_contexts: Optional image contexts
            
        Returns:
            AnswerResult: Final synthesized answer
        """
        # Build context for LLM
        results_context = self._build_results_context(organized_results)
        image_context = self._build_image_context(image_contexts) if image_contexts else ""
        
        prompt = f"""Based on the probabilistic reasoning results from multiple subqueries, provide a comprehensive answer to the ultimate question.

Ultimate Question: "{ultimate_question}"

Subquery Analysis Results:
{results_context}

{image_context}

Synthesize a final answer that:

1. **Directly answers the ultimate question** with appropriate confidence
2. **Explains the reasoning** using the subquery evidence 
3. **Cites specific evidence** from the highest-confidence subqueries
4. **Acknowledges uncertainty** where appropriate
5. **Provides a confidence assessment** for the overall answer

Return JSON with this exact format:
{{
  "answer": "Direct answer to the ultimate question with appropriate confidence",
  "confidence": "high|medium|low",
  "explanation": "Detailed explanation citing specific evidence and reasoning steps",
  "key_evidence": [
    "Most important piece of evidence 1",
    "Most important piece of evidence 2",
    "Most important piece of evidence 3"
  ],
  "reasoning_chain": "Brief summary of how subqueries led to this conclusion"
}}

Guidelines:
- Be specific and cite actual evidence probabilities
- If evidence is contradictory, acknowledge and explain
- If confidence is low overall, clearly state limitations
- Focus on the most reliable evidence (high-confidence subqueries)
- Make the answer actionable and well-supported

Example patterns:
- High confidence: "Person A appears significantly more powerful (high confidence)..."
- Medium confidence: "Person A likely has an advantage (medium confidence)..."  
- Low confidence: "The evidence is mixed and inconclusive (low confidence)..."
"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert at synthesizing final answers from probabilistic evidence. Provide comprehensive, well-reasoned answers that appropriately reflect the confidence levels of the underlying evidence. Be precise about uncertainty and cite specific evidence. Return strict JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            response = llm_client.chat(
                messages,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            response_data = json.loads(response)
            
            # Extract and validate response
            answer_text = response_data.get("answer", "Unable to determine answer")
            confidence = response_data.get("confidence", "low")
            explanation = response_data.get("explanation", "No explanation available")
            key_evidence = response_data.get("key_evidence", [])
            reasoning_chain = response_data.get("reasoning_chain", "")
            
            # Build comprehensive explanation
            full_explanation = self._build_full_explanation(
                explanation, key_evidence, reasoning_chain, organized_results
            )
            
            return AnswerResult(
                text=answer_text,
                explanation=full_explanation
            )
            
        except Exception as e:
            print(f"Warning: Failed to synthesize answer: {e}")
            # Return fallback answer
            return self._generate_fallback_answer(ultimate_question, organized_results)
    
    def _build_results_context(
        self,
        organized_results: Dict[str, Any]
    ) -> str:
        """
        Build formatted context string from organized results.
        
        Args:
            organized_results: Organized subquery results
            
        Returns:
            str: Formatted results context
        """
        context_parts = []
        
        # Overall statistics
        context_parts.append(f"Total Subqueries: {organized_results['total_subqueries']}")
        context_parts.append(f"Average Probability: {organized_results['avg_probability']:.2f}")
        context_parts.append("")
        
        # High confidence results
        high_conf = organized_results['high_confidence']
        if high_conf:
            context_parts.append("HIGH CONFIDENCE FINDINGS:")
            for result in high_conf[:5]:  # Limit to top 5
                context_parts.append(f"- \"{result.subquery}\" → {result.probability:.2f}")
            context_parts.append("")
        
        # Medium confidence results  
        med_conf = organized_results['medium_confidence']
        if med_conf:
            context_parts.append("MEDIUM CONFIDENCE FINDINGS:")
            for result in med_conf[:3]:  # Limit to top 3
                context_parts.append(f"- \"{result.subquery}\" → {result.probability:.2f}")
            context_parts.append("")
        
        # Positive vs negative findings
        pos_findings = organized_results['positive_findings']
        neg_findings = organized_results['negative_findings']
        
        if pos_findings:
            context_parts.append(f"POSITIVE EVIDENCE ({len(pos_findings)} findings):")
            for result in pos_findings[:3]:
                context_parts.append(f"- {result.subquery} ({result.probability:.2f})")
            context_parts.append("")
        
        if neg_findings:
            context_parts.append(f"NEGATIVE EVIDENCE ({len(neg_findings)} findings):")
            for result in neg_findings[:3]:
                context_parts.append(f"- {result.subquery} ({result.probability:.2f})")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _build_image_context(
        self,
        image_contexts: Dict[str, str]
    ) -> str:
        """
        Build formatted image context string.
        
        Args:
            image_contexts: Image contexts from Florence-2
            
        Returns:
            str: Formatted image context
        """
        if not image_contexts:
            return ""
        
        context_parts = ["Visual Context:"]
        
        for image_id, context in image_contexts.items():
            context_parts.append(f"- {image_id.upper()}: {context}")
        
        context_parts.append("")
        return "\n".join(context_parts)
    
    def _build_full_explanation(
        self,
        explanation: str,
        key_evidence: List[str],
        reasoning_chain: str,
        organized_results: Dict[str, Any]
    ) -> str:
        """
        Build comprehensive explanation with all components.
        
        Args:
            explanation: Main explanation from LLM
            key_evidence: Key evidence points
            reasoning_chain: Reasoning chain summary
            organized_results: Organized results for additional context
            
        Returns:
            str: Comprehensive explanation
        """
        parts = []
        
        parts.append("## Answer Explanation")
        parts.append(explanation)
        parts.append("")
        
        if key_evidence:
            parts.append("## Key Evidence")
            for evidence in key_evidence:
                parts.append(f"• {evidence}")
            parts.append("")
        
        if reasoning_chain:
            parts.append("## Reasoning Chain")
            parts.append(reasoning_chain)
            parts.append("")
        
        # Add statistical summary
        parts.append("## Statistical Summary")
        parts.append(f"- Total subqueries analyzed: {organized_results['total_subqueries']}")
        parts.append(f"- Average confidence: {organized_results['avg_probability']:.2f}")
        parts.append(f"- High confidence findings: {len(organized_results['high_confidence'])}")
        parts.append(f"- Positive evidence: {len(organized_results['positive_findings'])}")
        parts.append(f"- Negative evidence: {len(organized_results['negative_findings'])}")
        
        return "\n".join(parts)
    
    def _generate_fallback_answer(
        self,
        ultimate_question: str,
        organized_results: Dict[str, Any]
    ) -> AnswerResult:
        """
        Generate fallback answer when synthesis fails.
        
        Args:
            ultimate_question: Original question
            organized_results: Organized results
            
        Returns:
            AnswerResult: Fallback answer
        """
        # Simple statistical analysis
        avg_prob = organized_results['avg_probability']
        high_conf_count = len(organized_results['high_confidence'])
        
        if avg_prob > 0.7:
            answer = f"Based on the evidence, the answer appears to be positive (confidence: {avg_prob:.2f})"
        elif avg_prob < 0.3:
            answer = f"Based on the evidence, the answer appears to be negative (confidence: {1.0 - avg_prob:.2f})"
        else:
            answer = f"The evidence is mixed and inconclusive (uncertainty: {abs(0.5 - avg_prob):.2f})"
        
        explanation = f"""This answer is based on statistical analysis of {organized_results['total_subqueries']} subqueries.
        
Average probability: {avg_prob:.2f}
High confidence findings: {high_conf_count}
Positive evidence: {len(organized_results['positive_findings'])}
Negative evidence: {len(organized_results['negative_findings'])}

Note: This is a fallback analysis due to synthesis limitations."""
        
        return AnswerResult(
            text=answer,
            explanation=explanation
        )


# Example usage and testing
if __name__ == "__main__":
    # Test answer generator
    generator = AnswerGenerator()
    
    # Sample data
    from src.core.types import SubqueryResult
    
    subquery_results = [
        SubqueryResult(
            subquery="Is person_a_0 more muscular than person_b_0?",
            probability=0.85,
            supporting_facts=["0.89::attribute(person_a_0, muscle_mass, high)"],
            evidence_trail=["High muscle mass evidence for person A"]
        )
    ]
    
    try:
        # Note: This test requires actual LLM client to be available
        print("✓ AnswerGenerator component created")
        print("✓ Ready for integration testing with LLM client")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")