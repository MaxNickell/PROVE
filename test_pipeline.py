#!/usr/bin/env python3
"""
PROVE Pipeline - Aligned Architecture
Subquestions → Agents Collect Evidence → ProbLog Reasoning → Probabilities

This script implements the clean, aligned pipeline:
- No LLM answer synthesis
- ProbLog composes subquestions to answer ultimate question
- Pure probabilistic output
"""

import sys
import os
import warnings

# Suppress transformers generation warnings
warnings.filterwarnings("ignore", message="The following generation flags")

sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline.detector import Detector
from src.pipeline.subquestion_generator import SubquestionGenerator
from src.pipeline.attribute_agent import AttributeAgent
from src.pipeline.relationship_agent import RelationshipAgent
from src.pipeline.count_processor import CountProcessor
from src.pipeline.scene_attribute_agent import SceneAttributeAgent
from src.pipeline.problog_builder import ProbLogBuilder
from src.pipeline.problog_executor import ProbLogExecutor
from src.core.knowledge_base import KnowledgeBase
from src.core.types import PipelineResult


def main():
    print("=" * 80)
    print("PROVE Pipeline: Subquery-Driven Probabilistic Reasoning")
    print("=" * 80)

    # Ultimate question to answer
    ultimate_question = "Are there more birds in image A than in image B and are all birds orange?"
    print(f"\n🎯 ULTIMATE QUESTION: {ultimate_question}\n")

    # Initialize knowledge base
    kb = KnowledgeBase(ultimate_question)

    # Test images
    image_paths = {
        "image_a": "./test_images/dev-473-3-img0.png",
        "image_b": "./test_images/dev-473-3-img1.png"
    }

    try:
        # ========================================
        # STEP 1: Image Context Generation
        # ========================================
        print("Step 1: Image Context Generation")
        print("-" * 80)

        detector = Detector()
        image_contexts = detector.generate_detailed_captions(image_paths)

        for image_id, context in image_contexts.items():
            print(f"  {image_id}: {context}")
            kb.add_scene_context(image_id, {"caption": context})

        print()

        # ========================================
        # STEP 2: Object Detection
        # ========================================
        print("Step 2: Object Detection")
        print("-" * 80)

        for image_id, image_path in image_paths.items():
            print(f"  Processing {image_id}...")

            if not os.path.exists(image_path):
                print(f"    ⚠ Warning: Image not found: {image_path}")
                continue

            caption = image_contexts[image_id]
            objects = detector.detect_from_caption(image_path, caption, visualize=True)
            kb.add_objects(image_id, objects)

            print(f"    ✓ Detected {len(objects)} objects: {[obj.label for obj in objects]}")

        total_objects = sum(len(image_data.objects) for image_data in kb.images.values())
        print(f"\n  Total objects detected: {total_objects}")
        print()

        # ========================================
        # STEP 3: Subquestion Generation
        # ========================================
        print("Step 3: Subquestion Generation")
        print("-" * 80)

        subquestion_generator = SubquestionGenerator()
        subquestions = subquestion_generator.generate_binary_subquestions(
            ultimate_question, kb.images
        )
        kb.add_subquestions(subquestions)

        print(f"  Generated {len(subquestions)} binary subquestions:")
        for i, sq in enumerate(subquestions, 1):
            print(f"    {i}. {sq.question} ({sq.subquestion_type})")

        print()

        # ========================================
        # STEP 4: Route Subquestions by Type
        # ========================================
        print("Step 4: Route Subquestions by Type")
        print("-" * 80)

        attribute_subquestions = [sq for sq in kb.subquestions if sq.subquestion_type == "attribute"]
        relationship_subquestions = [sq for sq in kb.subquestions if sq.subquestion_type == "relationship"]
        scene_attribute_subquestions = [sq for sq in kb.subquestions if sq.subquestion_type == "scene_attribute"]
        count_subquestions = [sq for sq in kb.subquestions if sq.subquestion_type == "count"]

        print(f"  Attribute: {len(attribute_subquestions)} subquestions")
        print(f"  Relationship: {len(relationship_subquestions)} subquestions")
        print(f"  Scene Attribute: {len(scene_attribute_subquestions)} subquestions")
        print(f"  Count: {len(count_subquestions)} subquestions")
        print()

        # ========================================
        # STEP 5: Attribute Agent (Evidence Collection)
        # ========================================
        print("Step 5: Attribute Agent - Collect Evidence")
        print("-" * 80)

        if attribute_subquestions:
            print(f"  Processing {len(attribute_subquestions)} attribute subquestions...")
            attribute_agent = AttributeAgent(debug=False)
            attributes_per_image = attribute_agent.process_attribute_subquestions(
                attribute_subquestions, image_paths, kb.images
            )
            print(f"  ✓ Attribute evidence collected")
            for image_id, count in attributes_per_image.items():
                print(f"    {image_id}: {count} attributes")
        else:
            print("  No attribute subquestions")

        print()

        # ========================================
        # STEP 6: Relationship Agent (Evidence Collection)
        # ========================================
        print("Step 6: Relationship Agent - Collect Evidence")
        print("-" * 80)

        if relationship_subquestions:
            print(f"  Processing {len(relationship_subquestions)} relationship subquestions...")
            relationship_agent = RelationshipAgent(debug=False)
            relationships = relationship_agent.process_relationship_subquestions(
                relationship_subquestions, image_paths, kb.images
            )

            # Store relationships
            relationships_by_image = {}
            for relationship in relationships:
                try:
                    parts = relationship.subject_id.split('_')
                    if len(parts) >= 2:
                        image_id = f"image_{parts[1]}"
                    else:
                        image_id = "image_a"
                except (AttributeError, IndexError):
                    image_id = "image_a"

                if image_id not in relationships_by_image:
                    relationships_by_image[image_id] = []
                relationships_by_image[image_id].append(relationship)

            for image_id, image_relationships in relationships_by_image.items():
                kb.add_relationships_for_image(image_id, image_relationships)
                print(f"    {image_id}: {len(image_relationships)} relationships")

            print(f"  ✓ Relationship evidence collected")
        else:
            print("  No relationship subquestions")

        print()

        # ========================================
        # STEP 7: Count Processor (Evidence Collection)
        # ========================================
        print("Step 7: Count Processor - Collect Evidence")
        print("-" * 80)

        if count_subquestions:
            print(f"  Processing {len(count_subquestions)} count subquestions...")
            count_processor = CountProcessor()
            counts_per_image = count_processor.process_count_subquestions(
                count_subquestions, kb.images
            )
            print(f"  ✓ Count evidence collected")
            for image_id, count in counts_per_image.items():
                print(f"    {image_id}: {count} object classes counted")
        else:
            print("  No count subquestions")

        print()

        # ========================================
        # STEP 8: Scene Attribute Agent (Evidence Collection)
        # ========================================
        print("Step 8: Scene Attribute Agent - Collect Evidence")
        print("-" * 80)

        if scene_attribute_subquestions:
            print(f"  Processing {len(scene_attribute_subquestions)} scene attribute subquestions...")
            scene_attribute_agent = SceneAttributeAgent(debug=False)
            scene_attributes_counts = scene_attribute_agent.process_scene_attribute_subquestions(
                scene_attribute_subquestions, image_paths, kb.images, image_contexts
            )
            print(f"  ✓ Scene attribute evidence collected")
            for image_id, count in scene_attributes_counts.items():
                print(f"    {image_id}: {count} scene attributes")
        else:
            print("  No scene attribute subquestions")

        print()

        # ========================================
        # STEP 9: Build ProbLog Knowledge Base
        # ========================================
        print("Step 9: Build ProbLog Knowledge Base")
        print("-" * 80)

        problog_builder = ProbLogBuilder()
        problog_facts = problog_builder.build_knowledge_base(kb.images)
        kb.add_problog_facts(problog_facts)

        summary = problog_builder.get_building_summary(problog_facts)
        print(f"  Total facts: {summary['total_facts']}")
        print(f"  Average confidence: {summary['avg_confidence']:.3f}")
        print(f"  Fact breakdown:")
        for predicate, count in summary['predicates'].items():
            print(f"    {predicate}: {count} facts")

        print()

        # ========================================
        # STEP 10: ProbLog Execution (WITH Ultimate Composition)
        # ========================================
        print("Step 10: ProbLog Execution with Ultimate Composition")
        print("-" * 80)

        problog_executor = ProbLogExecutor()

        # Execute with ultimate question composition
        subquestion_results, ultimate_probability = problog_executor.execute_subquestions(
            kb.subquestions,
            kb.problog_facts,
            ultimate_question=ultimate_question  # NEW: Pass ultimate question
        )

        kb.add_subquestion_results(subquestion_results)

        # Read the generated ProbLog program
        with open('knowledge_base.pl', 'r') as f:
            problog_program = f.read()

        print()

        # ========================================
        # FINAL RESULT: Pure Probabilistic Output
        # ========================================
        print("=" * 80)
        print("PIPELINE RESULT")
        print("=" * 80)

        # Create pipeline result
        result = PipelineResult(
            ultimate_question=ultimate_question,
            ultimate_probability=ultimate_probability,
            subquestion_results=subquestion_results,
            problog_program=problog_program
        )

        # Display result
        print(f"\n🎯 ULTIMATE QUESTION:")
        print(f"   {result.ultimate_question}")
        print(f"\n📊 PROBABILITY: {result.ultimate_probability:.4f}")

        print(f"\n📋 SUBQUESTION EVIDENCE ({len(result.subquestion_results)} total):")
        for i, sq_result in enumerate(result.subquestion_results, 1):
            print(f"   {i}. {sq_result.subquestion}")
            print(f"      → {sq_result.probability:.4f}")

        print(f"\n📄 PROBLOG PROGRAM: knowledge_base.pl")
        print(f"   {len(problog_program)} characters")

        # Calculate evidence summary
        total_objects = sum(len(img.objects) for img in kb.images.values())
        total_attributes = sum(len(img.attributes) for img in kb.images.values())
        total_relationships = sum(len(img.relationships) for img in kb.images.values())
        total_scene_attributes = sum(len(img.scene_attributes) for img in kb.images.values())
        total_counts = sum(len(img.counts) for img in kb.images.values())

        print(f"\n📊 EVIDENCE COLLECTED:")
        print(f"   Objects: {total_objects}")
        print(f"   Attributes: {total_attributes}")
        print(f"   Relationships: {total_relationships}")
        print(f"   Scene Attributes: {total_scene_attributes}")
        print(f"   Count Distributions: {total_counts}")

        print("\n" + "=" * 80)
        print("✓ PIPELINE COMPLETE")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
