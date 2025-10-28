#!/usr/bin/env python3
"""
PROVE Pipeline - Simplified End-to-End Test
Runs the complete 11-step visual reasoning pipeline and generates a final answer.
"""

import sys
import os
import warnings

# Suppress transformers warnings for cleaner output
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
from src.pipeline.answer_generator import AnswerGenerator
from src.core.knowledge_base import KnowledgeBase


def main():
    # Clear Python cache to avoid stale imports
    if 'src.pipeline.count_processor' in sys.modules:
        del sys.modules['src.pipeline.count_processor']

    # Configuration
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    ultimate_question = "What is unique in both of these images?"
    image_paths = {
        "image_a": "./test_images/dev-473-3-img0.png",
        "image_b": "./test_images/dev-473-3-img1.png"
    }

    print("=" * 60)
    print("PROVE: Visual Reasoning Pipeline")
    print("=" * 60)
    print(f"\nQuestion: {ultimate_question}\n")

    try:
        # Initialize knowledge base
        kb = KnowledgeBase(ultimate_question)

        # ========================================
        # STEP 1: Image Context Generation
        # ========================================
        print("[1/11] Generating image captions...")
        detector = Detector()
        image_contexts = detector.generate_detailed_captions(image_paths)

        for image_id, context in image_contexts.items():
            kb.add_scene_context(image_id, {"caption": context})
            if DEBUG:
                print(f"  {image_id}: {context[:100]}...")

        print(f"  ✓ Captions generated for {len(image_contexts)} images\n")

        # ========================================
        # STEP 2: Object Detection
        # ========================================
        print("[2/11] Detecting objects...")
        total_objects = 0
        for image_id, image_path in image_paths.items():
            caption = image_contexts[image_id]
            objects = detector.detect_from_caption(image_path, caption, visualize=DEBUG)
            kb.add_objects(image_id, objects)
            total_objects += len(objects)
            if DEBUG:
                print(f"  {image_id}: {len(objects)} objects - {[obj.label for obj in objects]}")

        print(f"  ✓ Detected {total_objects} objects across {len(image_paths)} images\n")

        # ========================================
        # STEP 3: Subquestion Generation
        # ========================================
        print("[3/11] Generating subquestions...")
        subquestion_generator = SubquestionGenerator()
        subquestions = subquestion_generator.generate_binary_subquestions(
            ultimate_question, kb.images
        )
        kb.add_subquestions(subquestions)

        if DEBUG:
            for i, sq in enumerate(subquestions, 1):
                print(f"  {i}. [{sq.subquestion_type}] {sq.question}")

        print(f"  ✓ Generated {len(subquestions)} binary subquestions\n")

        # Route subquestions by type
        attribute_subquestions = [sq for sq in subquestions if sq.subquestion_type == "attribute"]
        relationship_subquestions = [sq for sq in subquestions if sq.subquestion_type == "relationship"]
        scene_attribute_subquestions = [sq for sq in subquestions if sq.subquestion_type == "scene_attribute"]
        count_subquestions = [sq for sq in subquestions if sq.subquestion_type == "count"]

        # ========================================
        # STEP 4: Attribute Processing
        # ========================================
        print("[4/11] Processing attributes...")
        if attribute_subquestions:
            attribute_agent = AttributeAgent(debug=DEBUG)
            attributes_per_image = attribute_agent.process_attribute_subquestions(
                attribute_subquestions, image_paths, kb.images
            )
            total_attributes = sum(attributes_per_image.values())
            print(f"  ✓ Extracted {total_attributes} attributes\n")
        else:
            print(f"  ✓ No attributes to process\n")

        # ========================================
        # STEP 5: Relationship Extraction
        # ========================================
        print("[5/11] Processing relationships...")
        if relationship_subquestions:
            try:
                relationship_agent = RelationshipAgent(debug=DEBUG)
                relationships = relationship_agent.process_relationship_subquestions(
                    relationship_subquestions, image_paths, kb.images
                )

                # Store relationships (KB auto-groups by image)
                kb.add_relationships(relationships)

                print(f"  ✓ Extracted {len(relationships)} relationships\n")
            except Exception as e:
                print(f"  ⚠ Warning: Relationship extraction failed: {e}")
                if DEBUG:
                    import traceback
                    traceback.print_exc()
                print(f"  → Continuing pipeline without relationships\n")
        else:
            print(f"  ✓ No relationships to process\n")

        # ========================================
        # STEP 6: Count Processing
        # ========================================
        print("[6/11] Processing counts...")
        if count_subquestions:
            count_processor = CountProcessor()
            counts_per_image = count_processor.process_count_subquestions(
                count_subquestions, kb.images
            )
            total_counts = sum(counts_per_image.values())
            print(f"  ✓ Computed {total_counts} probabilistic counts\n")
        else:
            print(f"  ✓ No counts to process\n")

        # ========================================
        # STEP 7: Scene Attribute Processing
        # ========================================
        print("[7/11] Processing scene attributes...")
        if scene_attribute_subquestions:
            scene_attribute_agent = SceneAttributeAgent()
            scene_attributes_counts = scene_attribute_agent.process_scene_attribute_subquestions(
                scene_attribute_subquestions, image_paths, kb.images, image_contexts
            )
            total_scene_attrs = sum(
                len(kb.images[img_id].scene_attributes)
                for img_id in scene_attributes_counts.keys()
            )
            print(f"  ✓ Extracted {total_scene_attrs} scene attributes\n")
        else:
            print(f"  ✓ No scene attributes to process\n")

        # ========================================
        # STEP 8: ProbLog Knowledge Base Generation
        # ========================================
        print("[8/11] Building ProbLog knowledge base...")
        problog_builder = ProbLogBuilder()
        problog_facts = problog_builder.build_knowledge_base(kb.images)
        kb.add_problog_facts(problog_facts)

        summary = problog_builder.get_building_summary(problog_facts)
        prolog_program = problog_builder.facts_to_prolog_string(problog_facts)

        with open("knowledge_base.pl", "w") as f:
            f.write(prolog_program)

        print(f"  ✓ Built {summary['total_facts']} facts (avg confidence: {summary['avg_confidence']:.3f})")
        if DEBUG:
            for predicate, count in summary['predicates'].items():
                print(f"    - {predicate}: {count} facts")
        print()

        # ========================================
        # STEP 9: ProbLog Execution
        # ========================================
        print("[9/11] Executing ProbLog reasoning...")
        problog_executor = ProbLogExecutor()
        subquestion_results = problog_executor.execute_subquestions(
            kb.subquestions, kb.problog_facts
        )
        kb.add_subquestion_results(subquestion_results)

        print(f"  ✓ Executed {len(subquestion_results)} subquestions")
        if DEBUG:
            for result in subquestion_results:
                print(f"    - {result.subquestion[:60]}... → P={result.probability:.3f}")
        print()

        # ========================================
        # STEP 10: Final Answer Generation
        # ========================================
        print("[10/11] Generating final answer...")
        answer_generator = AnswerGenerator()

        # Create image_contexts dict from ImageData
        image_context_dict = {
            image_id: image_data.scene_context.get("caption", "")
            for image_id, image_data in kb.images.items()
        }

        final_answer = answer_generator.generate_final_answer(
            ultimate_question, kb.subquestion_results, image_context_dict
        )
        kb.set_answer(final_answer)
        print(f"  ✓ Answer generated\n")

        # ========================================
        # STEP 11: Save Results
        # ========================================
        print("[11/11] Saving results...")
        kb.save_to_file("pipeline_results.json")
        print(f"  ✓ Results saved to pipeline_results.json\n")

        # ========================================
        # Display Final Answer
        # ========================================
        print("=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(f"\nQuestion:")
        print(f"  {ultimate_question}\n")
        print(f"Answer:")
        print(f"  {final_answer.text}\n")
        print(f"Explanation:")
        for line in final_answer.explanation.split('\n'):
            print(f"  {line}")
        print("\n" + "=" * 60)

        print("\n✓ Pipeline completed successfully!")
        return 0

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
