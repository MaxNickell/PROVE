#!/usr/bin/env python3
"""
PROVE Pipeline Test - Complete 11-Step Subquestion-Driven Architecture.
Tests the complete subquestion-driven evidence extraction pipeline with binary verification.
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
from src.pipeline.answer_generator import AnswerGenerator
from src.core.knowledge_base import KnowledgeBase


def main():
    # Clear Python cache to avoid stale imports (especially for count_processor)
    if 'src.pipeline.count_processor' in sys.modules:
        del sys.modules['src.pipeline.count_processor']

    print("=== PROVE Pipeline: 11-Step Subquery-Driven Architecture ===")

    # Test imports first
    print("Testing component imports...")
    try:
        # Test all component imports
        detector = Detector()
        subquestion_generator = SubquestionGenerator()
        attribute_agent = AttributeAgent()
        relationship_agent = RelationshipAgent()
        count_processor = CountProcessor()
        scene_attribute_agent = SceneAttributeAgent()
        problog_builder = ProbLogBuilder()
        problog_executor = ProbLogExecutor()
        answer_generator = AnswerGenerator()
        print("✓ All components imported successfully")
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Ultimate question to answer
    ultimate_question = "Do both images depict an sad and injured dog in an outdoor environment where both dogs are wearing a bright green harness or black collar?"
    print(f"Ultimate Question: {ultimate_question}")
    print()
    
    # Initialize knowledge base
    kb = KnowledgeBase(ultimate_question)
    
    # Test images
    image_paths = {
        "image_a": "./test_images/dev-350-2-img0.png",
        "image_b": "./test_images/dev-350-2-img1.png"
    }
    
    try:
        # ========================================
        # STEP 1: Image Context Generation
        # ========================================
        print("Step 1: Image Context Generation")
        print("-" * 40)

        detector = Detector()

        # Generate detailed captions upfront and store in KB
        image_contexts = detector.generate_detailed_captions(image_paths)

        for image_id, context in image_contexts.items():
            print(f"{image_id}: {context}")
            kb.add_scene_context(image_id, {"caption": context})

        print()

        # ========================================
        # STEP 2: Object Extraction
        # ========================================
        print("Step 2: Object Extraction")
        print("-" * 40)

        for image_id, image_path in image_paths.items():
            print(f"Processing {image_id}: {image_path}")

            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue

            # Detect objects using pre-generated caption
            caption = image_contexts[image_id]
            objects = detector.detect_from_caption(image_path, caption, visualize=True)
            kb.add_objects(image_id, objects)

            print(f"  ✓ Detected {len(objects)} objects: {[obj.label for obj in objects]}")

        print(f"Total objects detected: {sum(len(image_data.objects) for image_data in kb.images.values())}")
        print()
        
        # ========================================
        # STEP 3: Contextual Subquery Generation
        # ========================================
        print("Step 3: Subquery Generation")
        print("-" * 40)
        
        subquestion_generator = SubquestionGenerator()
        
        # Generate binary subquestions using clean ImageData structure
        subquestions = subquestion_generator.generate_binary_subquestions(
            ultimate_question, kb.images
        )

        kb.add_subquestions(subquestions)

        print(f"Generated {len(subquestions)} binary subquestions:")
        for i, subquestion in enumerate(subquestions, 1):
            print(f"  {i}. {subquestion.question}")
            print(f"     Type: {subquestion.subquestion_type}")
        
        print()
        
        # ========================================
        # STEP 4: Subquery Routing by Type
        # ========================================
        print("Step 4: Subquery Type Routing")
        print("-" * 40)

        if not kb.subquestions:
            print("Warning: No subquestions generated, skipping processing")
            attribute_subquestions = []
            relationship_subquestions = []
            scene_attribute_subquestions = []
            count_subquestions = []
        else:
            # Filter subquestions by type for targeted processing
            attribute_subquestions = [sq for sq in kb.subquestions if sq.subquestion_type == "attribute"]
            relationship_subquestions = [sq for sq in kb.subquestions if sq.subquestion_type == "relationship"]
            scene_attribute_subquestions = [sq for sq in kb.subquestions if sq.subquestion_type == "scene_attribute"]
            count_subquestions = [sq for sq in kb.subquestions if sq.subquestion_type == "count"]

            print(f"Subquestion routing:")
            print(f"  Attribute: {len(attribute_subquestions)} subquestions")
            print(f"  Relationship: {len(relationship_subquestions)} subquestions")
            print(f"  Scene Attribute: {len(scene_attribute_subquestions)} subquestions")
            print(f"  Count: {len(count_subquestions)} subquestions")

            # Count processing will be handled in dedicated step

        print()

        # ========================================
        # STEP 5: Attribute Processing (Per-Subquery Planning + Extraction)
        # ========================================
        print("Step 5: Attribute Processing")
        print("-" * 40)

        if not attribute_subquestions:
            print("No attribute subquestions to process")
        else:
            print(f"Processing {len(attribute_subquestions)} attribute subquestions")

            attribute_agent = AttributeAgent(debug=True)

            # Process attribute subquestions individually - returns attributes per image
            attributes_per_image = attribute_agent.process_attribute_subquestions(
                attribute_subquestions, image_paths, kb.images
            )

            print(f"Attribute processing completed:")
            for image_id, count in attributes_per_image.items():
                print(f"  {image_id}: {count} attributes extracted")

        print()

        # ========================================
        # STEP 6: Relationship Extraction (Relationship Subqueries Only)
        # ========================================
        print("Step 6: Relationship Extraction")
        print("-" * 40)

        if not relationship_subquestions:
            print("No relationship subquestions to process")
        else:
            print(f"Processing {len(relationship_subquestions)} relationship subquestions")

            relationship_agent = RelationshipAgent(debug=True)  # Enable debug mode

            # Process only relationship subquestions
            relationships = relationship_agent.process_relationship_subquestions(
                relationship_subquestions, image_paths, kb.images
            )

            # Group relationships by image for batch storage
            relationships_by_image = {}
            for relationship in relationships:
                # Parse image_id from string subject_id format (e.g., "bird_a_0" -> "image_a")
                try:
                    parts = relationship.subject_id.split('_')
                    if len(parts) >= 2:
                        image_id = f"image_{parts[1]}"
                    else:
                        image_id = "image_a"  # fallback
                except (AttributeError, IndexError):
                    image_id = "image_a"  # fallback

                # Group relationships by image
                if image_id not in relationships_by_image:
                    relationships_by_image[image_id] = []
                relationships_by_image[image_id].append(relationship)

            # Store relationships per image using correct KnowledgeBase method
            total_stored = 0
            for image_id, image_relationships in relationships_by_image.items():
                kb.add_relationships_for_image(image_id, image_relationships)
                total_stored += len(image_relationships)
                print(f"  Stored {len(image_relationships)} relationships for {image_id}")

            print(f"Total relationships extracted and stored: {total_stored}")

        print()

        # ========================================
        # STEP 7: Count Processing (Count Subqueries Only)
        # ========================================
        print("Step 7: Count Processing")
        print("-" * 40)

        if not count_subquestions:
            print("No count subquestions to process")
        else:
            print(f"Processing {len(count_subquestions)} count subquestions:")
            print("  (Note: Poisson-Binomial probabilistic counting using detection confidences)")
            for i, sq in enumerate(count_subquestions[:3], 1):  # Show first 3
                print(f"  {i}. {sq.question}")
            if len(count_subquestions) > 3:
                print(f"     ... and {len(count_subquestions) - 3} more")

            count_processor = CountProcessor()

            # Process count subquestions using Poisson-Binomial counting
            counts_per_image = count_processor.process_count_subquestions(
                count_subquestions, kb.images
            )

            print(f"Count processing completed:")
            for image_id, count in counts_per_image.items():
                print(f"  {image_id}: {count} object classes counted")

        print()

        # ========================================
        # STEP 8: Scene Attribute Processing
        # ========================================
        print("Step 8: Scene Attribute Processing")
        print("-" * 40)

        if not scene_attribute_subquestions:
            print("No scene attribute subquestions to process")
        else:
            print(f"Processing {len(scene_attribute_subquestions)} scene attribute subquestions")

            # Process scene attributes using agentic approach
            try:
                scene_attribute_agent = SceneAttributeAgent()

                scene_attributes_counts = scene_attribute_agent.process_scene_attribute_subquestions(
                    scene_attribute_subquestions, image_paths, kb.images, image_contexts
                )

                # Count scene attributes now stored directly in ImageData
                total_attributes = 0
                images_with_attributes = 0

                for image_id, count in scene_attributes_counts.items():
                    scene_attributes = kb.images[image_id].scene_attributes
                    print(f"  {image_id}: Found {len(scene_attributes)} scene attributes")

                    if scene_attributes:
                        images_with_attributes += 1
                        print(f"    Scene attributes for {image_id}:")
                        for attr_class, attr_values in scene_attributes.items():
                            for attr_value in attr_values:
                                print(f"      {attr_class}: {attr_value['value']} (confidence: {attr_value['confidence']:.3f})")
                                total_attributes += 1
                    else:
                        print(f"    No scene attributes extracted for {image_id}")

                print(f"✓ Scene attribute processing completed:")
                print(f"  - Total scene attributes extracted: {total_attributes}")
                print(f"  - Images with scene attributes: {images_with_attributes}/{len(scene_attributes_counts)}")

            except Exception as e:
                print(f"❌ Scene attribute processing failed: {e}")
                import traceback
                traceback.print_exc()

        print()

        # ========================================
        # STEP 9: ProbLog Knowledge Base Generation
        # ========================================
        print("Step 9: ProbLog Knowledge Base Generation")
        print("-" * 40)

        try:
            problog_builder = ProbLogBuilder()

            # Build ProbLog facts from clean ImageData structure
            problog_facts = problog_builder.build_knowledge_base(kb.images)

            # Add facts to knowledge base
            kb.add_problog_facts(problog_facts)

            # Get building summary
            summary = problog_builder.get_building_summary(problog_facts)

            print(f"Built ProbLog knowledge base:")
            print(f"  Total facts: {summary['total_facts']}")
            print(f"  Average confidence: {summary['avg_confidence']:.3f}")
            print(f"  Specification compliance: {summary['specification_compliance']}")

            # Show fact breakdown by predicate
            print(f"  Fact breakdown:")
            for predicate, count in summary['predicates'].items():
                print(f"    {predicate}: {count} facts")

            # Generate executable ProbLog program
            prolog_program = problog_builder.facts_to_prolog_string(problog_facts)

            # Save ProbLog program to file
            with open("knowledge_base.pl", "w") as f:
                f.write(prolog_program)
            print(f"  ✓ ProbLog program saved to knowledge_base.pl ({len(prolog_program)} characters)")

            # Show sample facts from knowledge_base.pl
            print(f"\n  Sample facts from knowledge_base.pl:")
            with open("knowledge_base.pl", "r") as f:
                lines = f.readlines()
                for line in lines[:10]:  # Show first 10 lines
                    print(f"    {line.rstrip()}")
                if len(lines) > 10:
                    print(f"    ... and {len(lines) - 10} more facts")

            # Show sample facts for verification
            facts_by_predicate = {}
            for fact in problog_facts:
                if fact.predicate not in facts_by_predicate:
                    facts_by_predicate[fact.predicate] = []
                facts_by_predicate[fact.predicate].append(fact)

            # Show 2 sample facts per predicate type
            print(f"  Sample facts:")
            predicate_order = ["entity", "attribute", "relation", "scene_attr", "count"]
            for predicate in predicate_order:
                if predicate in facts_by_predicate:
                    sample_facts = facts_by_predicate[predicate][:2]
                    for fact in sample_facts:
                        print(f"    {fact.to_prolog_string()}")

        except Exception as e:
            print(f"❌ ProbLog generation failed: {e}")
            import traceback
            traceback.print_exc()

        print()

        # ========================================
        # KNOWLEDGE BASE CONSTRUCTION COMPLETE
        # ========================================
        #
        # print()
        
        # ========================================
        # STEP 10: ProbLog Execution and Evidence Tracing (COMMENTED OUT)
        # ========================================
        # print("Step 10: ProbLog Execution and Evidence Tracing")
        # print("-" * 40)
        #
        # if not kb.subqueries or not kb.problog_facts:
        #     print("Warning: Insufficient data for ProbLog execution, generating placeholder results")
        #     subquery_results = []
        # else:
        #     problog_executor = ProbLogExecutor()
        #
        #     # Execute subqueries against knowledge base
        #     subquery_results = problog_executor.execute_subqueries(
        #         kb.subqueries, kb.problog_facts
        #     )
        #
        # kb.add_subquery_results(subquery_results)
        #
        # print(f"Executed {len(subquery_results)} subqueries:")
        # for result in subquery_results:
        #     print(f"  \"{result.subquery}\"")
        #     print(f"    Probability: {result.probability:.2f}")
        #     print(f"    Supporting facts: {len(result.supporting_facts)}")
        #     if result.evidence_trail:
        #         print(f"    Evidence: {result.evidence_trail[0]}")
        #
        # print()
        
        # ========================================
        # STEP 11: Final Answer Generation (COMMENTED OUT)
        # ========================================
        # print("Step 11: Final Answer Generation")
        # print("-" * 40)
        #
        # answer_generator = AnswerGenerator()
        #
        # # Create image_contexts dict from ImageData for compatibility
        # image_contexts = {}
        # for image_id, image_data in kb.images.items():
        #     caption = image_data.scene_context.get("caption", "")
        #     image_contexts[image_id] = caption
        #
        # # Generate final answer using subquery results
        # final_answer = answer_generator.generate_final_answer(
        #     ultimate_question, kb.subquery_results, image_contexts
        # )
        #
        # kb.set_answer(final_answer)
        #
        # print("FINAL ANSWER:")
        # print(f"Question: {ultimate_question}")
        # print(f"Answer: {final_answer.text}")
        # print()
        # print("EXPLANATION:")
        # print(final_answer.explanation)
        # print()
        
        # ========================================
        # Knowledge Base Construction Summary
        # ========================================
        print("=" * 60)
        print("KNOWLEDGE BASE CONSTRUCTION COMPLETE")
        print("=" * 60)

        # Custom summary of what was built
        print("Knowledge Base Summary:")
        total_objects = sum(len(img.objects) for img in kb.images.values())
        total_attributes = sum(len(img.attributes) for img in kb.images.values())
        total_relationships = sum(len(img.relationships) for img in kb.images.values())
        total_scene_attributes = sum(len(img.scene_attributes) for img in kb.images.values())
        total_counts = sum(len(img.counts) for img in kb.images.values())

        print(f"  Images processed: {len(kb.images)}")
        print(f"  Total objects detected: {total_objects}")
        print(f"  Total attributes extracted: {total_attributes}")
        print(f"  Total relationships extracted: {total_relationships}")
        print(f"  Total scene attributes extracted: {total_scene_attributes}")
        print(f"  Total object classes counted: {total_counts}")
        print(f"  Subquestions generated: {len(kb.subquestions)}")

        # Show breakdown by image
        print(f"\nBreakdown by image:")
        for image_id, image_data in kb.images.items():
            scene_attrs_count = len(image_data.scene_attributes)
            counts_count = len(image_data.counts)
            print(f"  {image_id}:")
            print(f"    Objects: {len(image_data.objects)}")
            print(f"    Attributes: {len(image_data.attributes)}")
            print(f"    Relationships: {len(image_data.relationships)}")
            print(f"    Scene Attributes: {scene_attrs_count}")
            print(f"    Counts: {counts_count}")

            # Show scene attributes if they exist
            if scene_attrs_count > 0:
                scene_attrs = image_data.scene_attributes
                shown_count = 0
                for attr_class, attr_values in list(scene_attrs.items())[:3]:  # Show first 3 classes
                    for attr_value in attr_values[:1]:  # Show first value of each class
                        print(f"      - {attr_class}: {attr_value['value']} (conf: {attr_value['confidence']:.3f})")
                        shown_count += 1
                if scene_attrs_count > shown_count:
                    print(f"      ... and {scene_attrs_count - shown_count} more")

            # Show counts if they exist
            if counts_count > 0:
                counts = image_data.counts
                for class_name, count_data in list(counts.items())[:3]:  # Show first 3 classes
                    distribution = count_data['distribution']
                    # Find most likely count and its probability
                    most_likely_count = max(distribution.keys(), key=lambda k: distribution[k])
                    most_likely_prob = distribution[most_likely_count]

                    # Show distribution summary
                    print(f"      - {class_name} distribution: P({most_likely_count})={most_likely_prob:.3f} (most likely)")

                    # Show top 2-3 probabilities for context
                    sorted_counts = sorted(distribution.items(), key=lambda x: x[1], reverse=True)
                    top_probs = []
                    for count_val, prob in sorted_counts[:3]:
                        if prob > 0.01:  # Only show probabilities > 1%
                            top_probs.append(f"P({count_val})={prob:.3f}")
                    if len(top_probs) > 1:
                        print(f"        Distribution: {', '.join(top_probs)}")

                if counts_count > 3:
                    print(f"      ... and {counts_count - 3} more count classes")

        # Show subquestion breakdown by type
        print(f"\nSubquestion breakdown:")
        subquestion_types = {}
        for sq in kb.subquestions:
            subquestion_types[sq.subquestion_type] = subquestion_types.get(sq.subquestion_type, 0) + 1
        for sq_type, count in subquestion_types.items():
            print(f"  {sq_type}: {count}")

        # Validate scene attributes before JSON save
        print(f"\n" + "=" * 40)
        print("SCENE ATTRIBUTES VALIDATION")
        print("=" * 40)

        scene_attrs_found = False
        for image_id, image_data in kb.images.items():
            scene_attrs = image_data.scene_attributes

            print(f"{image_id} scene_attributes: {len(scene_attrs)} items")
            if scene_attrs:
                scene_attrs_found = True
                print(f"  ✓ {len(scene_attrs)} scene attribute classes found")
                for i, (attr_class, attr_values) in enumerate(list(scene_attrs.items())[:2], 1):  # Show first 2
                    for j, attr_value in enumerate(attr_values[:1]):  # Show first value of each class
                        print(f"    {i}.{j+1}. {attr_class}: {attr_value['value']} (confidence: {attr_value['confidence']:.3f})")
            else:
                print(f"  ⚠ No scene attributes found")

        if scene_attrs_found:
            print("✓ Scene attributes successfully stored in knowledge base")
        else:
            print("❌ WARNING: No scene attributes found in any image!")

        print("✓ Knowledge base ready for ProbLog mapping (Steps 8-11)")
        print("  (Note: Captions were used as processing aids only, not stored in KB)")

        # Export to JSON
        print(f"\n" + "=" * 40)
        print("SAVING TO JSON")
        print("=" * 40)
        kb.save_to_file("pipeline_results.json")
        print(f"✓ Results saved to pipeline_results.json")

        print("\n🎯 Knowledge base construction completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)