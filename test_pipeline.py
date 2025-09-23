#!/usr/bin/env python3
"""
PROVE Pipeline Test - Complete 11-Step Subquery-Driven Architecture.
Tests the complete subquery-driven evidence extraction pipeline with binary verification.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline.detector import Detector
from src.pipeline.subquery_generator import SubqueryGenerator
from src.pipeline.attribute_planner import AttributePlanner
from src.pipeline.attribute_extractor import AttributeExtractor
from src.pipeline.relationship_extractor import RelationshipExtractor
from src.pipeline.context_processor import ContextProcessor
from src.pipeline.problog_builder import ProbLogBuilder
from src.pipeline.problog_executor import ProbLogExecutor
from src.pipeline.answer_generator import AnswerGenerator
from src.core.knowledge_base import KnowledgeBase


def main():
    print("=== PROVE Pipeline: 11-Step Subquery-Driven Architecture ===")
    
    # Test imports first
    print("Testing component imports...")
    try:
        # Test all component imports
        detector = Detector()
        subquery_generator = SubqueryGenerator()
        attribute_planner = AttributePlanner()
        attribute_extractor = AttributeExtractor()
        relationship_extractor = RelationshipExtractor()
        context_processor = ContextProcessor()
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
    ultimate_question = "What is uniquely similar about these images?"
    print(f"Ultimate Question: {ultimate_question}")
    print()
    
    # Initialize knowledge base
    kb = KnowledgeBase(ultimate_question)
    
    # Test images
    image_paths = {
        "image_a": "./test_images/dev-473-3-img0.png",
        "image_b": "./test_images/dev-473-3-img1.png"
    }
    
    try:
        # ========================================
        # STEP 1: Object Detection
        # ========================================
        print("Step 1: Object Extraction")
        print("-" * 40)
        
        detector = Detector()
        
        for image_id, image_path in image_paths.items():
            print(f"Processing {image_id}: {image_path}")
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Detect objects
            objects = detector.detect(image_path, visualize=True)
            kb.add_objects(image_id, objects)
            
            print(f"  ✓ Detected {len(objects)} objects: {[obj.label for obj in objects]}")
        
        print(f"Total objects detected: {sum(len(image_data.objects) for image_data in kb.images.values())}")
        print()
        
        # ========================================
        # STEP 2: Detailed Image Context Generation
        # ========================================
        print("Step 2: Image Context Generation")
        print("-" * 40)
        
        # Generate detailed captions
        image_contexts = detector.generate_detailed_captions(image_paths)
        
        for image_id, context in image_contexts.items():
            kb.add_image_context(image_id, context)
            print(f"{image_id}: {context}")
        
        print()
        
        # ========================================
        # STEP 3: Contextual Subquery Generation
        # ========================================
        print("Step 3: Subquery Generation")
        print("-" * 40)
        
        subquery_generator = SubqueryGenerator()
        
        # Generate binary subqueries using clean ImageData structure
        subqueries = subquery_generator.generate_binary_subqueries(
            ultimate_question, kb.images
        )
        
        kb.add_subqueries(subqueries)
        
        print(f"Generated {len(subqueries)} binary subqueries:")
        for i, subquery in enumerate(subqueries, 1):
            print(f"  {i}. {subquery.question}")
            print(f"     Type: {subquery.subquery_type}")
            print(f"     Objects: {subquery.referenced_objects}")
        
        print()
        
        # ========================================
        # STEP 4: Subquery Routing by Type
        # ========================================
        print("Step 4: Subquery Type Routing")
        print("-" * 40)

        if not kb.subqueries:
            print("Warning: No subqueries generated, skipping processing")
            attribute_subqueries = []
            relationship_subqueries = []
            scene_attribute_subqueries = []
            count_subqueries = []
        else:
            # Filter subqueries by type for targeted processing
            attribute_subqueries = [sq for sq in kb.subqueries if sq.subquery_type == "attribute"]
            relationship_subqueries = [sq for sq in kb.subqueries if sq.subquery_type == "relationship"]
            scene_attribute_subqueries = [sq for sq in kb.subqueries if sq.subquery_type == "scene_attribute"]
            count_subqueries = [sq for sq in kb.subqueries if sq.subquery_type == "count"]

            print(f"Subquery routing:")
            print(f"  Attribute: {len(attribute_subqueries)} subqueries")
            print(f"  Relationship: {len(relationship_subqueries)} subqueries")
            print(f"  Scene Attribute: {len(scene_attribute_subqueries)} subqueries")
            print(f"  Count: {len(count_subqueries)} subqueries")

            # Skip count processing for now
            if count_subqueries:
                print(f"  → Skipping {len(count_subqueries)} count subqueries (not implemented)")

        print()

        # ========================================
        # STEP 5: Attribute Planning (Attribute Subqueries Only)
        # ========================================
        print("Step 5: Attribute Planning")
        print("-" * 40)

        if not attribute_subqueries:
            print("No attribute subqueries to process")
            requirements = []
        else:
            print(f"Processing {len(attribute_subqueries)} attribute subqueries:")
            print("  (Note: Enhanced to handle compound subqueries requiring cross-object comparisons)")
            for i, sq in enumerate(attribute_subqueries[:3], 1):  # Show first 3
                print(f"  {i}. {sq.question}")
            if len(attribute_subqueries) > 3:
                print(f"     ... and {len(attribute_subqueries) - 3} more")

            attribute_planner = AttributePlanner()

            # Create objects dict from ImageData
            all_objects = {image_id: image_data.objects for image_id, image_data in kb.images.items()}

            # Determine required attributes from ONLY attribute subqueries
            requirements = attribute_planner.determine_required_attributes(
                attribute_subqueries, all_objects
            )

            print(f"Determined attribute requirements for {len(requirements)} objects:")
            for req in requirements:
                print(f"  {req.image_id} Object {req.object_id}: {req.attribute_classes}")
                print(f"    Required for: {len(req.required_for_subqueries)} subqueries")

        print()
        
        # ========================================
        # STEP 6: Attribute Value Extraction
        # ========================================
        print("Step 6: Attribute Extraction")
        print("-" * 40)

        if not requirements:
            print("No attribute requirements to process")
        else:
            attribute_extractor = AttributeExtractor()

            # Extract attributes using clean ImageData interface
            attributes = attribute_extractor.extract_attributes(
                image_paths, kb.images, requirements
            )

            # Store attributes using new clean structure
            # Match attributes back to their requirements to get image_id and object_id
            total_attributes = 0
            for i, attr_data in enumerate(attributes):
                if i < len(requirements):
                    requirement = requirements[i]
                    image_id = requirement.image_id
                    object_id = requirement.object_id

                    # Store using per-object method
                    kb.add_attributes_for_object(image_id, object_id, attr_data)
                    total_attributes += 1

            print(f"Extracted attributes for {total_attributes} objects")

        print()

        # ========================================
        # STEP 7: Relationship Extraction (Relationship Subqueries Only)
        # ========================================
        print("Step 7: Relationship Extraction")
        print("-" * 40)

        if not relationship_subqueries:
            print("No relationship subqueries to process")
        else:
            print(f"Processing {len(relationship_subqueries)} relationship subqueries:")
            print("  (Note: Enhanced to handle compound subqueries requiring cross-image relationships)")
            for i, sq in enumerate(relationship_subqueries[:3], 1):  # Show first 3
                print(f"  {i}. {sq.question}")
            if len(relationship_subqueries) > 3:
                print(f"     ... and {len(relationship_subqueries) - 3} more")
            relationship_extractor = RelationshipExtractor()

            # Process only relationship subqueries
            relationships = relationship_extractor.extract_relationships(
                relationship_subqueries, image_paths, kb.images
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
        # STEP 8: Scene Attribute Processing (Placeholder)
        # ========================================
        print("Step 8: Scene Attribute Processing")
        print("-" * 40)

        if not scene_attribute_subqueries:
            print("No scene attribute subqueries to process")
        else:
            print(f"Processing {len(scene_attribute_subqueries)} scene attribute subqueries:")
            print("  (Note: Each subquery may decompose into multiple atomic binary questions)")
            for i, sq in enumerate(scene_attribute_subqueries, 1):
                print(f"  {i}. {sq.question}")
                print(f"     Type: {sq.subquery_type}, Referenced Objects: {sq.referenced_objects}")

            # Process scene attributes using new ContextProcessor
            try:
                print("  Initializing ContextProcessor...")
                context_processor = ContextProcessor()

                print(f"  Processing scene attributes with {len(image_paths)} images...")
                scene_context = context_processor.process_scene_attribute_subqueries(
                    scene_attribute_subqueries, image_paths, kb.images
                )

                print(f"  Scene context processor returned data for {len(scene_context)} images")

                # Store scene attributes in knowledge base
                total_attributes = 0
                images_with_attributes = 0

                for image_id, context_data in scene_context.items():
                    scene_attributes = context_data.get("scene_attributes", [])
                    print(f"  {image_id}: Found {len(scene_attributes)} scene attributes")

                    if scene_attributes:
                        images_with_attributes += 1
                        print(f"    Scene attributes for {image_id}:")
                        for attr in scene_attributes:
                            print(f"      {attr['attribute_class']}: {attr['value']} (confidence: {attr['confidence']:.3f})")
                            total_attributes += 1

                        # Store in knowledge base using proper method
                        print(f"    Storing {len(scene_attributes)} scene attributes in knowledge base...")
                        kb.add_scene_attributes(image_id, scene_attributes)
                    else:
                        print(f"    No scene attributes extracted for {image_id}")

                print(f"✓ Scene attribute processing completed:")
                print(f"  - Total scene attributes extracted: {total_attributes}")
                print(f"  - Images with scene attributes: {images_with_attributes}/{len(scene_context)}")

            except Exception as e:
                print(f"❌ Scene attribute processing failed: {e}")
                import traceback
                traceback.print_exc()

        print()

        # ========================================
        # KNOWLEDGE BASE CONSTRUCTION COMPLETE
        # ========================================
        # The following steps are commented out to focus on knowledge base construction:
        # - ProbLog fact construction
        # - ProbLog query execution
        # - Final answer generation

        # problog_builder = ProbLogBuilder()
        #
        # # Build ProbLog facts from clean ImageData structure
        # problog_facts = problog_builder.build_knowledge_base(kb.images)
        #
        # kb.add_problog_facts(problog_facts)
        #
        # print(f"Built knowledge base with {len(problog_facts)} ProbLog facts:")
        #
        # # Show sample facts by predicate
        # facts_by_predicate = {}
        # for fact in problog_facts:
        #     if fact.predicate not in facts_by_predicate:
        #         facts_by_predicate[fact.predicate] = []
        #     facts_by_predicate[fact.predicate].append(fact)
        #
        # for predicate, facts in facts_by_predicate.items():
        #     print(f"  {predicate}: {len(facts)} facts")
        #     for fact in facts[:2]:  # Show first 2 facts
        #         print(f"    {fact.to_prolog_string()}")
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
        total_scene_attributes = sum(len(img.scene_context.get("scene_attributes", [])) for img in kb.images.values())

        print(f"  Images processed: {len(kb.images)}")
        print(f"  Total objects detected: {total_objects}")
        print(f"  Total attributes extracted: {total_attributes}")
        print(f"  Total relationships extracted: {total_relationships}")
        print(f"  Total scene attributes extracted: {total_scene_attributes}")
        print(f"  Subqueries generated: {len(kb.subqueries)}")

        # Show breakdown by image
        print(f"\nBreakdown by image:")
        for image_id, image_data in kb.images.items():
            scene_attrs_count = len(image_data.scene_context.get("scene_attributes", []))
            print(f"  {image_id}:")
            print(f"    Objects: {len(image_data.objects)}")
            print(f"    Attributes: {len(image_data.attributes)}")
            print(f"    Relationships: {len(image_data.relationships)}")
            print(f"    Scene Attributes: {scene_attrs_count}")

            # Show scene attributes if they exist
            if scene_attrs_count > 0:
                scene_attrs = image_data.scene_context.get("scene_attributes", [])
                for attr in scene_attrs[:3]:  # Show first 3
                    print(f"      - {attr.get('attribute_class', 'unknown')}: {attr.get('value', 'unknown')} (conf: {attr.get('confidence', 0):.3f})")
                if scene_attrs_count > 3:
                    print(f"      ... and {scene_attrs_count - 3} more")

        # Show subquery breakdown by type
        print(f"\nSubquery breakdown:")
        subquery_types = {}
        for sq in kb.subqueries:
            subquery_types[sq.subquery_type] = subquery_types.get(sq.subquery_type, 0) + 1
        for sq_type, count in subquery_types.items():
            print(f"  {sq_type}: {count}")

        # Validate scene context before JSON save
        print(f"\n" + "=" * 40)
        print("SCENE CONTEXT VALIDATION")
        print("=" * 40)

        scene_attrs_found = False
        for image_id, image_data in kb.images.items():
            scene_context = image_data.scene_context
            scene_attrs = scene_context.get("scene_attributes", [])

            print(f"{image_id} scene_context keys: {list(scene_context.keys())}")
            if scene_attrs:
                scene_attrs_found = True
                print(f"  ✓ {len(scene_attrs)} scene attributes found")
                for i, attr in enumerate(scene_attrs[:2], 1):  # Show first 2
                    print(f"    {i}. {attr}")
            else:
                print(f"  ⚠ No scene attributes found in scene_context")

        if scene_attrs_found:
            print("✓ Scene attributes successfully stored in knowledge base")
        else:
            print("❌ WARNING: No scene attributes found in any image!")

        # Clean up captions from knowledge base (they should not be in final KB for ProbLog)
        print(f"\n" + "=" * 40)
        print("KNOWLEDGE BASE CLEANUP")
        print("=" * 40)
        print("Removing captions from knowledge base (captions are for processing only, not ProbLog)")

        captions_removed = 0
        for image_id, image_data in kb.images.items():
            if "caption" in image_data.scene_context:
                del image_data.scene_context["caption"]
                captions_removed += 1

        print(f"✓ Removed captions from {captions_removed} images")
        print("✓ Knowledge base ready for ProbLog mapping (Steps 8-11)")

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