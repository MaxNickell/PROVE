#!/usr/bin/env python3
"""
PROVE pipeline test - Steps 1-5: Object detection through Qwen verification.
Tests the complete evidence collection pipeline with Qwen 2.5-VL-7B integration.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline.detector import Detector
from src.pipeline.attribute_extractor import AttributeExtractor
from src.pipeline.intra_question_generator import IntraQuestionGenerator
from src.pipeline.inter_question_generator import InterQuestionGenerator
from src.pipeline.qwen_verifier import QwenVerifier
from src.core.knowledge_base import KnowledgeBase

def main():
    print("=== PROVE Pipeline Test: Steps 1-5 ===")
    print("Question: What is uniquely similar about these images?")
    print()
    
    # Initialize components
    print("Step 1: Initializing core components...")
    detector = Detector()
    extractor = AttributeExtractor()
    intra_generator = IntraQuestionGenerator()
    inter_generator = InterQuestionGenerator()
    verifier = QwenVerifier()
    kb = KnowledgeBase("What is uniquely similar about these images?")
    print("✓ All components initialized with ModelManager singleton")
    
    # Test images
    image_paths = [
        "./test_images/dev-505-2-img0.png",
        "./test_images/dev-505-2-img1.png"
    ]
    
    # Store objects for question generation
    all_objects = {}
    
    # Step 2: Object detection and attribute extraction
    print("\nStep 2: Processing images...")
    for i, image_path in enumerate(image_paths):
        image_id = 'a' if i == 0 else 'b'
        print(f"Processing Image {image_id.upper()}: {image_path}")
        
        # Object detection with Florence-2
        objects = detector.detect(image_path, visualize=True)
        print(f"  ✓ Detected {len(objects)} objects: {[obj.label for obj in objects]}")
        
        # Attribute extraction with LLM candidates (enhanced)
        attributes = extractor.extract_attributes_with_candidates(
            image_path, objects, 
            ultimate_question="What is uniquely similar about these images?",
            save_crops=True
        )
        print(f"  ✓ Extracted contextual attributes for {len(attributes)} objects")
        
        # Store for question generation
        all_objects[image_id] = objects
        
        # Add to knowledge base
        kb.add_objects(image_id, objects)
        kb.add_attributes(image_id, attributes)
    
    # Step 3: LLM-driven candidate generation
    print("\nStep 3: Generating LLM-driven candidates and questions...")
    
    # Store relation candidates and attribute candidates
    relation_candidates_by_image = {}
    attribute_candidates = {}
    
    # Generate intra-relationship candidates for each image using LLM
    for image_id, objects in all_objects.items():
        if len(objects) >= 2:
            # Generate contextual relation candidates using LLM
            relation_candidates = intra_generator.generate_relation_candidates(
                "What is uniquely similar about these images?", 
                objects
            )
            relation_candidates_by_image[image_id] = relation_candidates
            
            # Also generate questions for backward compatibility
            intra_questions = intra_generator.generate_questions(
                "What is uniquely similar about these images?", 
                objects
            )
            kb.set_intra_questions(image_id, intra_questions)
            
            # Count total relation candidates
            total_candidates = sum(len(relations) for relations in relation_candidates.values())
            print(f"  ✓ Generated {total_candidates} relation candidates for Image {image_id.upper()}")
            
            # Show sample candidates
            for (obj1_id, obj2_id), relations in list(relation_candidates.items())[:2]:
                obj1 = next(obj for obj in objects if obj.object_id == obj1_id)
                obj2 = next(obj for obj in objects if obj.object_id == obj2_id)
                print(f"    - {obj1.label}-{obj2.label}: {', '.join(relations[:3])}")
    
    # Generate inter-comparison attribute candidates using LLM
    if 'a' in all_objects and 'b' in all_objects:
        # Generate contextual attribute candidates
        attribute_candidates = inter_generator.generate_attribute_candidates(
            "What is uniquely similar about these images?",
            all_objects['a'],
            all_objects['b']
        )
        
        # Also generate questions for backward compatibility
        inter_questions = inter_generator.generate_questions(
            "What is uniquely similar about these images?",
            all_objects['a'],
            all_objects['b']
        )
        kb.set_inter_questions(inter_questions)
        
        total_attr_candidates = sum(len(attrs) for attrs in attribute_candidates.values())
        print(f"  ✓ Generated {total_attr_candidates} attribute candidates for inter-comparisons")
        
        # Show sample attribute candidates
        for (obj_a_id, obj_b_id), attributes in list(attribute_candidates.items())[:2]:
            obj_a = next(obj for obj in all_objects['a'] if obj.object_id == obj_a_id)
            obj_b = next(obj for obj in all_objects['b'] if obj.object_id == obj_b_id)
            print(f"    - {obj_a.label}-{obj_b.label}: {', '.join(attributes[:3])}")
    
    # Step 5: Qwen verification with LLM-generated candidates
    print("\nStep 5: Verifying LLM candidates with Qwen VL...")
    
    # Verify intra-relationships using new LLM-driven approach
    for image_id, objects in all_objects.items():
        relation_candidates = relation_candidates_by_image.get(image_id, {})
        if relation_candidates and len(objects) >= 2:
            image_path = image_paths[0 if image_id == 'a' else 1]
            
            # Use Qwen with LLM candidates and full image context
            intra_results = verifier.verify_intra_relations(
                image_path, objects, relation_candidates
            )
            kb.add_intra_relations(image_id, intra_results)
            print(f"  ✓ Verified {len(intra_results)} relation candidates for Image {image_id.upper()}")
            
            # Show results with specific relations and binary probabilities
            for result in intra_results[:3]:  # Show first 3
                relation = result.get('relation', 'unknown')
                probability = result.get('probability', 0.0)
                obj1_label = result.get('object_1_label', 'obj1')
                obj2_label = result.get('object_2_label', 'obj2')
                print(f"    - {obj1_label} {relation} {obj2_label}: {probability:.1f}")
    
    # Verify inter-comparisons using attribute candidates (enhanced)
    if 'a' in all_objects and 'b' in all_objects and attribute_candidates:
        inter_results = verifier.verify_inter_comparisons(
            image_paths[0], image_paths[1],
            all_objects['a'], all_objects['b'],
            attribute_candidates
        )
        kb.add_inter_comparisons(inter_results)
        print(f"  ✓ Verified {len(inter_results)} inter-comparisons")
        for result in inter_results[:3]:  # Show first 3
                attribute = result.get('attribute', 'unknown')
                value_a = result.get('value_a', 'unknown')
                value_b = result.get('value_b', 'unknown')
                conf_a = result.get('confidence_a', 0.0)
                conf_b = result.get('confidence_b', 0.0)
                print(f"    - {attribute}: {value_a} ({conf_a:.1f}) vs {value_b} ({conf_b:.1f})")
    
    # Save complete results
    print("\nSaving complete evidence chain...")
    kb.save_to_file('./pipeline_results.json')
    print("✓ Results saved to pipeline_results.json")
    
    # Summary
    print(f"\n=== LLM-Driven Pipeline Summary ===")
    print(f"✓ Step 1: Core infrastructure initialized with ModelManager singleton")
    print(f"✓ Step 2: {len(all_objects)} images processed with contextual attribute extraction") 
    print(f"✓ Step 3: LLM-generated relation & attribute candidates")
    print(f"✓ Step 4: Image processing utilities ready for Qwen verification")
    print(f"✓ Step 5: Qwen VL verification with direct logit probabilities")
    print(f"\n🎯 Results: Unconstrained Qwen responses with direct logit confidence")
    print(f"🎯 Enhanced: LLM reasoning + Qwen visual extraction with native bbox support")
    print(f"\nReady for Steps 6-10: ProbLog reasoning and answer generation")

if __name__ == "__main__":
    main()