"""
Relationship extraction component for PROVE pipeline.
Extracts spatial and interaction relationships using subquery-driven binary VLM verification.
"""

from typing import List, Dict, Any, Tuple
from PIL import Image

from src.core.model_manager import ModelManager
from src.core.types import BinarySubquery, ObjectDetection, RelationshipCandidate, IntraRelation, ImageData


class RelationshipExtractorError(RuntimeError):
    """Custom exception for relationship extraction failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return self.message


class RelationshipExtractor:
    """
    Extract spatial and interaction relationships using subquery-driven analysis.
    Determines required relationships from subqueries and verifies using binary VLM.
    """
    
    def __init__(self):
        """Initialize extractor with ModelManager singleton."""
        self.model_manager = ModelManager()
    
    def extract_relationships(
        self,
        relationship_subqueries: List[BinarySubquery],
        image_paths: Dict[str, str],  # {"image_a": "/path/to/image", ...}
        images: Dict[str, ImageData]  # Clean ImageData structure
    ) -> List[IntraRelation]:
        """
        Extract relationships needed to answer ONLY relationship subqueries.

        Args:
            relationship_subqueries: Binary subqueries with subquery_type == "relationship"
            image_paths: Paths to images
            images: ImageData structure containing objects and context per image

        Returns:
            List[IntraRelation]: Extracted relationships with confidence scores

        Raises:
            RelationshipExtractorError: If extraction fails or non-relationship subqueries provided
        """
        try:
            if not relationship_subqueries:
                return []

            # Validate that all subqueries are relationship type (research-grade validation)
            non_relationship_subqueries = [sq for sq in relationship_subqueries if sq.subquery_type != "relationship"]
            if non_relationship_subqueries:
                invalid_types = [sq.subquery_type for sq in non_relationship_subqueries]
                raise RelationshipExtractorError(
                    f"RelationshipExtractor only accepts relationship subqueries. "
                    f"Received {len(non_relationship_subqueries)} non-relationship subqueries: {set(invalid_types)}. "
                    f"Route attribute/scene_attribute subqueries to appropriate processors. Count subqueries not implemented yet."
                )
            
            # Get model clients
            llm_client = self.model_manager.get_llm_client()
            qwen_client = self.model_manager.get_qwen_vl()
            
            # Load PIL images
            loaded_images = {}
            for image_id, image_path in image_paths.items():
                loaded_images[image_id] = Image.open(image_path)

            # Determine required relationships from relationship subqueries
            relationship_candidates = self._determine_required_relationships(
                llm_client, relationship_subqueries, images
            )
            
            # Verify relationships using binary VLM
            verified_relations = []

            for candidate in relationship_candidates:
                relation = self._verify_relationship(
                    qwen_client, candidate, loaded_images, images
                )

                # All relationships are now returned with their probabilities
                if relation:
                    verified_relations.append(relation)

            return verified_relations
            
        except Exception as err:
            raise RelationshipExtractorError(f"Relationship extraction failed: {err}")
    
    def extract_relationships_for_image(
        self,
        subqueries: List[BinarySubquery],
        image_path: str,
        objects: List[ObjectDetection]
    ) -> List[IntraRelation]:
        """
        Extract relationships for objects in a single image.
        
        Args:
            subqueries: Relationship-type subqueries
            image_path: Path to the image
            objects: Detected objects in this image
            
        Returns:
            List[IntraRelation]: Extracted relationships for this image
        """
        try:
            if not subqueries or not objects:
                return []
            
            # For now, return empty list - this needs full implementation
            # In the interest of getting the pipeline working, we'll implement this later
            print(f"    Relationship extraction for single image not yet fully implemented")
            return []
            
        except Exception as e:
            raise RelationshipExtractorError(f"Failed to extract relationships for image: {str(e)}")
    
    def _determine_required_relationships(
        self,
        llm_client,
        subqueries: List[BinarySubquery],
        images: Dict[str, ImageData]
    ) -> List[RelationshipCandidate]:
        """
        Analyze subqueries to determine what relationships need verification.

        Args:
            llm_client: LLM client
            subqueries: Binary subqueries to analyze
            images: ImageData structure containing objects and context per image

        Returns:
            List[RelationshipCandidate]: Relationships that need verification
        """
        all_candidates = []
        
        for i, subquery in enumerate(subqueries):
            # Only analyze relationship-type subqueries
            if subquery.subquery_type == "relationship":
                candidates = self._analyze_subquery_for_relationships(
                    llm_client, subquery, images
                )
                
                # Add subquery reference to candidates
                for candidate in candidates:
                    candidate.required_for_subqueries = [subquery.question]
                
                all_candidates.extend(candidates)
        
        # Consolidate duplicate relationships
        consolidated_candidates = self._consolidate_relationship_candidates(all_candidates)
        
        return consolidated_candidates
    
    def _analyze_subquery_for_relationships(
        self,
        llm_client,
        subquery: BinarySubquery,
        images: Dict[str, ImageData]
    ) -> List[RelationshipCandidate]:
        """
        Analyze a single subquery to determine required relationships.

        Args:
            llm_client: LLM client
            subquery: Binary subquery to analyze
            images: ImageData structure containing objects and context per image

        Returns:
            List[RelationshipCandidate]: Required relationships for this subquery
        """
        # Build context for referenced objects (starting point)
        referenced_object_context = self._build_object_context_from_images(subquery.referenced_objects, images)

        # Build context for ALL available objects (for compound subquery analysis)
        all_objects_context = self._build_all_objects_context_from_images(images)

        prompt = f"""Analyze this binary subquery to determine what spatial or interaction relationships need to be verified:

Subquery: "{subquery.question}"
Type: {subquery.subquery_type}
Referenced Objects: {referenced_object_context}

All Available Objects: {all_objects_context}

IMPORTANT: This subquery may be compound and require relationships between objects beyond just the referenced objects.
Consider cross-image relationship comparisons, multiple relationship types, and implicit object requirements.

Determine which object-to-object relationships need verification to fully answer this question.

Consider these relationship types:
- **Spatial**: near, far, above, below, left, right, inside, outside, touching
- **Interaction**: lifting, carrying, using, holding, supporting, following
- **State**: looking_at, facing, turned_away_from, approaching, avoiding

Return JSON with this exact format:
{{
  "relationships": [
    {{
      "subject_id": "object_id1",
      "object_id": "object_id2", 
      "relation": "relationship_name"
    }}
  ]
}}

Rules:
- Include ALL relationships needed to answer the subquery (not just between referenced objects)
- Handle compound subqueries that require cross-image relationship comparisons
- Use object IDs exactly as provided from all available objects
- Use specific relationship names (not generic descriptions)
- If no relationships needed, return empty array
- Focus on verifiable spatial/interaction relationships

Examples:
- "Is person_a_0 lifting weight_a_1?" → {{"relationships": [{{"subject_id": "person_a_0", "object_id": "weight_a_1", "relation": "lifting"}}]}}
- "Is carnivore_a_0 near zebra_a_1?" → {{"relationships": [{{"subject_id": "carnivore_a_0", "object_id": "zebra_a_1", "relation": "near"}}]}}
- "Do birds have the same spatial relationship to cattle in both images?" → {{"relationships": [{{"subject_id": "bird_a_0", "object_id": "cattle_a_1", "relation": "perched_on"}}, {{"subject_id": "bird_b_1", "object_id": "animal_b_0", "relation": "perched_on"}}]}}
- "Are bird_a_0 and bird_b_1 both touching their respective animals?" → {{"relationships": [{{"subject_id": "bird_a_0", "object_id": "cattle_a_1", "relation": "touching"}}, {{"subject_id": "bird_b_1", "object_id": "animal_b_0", "relation": "touching"}}]}}"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert at analyzing visual questions to determine what spatial and interaction relationships need verification. Focus on relationships that can be visually determined and are directly relevant to answering the question. Return strict JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            # Use Pydantic validation for robust JSON parsing
            response = llm_client.extract_relationships(
                messages,
                temperature=0.2
            )
            
            relationships = response.relationships
            
            # Convert RelationshipItem objects to RelationshipCandidate objects
            candidates = []
            for rel_item in relationships:
                try:
                    # Parse subject and object IDs to get image_id and object_id
                    subject_parts = rel_item.subject_id.split('_')
                    object_parts = rel_item.object_id.split('_')

                    if len(subject_parts) < 3 or len(object_parts) < 3:
                        print(f"Warning: Invalid object ID format: {rel_item.subject_id}, {rel_item.object_id}")
                        continue

                    # Extract image_id and object_index for subject
                    subject_simple_image_id = subject_parts[-2]
                    subject_object_index = int(subject_parts[-1])
                    subject_image_id = f"image_{subject_simple_image_id}"

                    # Extract image_id and object_index for object
                    object_simple_image_id = object_parts[-2]
                    object_object_index = int(object_parts[-1])
                    object_image_id = f"image_{object_simple_image_id}"

                    candidate = RelationshipCandidate(
                        image_id=subject_image_id,
                        subject_id=subject_object_index,
                        object_id=object_object_index,
                        relation=rel_item.relation.strip().lower(),
                        required_for_subqueries=[]  # Will be filled later
                    )
                    candidates.append(candidate)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse object IDs: {e}")
                    continue
            
            return candidates
            
        except Exception as e:
            print(f"Warning: Failed to analyze subquery for relationships: {e}")
            return []
    
    def _build_object_context_from_images(
        self,
        referenced_objects: List[str],
        images: Dict[str, ImageData]
    ) -> str:
        """
        Build context string for referenced objects from ImageData.

        Args:
            referenced_objects: Object IDs referenced in subquery
            images: ImageData structure containing objects and context per image

        Returns:
            str: Formatted object context
        """
        context_parts = []

        for obj_id in referenced_objects:
            # Try to find object details from ImageData
            obj_info = self._find_object_info_from_images(obj_id, images)
            if obj_info:
                context_parts.append(f"{obj_id} ({obj_info['label']}, conf={obj_info['confidence']:.2f})")
            else:
                context_parts.append(f"{obj_id} (unknown)")

        return ", ".join(context_parts)

    def _build_all_objects_context_from_images(
        self,
        images: Dict[str, ImageData]
    ) -> str:
        """
        Build context string for all available objects from ImageData.

        Args:
            images: ImageData structure containing objects and context per image

        Returns:
            str: Formatted context for all objects (limited for prompt size)
        """
        all_objects = []

        for image_id, image_data in images.items():
            # Extract simple image ID (e.g., "image_a" -> "a")
            simple_image_id = image_id.replace("image_", "")

            for obj in image_data.objects:
                obj_id = f"{obj.label}_{simple_image_id}_{obj.object_id}"
                all_objects.append(f"{obj_id} ({obj.label}, conf={obj.confidence:.2f})")

        # Limit to first 15 objects for prompt size
        if len(all_objects) <= 15:
            return ", ".join(all_objects)
        else:
            return ", ".join(all_objects[:15]) + f"... and {len(all_objects) - 15} more objects"

    def _find_object_info_from_images(
        self,
        object_id: str,
        images: Dict[str, ImageData]
    ) -> Dict[str, Any]:
        """
        Find object information by ID from ImageData structure.

        Args:
            object_id: Object ID to find
            images: ImageData structure containing objects per image

        Returns:
            Dict with object info or None if not found
        """
        try:
            # Parse object ID format: label_imageid_objectid
            parts = object_id.split('_')
            if len(parts) < 3:
                return None

            simple_image_id = parts[-2]  # Second to last part (e.g., "a")
            object_index = int(parts[-1])

            # Convert simple image ID back to full key (e.g., "a" -> "image_a")
            image_id = f"image_{simple_image_id}"

            if image_id in images:
                for obj in images[image_id].objects:
                    if obj.object_id == object_index:
                        return {
                            "label": obj.label,
                            "confidence": obj.confidence,
                            "bbox": obj.bbox
                        }

            return None

        except Exception:
            return None
    
    def _consolidate_relationship_candidates(
        self,
        candidates: List[RelationshipCandidate]
    ) -> List[RelationshipCandidate]:
        """
        Consolidate duplicate relationship candidates.
        
        Args:
            candidates: List of relationship candidates
            
        Returns:
            List of consolidated candidates
        """
        consolidated = {}
        
        for candidate in candidates:
            # Create unique key for relationship
            key = (candidate.image_id, candidate.subject_id, candidate.object_id, candidate.relation)
            
            if key not in consolidated:
                consolidated[key] = candidate
            else:
                # Merge required_for_subqueries
                existing = consolidated[key]
                existing.required_for_subqueries.extend(candidate.required_for_subqueries)
                # Remove duplicates
                existing.required_for_subqueries = list(set(existing.required_for_subqueries))
        
        return list(consolidated.values())
    
    def _verify_relationship(
        self,
        qwen_client,
        candidate: RelationshipCandidate,
        loaded_images: Dict[str, Image.Image],
        images: Dict[str, ImageData]
    ) -> IntraRelation:
        """
        Verify a relationship candidate using binary VLM.

        Args:
            qwen_client: Qwen VLM client
            candidate: Relationship candidate to verify
            loaded_images: Loaded PIL images
            images: ImageData structure containing objects and context

        Returns:
            IntraRelation if verified, None otherwise
        """
        try:
            # Find subject and object instances from ImageData
            subject_info = self._find_object_by_simple_ref_imagedata(candidate.image_id, candidate.subject_id, images)
            object_info = self._find_object_by_simple_ref_imagedata(candidate.image_id, candidate.object_id, images)

            if not subject_info or not object_info:
                print(f"Warning: Could not find objects for relationship {candidate.image_id} {candidate.subject_id} {candidate.relation} {candidate.object_id}")
                return None

            # Get the loaded PIL image
            image_id = candidate.image_id
            if image_id not in loaded_images:
                print(f"Warning: Image {image_id} not found")
                return None

            image = loaded_images[image_id]
            
            # Create binary verification question with bounding boxes and stronger Yes/No compliance
            subject_bbox = subject_info.bbox
            object_bbox = object_info.bbox

            question = f"""Look at these objects:
Subject: <box>({int(subject_bbox[0])},{int(subject_bbox[1])}),({int(subject_bbox[2])},{int(subject_bbox[3])})</box>{subject_info.label}
Object: <box>({int(object_bbox[0])},{int(object_bbox[1])}),({int(object_bbox[2])},{int(object_bbox[3])})</box>{object_info.label}

Question: Is the {subject_info.label} {candidate.relation} the {object_info.label}?

IMPORTANT: You must respond with exactly "Yes" or "No" only. Do not include any explanation or additional text.

Answer:"""
            
            # Get VLM response with logits
            response, logits = qwen_client.run_inference_with_logits(image, question)

            # Use proper softmax probability calculation for P(statement is true)
            prob_statement_true = qwen_client.extract_yes_no_probability_with_proper_softmax(logits, response)

            # Validate response format
            is_valid_response = qwen_client.validate_yes_no_response(response)
            if not is_valid_response:
                print(f"Warning: Invalid Yes/No response: '{response}' for relationship verification")

            # Always return relationship with probability - no filtering
            # ProbLog needs all probabilistic facts, including low-probability ones
            # Convert integer IDs back to string format for consistency
            subject_str_id = self._convert_to_string_id(candidate.image_id, candidate.subject_id, images)
            object_str_id = self._convert_to_string_id(candidate.image_id, candidate.object_id, images)

            return IntraRelation(
                subject_id=subject_str_id,
                object_id=object_str_id,
                relation=candidate.relation,
                probability=prob_statement_true
            )
            
        except Exception as e:
            print(f"Warning: Failed to verify relationship {candidate.relation}: {e}")
            return None

    def _convert_to_string_id(
        self,
        image_id: str,
        object_index: int,
        images: Dict[str, ImageData]
    ) -> str:
        """
        Convert integer object index back to string ID format.

        Args:
            image_id: Image ID (e.g., "image_a")
            object_index: Object index within image (0, 1, 2...)
            images: ImageData structure to lookup object label

        Returns:
            String object ID (e.g., "bird_a_0")
        """
        try:
            if image_id in images and object_index < len(images[image_id].objects):
                obj = images[image_id].objects[object_index]
                simple_image_id = image_id.replace("image_", "")
                return f"{obj.label}_{simple_image_id}_{object_index}"
            else:
                # Fallback if object not found
                simple_image_id = image_id.replace("image_", "")
                return f"unknown_{simple_image_id}_{object_index}"
        except Exception as e:
            print(f"Warning: Failed to convert object index {object_index} to string ID: {e}")
            simple_image_id = image_id.replace("image_", "")
            return f"unknown_{simple_image_id}_{object_index}"
    
    def _find_object_by_simple_ref_imagedata(
        self,
        image_id: str,
        object_id: int,
        images: Dict[str, ImageData]
    ) -> ObjectDetection:
        """
        Find object by simple image_id and object_id from ImageData structure.

        Args:
            image_id: Image identifier (e.g., "image_a")
            object_id: Object index within the image
            images: ImageData structure containing objects per image

        Returns:
            ObjectDetection: Found object or None if not found
        """
        try:
            if image_id in images:
                for obj in images[image_id].objects:
                    if obj.object_id == object_id:
                        return obj

            return None

        except Exception as e:
            print(f"Warning: Failed to find object {image_id} object {object_id}: {e}")
            return None
    
    def _find_object_details(
        self,
        object_id: str,
        all_objects: Dict[str, List[ObjectDetection]]
    ) -> Dict[str, Any]:
        """
        Find detailed object information by ID (legacy method).
        
        Args:
            object_id: Object ID to find
            all_objects: All detected objects
            
        Returns:
            Dict with detailed object info or None if not found
        """
        try:
            # Parse object ID format: label_imageid_objectid
            parts = object_id.split('_')
            if len(parts) < 3:
                return None
            
            simple_image_id = parts[-2]  # Second to last part (e.g., "a")
            object_index = int(parts[-1])
            
            # Convert simple image ID back to full key (e.g., "a" -> "image_a")
            image_id = f"image_{simple_image_id}"
            
            if image_id in all_objects:
                for obj in all_objects[image_id]:
                    if obj.object_id == object_index:
                        return {
                            "image_id": image_id,
                            "object_id": obj.object_id,
                            "label": obj.label,
                            "confidence": obj.confidence,
                            "bbox": obj.bbox
                        }
            
            return None
            
        except Exception as e:
            print(f"Warning: Failed to find object details for {object_id}: {e}")
            return None
    
    def get_extraction_summary(
        self, 
        relations: List[IntraRelation]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for relationship extraction results.
        
        Args:
            relations: List of IntraRelation instances
            
        Returns:
            Dict with summary information
        """
        if not relations:
            return {
                "total_relationships": 0,
                "avg_confidence": 0.0,
                "relationship_types": {},
                "unique_object_pairs": 0
            }
        
        total_relationships = len(relations)
        all_confidences = [rel.probability for rel in relations]
        avg_confidence = sum(all_confidences) / len(all_confidences)
        
        # Count relationship types
        relationship_types = {}
        unique_pairs = set()
        
        for rel in relations:
            relationship_types[rel.relation] = relationship_types.get(rel.relation, 0) + 1
            # Use new simple format with subject_id and object_id
            unique_pairs.add((rel.subject_id, rel.object_id))
        
        return {
            "total_relationships": total_relationships,
            "avg_confidence": avg_confidence,
            "relationship_types": relationship_types,
            "unique_object_pairs": len(unique_pairs),
            "confidence_distribution": {
                "high (>0.8)": len([c for c in all_confidences if c > 0.8]),
                "medium (0.5-0.8)": len([c for c in all_confidences if 0.5 <= c <= 0.8]),
                "low (<0.5)": len([c for c in all_confidences if c < 0.5])
            }
        }


# Example usage and testing
if __name__ == "__main__":
    # Test relationship extractor
    extractor = RelationshipExtractor()
    
    # Sample data
    from src.core.types import BinarySubquery, ObjectDetection
    
    subqueries = [
        BinarySubquery(
            question="Is person_a_0 lifting weight_a_1?",
            referenced_objects=["person_a_0", "weight_a_1"],
            subquery_type="relationship"
        ),
        BinarySubquery(
            question="Is carnivore_a_0 near zebra_a_1?",
            referenced_objects=["carnivore_a_0", "zebra_a_1"],
            subquery_type="relationship"
        )
    ]
    
    image_paths = {
        "image_a": "./test_images/dev-473-3-img0.png"
    }
    
    images = {
        "image_a": ImageData(
            objects=[
                ObjectDetection(0, "person", [10.0, 20.0, 100.0, 200.0], 0.95),
                ObjectDetection(1, "weight", [150.0, 50.0, 300.0, 250.0], 0.88),
                ObjectDetection(2, "carnivore", [200.0, 100.0, 400.0, 300.0], 0.92),
                ObjectDetection(3, "zebra", [300.0, 150.0, 500.0, 350.0], 0.87)
            ],
            attributes={},
            relationships=[],
            scene_context={}
        )
    }
    
    try:
        # Note: This test requires actual model clients to be available
        print("✓ RelationshipExtractor component created")
        print("✓ Ready for integration testing with model clients")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")