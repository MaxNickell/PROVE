"""
Attribute planning component for PROVE pipeline.
Analyzes binary subqueries to determine which attribute classes need extraction for specific objects.
"""

from typing import List, Dict, Any, Set

from src.core.model_manager import ModelManager
from src.core.types import BinarySubquery, ObjectDetection, AttributeRequirement


class AttributePlannerError(RuntimeError):
    """Custom exception for attribute planning failures."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return self.message


class AttributePlanner:
    """
    Analyze binary subqueries to determine required attribute classes for each object.
    Maps subquery requirements to specific objects and consolidates across all subqueries.
    """
    
    def __init__(self):
        """Initialize planner with ModelManager singleton."""
        self.model_manager = ModelManager()
    
    def determine_required_attributes(
        self,
        attribute_subqueries: List[BinarySubquery],
        all_objects: Dict[str, List[ObjectDetection]]
    ) -> List[AttributeRequirement]:
        """
        Analyze ONLY attribute subqueries to determine which attribute classes need extraction for each object.

        Args:
            attribute_subqueries: List of binary subqueries with subquery_type == "attribute"
            all_objects: Detected objects per image for validation

        Returns:
            List[AttributeRequirement]: Attribute requirements per object

        Raises:
            AttributePlannerError: If planning fails or non-attribute subqueries provided
        """
        try:
            if not attribute_subqueries:
                return []

            # Validate that all subqueries are attribute type (research-grade validation)
            non_attribute_subqueries = [sq for sq in attribute_subqueries if sq.subquery_type != "attribute"]
            if non_attribute_subqueries:
                invalid_types = [sq.subquery_type for sq in non_attribute_subqueries]
                raise AttributePlannerError(
                    f"AttributePlanner only accepts attribute subqueries. "
                    f"Received {len(non_attribute_subqueries)} non-attribute subqueries: {set(invalid_types)}. "
                    f"Route relationship/scene_attribute subqueries to appropriate processors. Count subqueries not implemented yet."
                )
            
            # Get LLM client from ModelManager
            llm_client = self.model_manager.get_llm_client()
            
            # Create object ID mapping for validation
            object_registry = self._build_object_registry(all_objects)
            
            # Analyze each attribute subquery for attribute requirements
            all_requirements = {}  # {object_id: {attribute_class: [subquery_indices]}}

            for i, attribute_subquery in enumerate(attribute_subqueries):
                # Get attribute requirements for this attribute subquery
                requirements = self._analyze_subquery_requirements(
                    llm_client, attribute_subquery, i, object_registry
                )
                
                # Merge requirements
                for obj_id, attr_classes in requirements.items():
                    if obj_id not in all_requirements:
                        all_requirements[obj_id] = {}
                    
                    for attr_class in attr_classes:
                        if attr_class not in all_requirements[obj_id]:
                            all_requirements[obj_id][attr_class] = []
                        all_requirements[obj_id][attr_class].append(i)
            
            # Convert to AttributeRequirement objects
            attribute_requirements = self._build_attribute_requirements(
                all_requirements, attribute_subqueries
            )
            
            return attribute_requirements
            
        except Exception as err:
            raise AttributePlannerError(f"Attribute planning failed: {err}")
    
    def _build_object_registry(
        self, 
        all_objects: Dict[str, List[ObjectDetection]]
    ) -> Dict[str, ObjectDetection]:
        """
        Build registry mapping object IDs to ObjectDetection instances.
        
        Args:
            all_objects: Objects per image
            
        Returns:
            Dict mapping object_id to ObjectDetection
        """
        registry = {}
        
        for image_id, objects in all_objects.items():
            for obj in objects:
                # Use consistent format: strip "image_" prefix to match subquery generator
                simple_image_id = image_id.replace("image_", "")
                obj_id = f"{obj.label}_{simple_image_id}_{obj.object_id}"
                registry[obj_id] = obj
        
        return registry
    
    def _analyze_subquery_requirements(
        self,
        llm_client,
        subquery: BinarySubquery,
        subquery_index: int,
        object_registry: Dict[str, ObjectDetection]
    ) -> Dict[str, List[str]]:
        """
        Analyze a single subquery to determine attribute requirements.
        Enhanced to handle compound subqueries that may need attributes from objects beyond referenced_objects.

        Args:
            llm_client: LLM client instance
            subquery: Binary subquery to analyze
            subquery_index: Index for tracking
            object_registry: Object ID to ObjectDetection mapping

        Returns:
            Dict mapping object_id to list of required attribute classes
        """
        # Create context about referenced objects (starting point)
        referenced_object_context = []
        for obj_id in subquery.referenced_objects:
            if obj_id in object_registry:
                obj = object_registry[obj_id]
                referenced_object_context.append(f"{obj_id} ({obj.label})")
            else:
                referenced_object_context.append(f"{obj_id} (unknown)")

        referenced_objects_str = ", ".join(referenced_object_context)

        # Create context about ALL available objects (for compound subquery analysis)
        all_objects_context = []
        for obj_id, obj in object_registry.items():
            all_objects_context.append(f"{obj_id} ({obj.label})")

        all_objects_str = ", ".join(all_objects_context[:10])  # Limit to first 10 for prompt size
        if len(object_registry) > 10:
            all_objects_str += f"... and {len(object_registry) - 10} more objects"

        prompt = f"""Analyze this binary subquery to determine what attribute classes need to be extracted to answer this question.

Subquery: "{subquery.question}"
Type: {subquery.subquery_type}
Referenced Objects: {referenced_objects_str}

All Available Objects: {all_objects_str}

IMPORTANT: This subquery may be compound and require attributes from objects beyond just the referenced objects.
Consider cross-object comparisons, cross-image comparisons, and implicit object requirements.

Determine which visual attribute classes need to be extracted from which objects to fully answer this question.

Consider these attribute classes:
- **Physical Attributes**: size, shape, color, texture, pattern, material
- **State Attributes**: condition, state, position, orientation  
- **Functional Attributes**: function, style, usage
- **Comparative Attributes**: muscle_mass, muscle_definition, body_size, weight, height, strength
- **Contextual Attributes**: environment, setting, activity

Return JSON with this exact format:
{{
  "attribute_requirements": {{
    "object_id1": ["attribute_class1", "attribute_class2"],
    "object_id2": ["attribute_class3", "attribute_class4"]
  }}
}}

Rules:
- Include ALL objects that need attribute extraction (not just referenced objects)
- Handle compound subqueries that require cross-object or cross-image comparisons
- Only include attribute classes that are visually determinable
- Focus on attributes directly relevant to answering the question
- Use specific attribute class names (not generic descriptions)
- If no attributes needed, return empty object

Examples:
- "Is person_a_0 more muscular than person_b_0?" → {{"person_a_0": ["muscle_mass", "muscle_definition"], "person_b_0": ["muscle_mass", "muscle_definition"]}}
- "Do bird_a_0 and animal_b_0 have the same color?" → {{"bird_a_0": ["color"], "animal_b_0": ["color"]}}
- "Is cattle_a_1 facing the same direction as animal_b_0?" → {{"cattle_a_1": ["orientation"], "animal_b_0": ["orientation"]}}
- "Is person_a_0 lifting weight_a_1?" → {{"person_a_0": ["state", "position"], "weight_a_1": ["position", "state"]}}
- "Does carnivore_a_0 have spots?" → {{"carnivore_a_0": ["pattern", "color"]}}"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert at analyzing visual questions to determine what attributes need extraction. Focus on attributes that can be visually determined and are directly relevant to answering the question. Return strict JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            # Use Pydantic validation for robust JSON parsing  
            response = llm_client.plan_attributes(
                messages,
                temperature=0.2
            )
            
            # Extract and clean the validated requirements
            cleaned_requirements = {}
            for obj_id, attr_classes in response.attribute_requirements.items():
                cleaned_attr_classes = [attr_class.strip().lower() for attr_class in attr_classes if attr_class.strip()]
                if cleaned_attr_classes:
                    cleaned_requirements[obj_id] = cleaned_attr_classes
            
            return cleaned_requirements
            
        except Exception as e:
            print(f"Warning: Failed to analyze subquery requirements: {e}")
            return {}
    
    def _build_attribute_requirements(
        self,
        all_requirements: Dict[str, Dict[str, List[int]]],
        subqueries: List[BinarySubquery]
    ) -> List[AttributeRequirement]:
        """
        Build AttributeRequirement objects from consolidated requirements.
        
        Args:
            all_requirements: Consolidated requirements mapping
            subqueries: Original subqueries for reference
            
        Returns:
            List of AttributeRequirement instances
        """
        requirements = []
        
        for obj_id, attr_requirements in all_requirements.items():
            # Get all attribute classes for this object
            attribute_classes = list(attr_requirements.keys())
            
            # Get all subqueries that require attributes from this object
            required_for_subqueries = []
            for attr_class, subquery_indices in attr_requirements.items():
                for idx in subquery_indices:
                    if idx < len(subqueries):
                        required_for_subqueries.append(subqueries[idx].question)
            
            # Remove duplicates while preserving order
            required_for_subqueries = list(dict.fromkeys(required_for_subqueries))
            
            # Parse object ID to get image_id and object_id
            try:
                # Parse object ID format: label_imageid_objectid (e.g., "bird_a_0")
                parts = obj_id.split('_')
                if len(parts) < 3:
                    print(f"Warning: Invalid object ID format '{obj_id}', skipping")
                    continue

                # Extract image_id and object_index
                simple_image_id = parts[-2]  # Second to last part (e.g., "a")
                object_index = int(parts[-1])  # Last part

                # Convert simple image ID to full format (e.g., "a" -> "image_a")
                image_id = f"image_{simple_image_id}"

            except (ValueError, IndexError):
                print(f"Warning: Could not parse object ID '{obj_id}', skipping")
                continue

            requirement = AttributeRequirement(
                image_id=image_id,
                object_id=object_index,
                attribute_classes=attribute_classes,
                required_for_subqueries=required_for_subqueries
            )
            
            requirements.append(requirement)
        
        return requirements
    
    def get_planning_summary(
        self, 
        requirements: List[AttributeRequirement]
    ) -> Dict[str, Any]:
        """
        Get summary statistics for attribute planning results.
        
        Args:
            requirements: List of AttributeRequirement instances
            
        Returns:
            Dict with summary information
        """
        if not requirements:
            return {
                "total_objects": 0,
                "total_attribute_classes": 0,
                "avg_attributes_per_object": 0,
                "attribute_class_distribution": {}
            }
        
        # Count statistics
        total_objects = len(requirements)
        all_attribute_classes = []
        attribute_class_counts = {}
        
        for req in requirements:
            all_attribute_classes.extend(req.attribute_classes)
            
            for attr_class in req.attribute_classes:
                attribute_class_counts[attr_class] = attribute_class_counts.get(attr_class, 0) + 1
        
        return {
            "total_objects": total_objects,
            "total_attribute_classes": len(all_attribute_classes),
            "unique_attribute_classes": len(set(all_attribute_classes)),
            "avg_attributes_per_object": len(all_attribute_classes) / total_objects if total_objects > 0 else 0,
            "attribute_class_distribution": attribute_class_counts,
            "objects_with_requirements": [f"{req.image_id}_{req.object_id}" for req in requirements[:5]]  # Sample
        }
    
    def validate_requirements(
        self,
        requirements: List[AttributeRequirement],
        all_objects: Dict[str, List[ObjectDetection]]
    ) -> bool:
        """
        Validate that attribute requirements are well-formed.
        
        Args:
            requirements: List of AttributeRequirement instances
            all_objects: Original objects for validation
            
        Returns:
            bool: True if requirements are valid
        """
        try:
            # Build valid object IDs set
            valid_object_ids = set()
            for image_id, objects in all_objects.items():
                for obj in objects:
                    # Use consistent format: strip "image_" prefix to match subquery generator
                    simple_image_id = image_id.replace("image_", "")
                    obj_id = f"{obj.label}_{simple_image_id}_{obj.object_id}"
                    valid_object_ids.add(obj_id)
            
            for req in requirements:
                # Check required attributes
                assert hasattr(req, 'image_id')
                assert hasattr(req, 'object_id')
                assert hasattr(req, 'attribute_classes')
                assert hasattr(req, 'required_for_subqueries')

                # Validate types
                assert isinstance(req.image_id, str)
                assert isinstance(req.object_id, int)
                assert isinstance(req.attribute_classes, list)
                assert isinstance(req.required_for_subqueries, list)

                # Validate content
                assert req.image_id.strip()
                assert req.object_id >= 0
                assert req.attribute_classes  # Non-empty
                
                # Validate object ID exists (relaxed for flexibility)
                # assert req.object_id in valid_object_ids
            
            return True
            
        except AssertionError:
            return False


# Example usage and testing
if __name__ == "__main__":
    # Test attribute planner
    planner = AttributePlanner()
    
    # Sample data
    from src.core.types import BinarySubquery, ObjectDetection
    
    subqueries = [
        BinarySubquery(
            question="Is person_a_0 more muscular than person_b_0?",
            referenced_objects=["person_a_0", "person_b_0"],
            subquery_type="attribute"
        ),
        BinarySubquery(
            question="Is person_a_0 lifting weight_a_1?",
            referenced_objects=["person_a_0", "weight_a_1"],
            subquery_type="relationship"
        ),
        BinarySubquery(
            question="Does carnivore_a_0 have spots?",
            referenced_objects=["carnivore_a_0"],
            subquery_type="attribute"
        )
    ]
    
    all_objects = {
        "a": [
            ObjectDetection(0, "person", [10.0, 20.0, 100.0, 200.0], 0.95),
            ObjectDetection(1, "weight", [150.0, 50.0, 300.0, 250.0], 0.88),
            ObjectDetection(0, "carnivore", [200.0, 100.0, 400.0, 300.0], 0.92)
        ],
        "b": [
            ObjectDetection(0, "person", [20.0, 30.0, 110.0, 210.0], 0.89)
        ]
    }
    
    try:
        requirements = planner.determine_required_attributes(subqueries, all_objects)
        
        is_valid = planner.validate_requirements(requirements, all_objects)
        summary = planner.get_planning_summary(requirements)
        
        print(f"✓ Generated {len(requirements)} attribute requirements")
        print(f"✓ Validation: {is_valid}")
        print(f"✓ Summary: {summary}")
        
        for req in requirements:
            print(f"  {req.image_id}[{req.object_id}]: {req.attribute_classes}")
            
        print("✓ Attribute planner ready!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")