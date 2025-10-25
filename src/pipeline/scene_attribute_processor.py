"""
Scene Attribute Processor for PROVE pipeline.
Public API wrapper around SceneAttributeAgent for backward compatibility.

This file provides the same interface as the old scene_attribute_processor,
but now uses the agentic approach internally.
"""

from typing import List, Dict
from src.pipeline.scene_attribute_agent import SceneAttributeAgent, SceneAttributeAgentError
from src.core.types import BinarySubquestion, ImageData


class SceneAttributeProcessorError(RuntimeError):
    """Custom exception for scene attribute processing failures (backward compatibility)."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return self.message


class SceneAttributeProcessor:
    """
    Scene attribute processor using agentic approach.
    Wraps SceneAttributeAgent to maintain backward-compatible API.
    """

    def __init__(self, max_qwen_calls: int = 15, debug: bool = False):
        """
        Initialize processor with agentic backend.

        Args:
            max_qwen_calls: Maximum Qwen VL calls per subquestion
            debug: If True, saves images and prints detailed verification info
        """
        self.agent = SceneAttributeAgent(max_qwen_calls=max_qwen_calls, debug=debug)

    def process_scene_attribute_subquestions(
        self,
        scene_subquestions: List[BinarySubquestion],
        image_paths: Dict[str, str],
        images: Dict[str, ImageData],
        image_contexts: Dict[str, str] = None
    ) -> Dict[str, int]:
        """
        Process scene_attribute subquestions using agentic approach.

        This method provides the same interface as the old processor,
        but now uses the agentic loop internally.

        Args:
            scene_subquestions: List of scene_attribute binary subquestions
            image_paths: Dict mapping image_id to file path
            images: ImageData structure containing objects per image
            image_contexts: Optional (kept for API compatibility, not used by agent)

        Returns:
            Dict[str, int]: Count of scene attributes extracted per image
            Scene attributes are stored directly in ImageData.scene_attributes field

        Raises:
            SceneAttributeProcessorError: If processing fails
        """
        try:
            # Delegate to agent
            return self.agent.process_scene_attribute_subquestions(
                scene_subquestions, image_paths, images, image_contexts
            )
        except SceneAttributeAgentError as e:
            raise SceneAttributeProcessorError(str(e))
        except Exception as e:
            raise SceneAttributeProcessorError(f"Failed to process scene attribute subquestions: {str(e)}")
