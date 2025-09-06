"""
VLM Interface for PROVE Pipeline.
Provides abstraction layer for swapping between different Vision-Language Models.
"""

from abc import ABC, abstractmethod
from typing import Union, Dict, Any
from PIL import Image
import numpy as np


class VLMInterface(ABC):
    """
    Abstract base class for Vision-Language Models.
    
    This interface allows easy swapping between different VLMs:
    - LLaVA
    - GPT-4V
    - Claude Vision
    - Gemini Vision
    - QwenVL
    - Any future VLM
    """
    
    @abstractmethod
    def run_inference(self, image: Union[Image.Image, str], prompt: str) -> str:
        """
        Run inference on an image with a text prompt.
        
        Args:
            image: PIL Image object or path to image file
            prompt: Text prompt for the vision-language model
            
        Returns:
            str: Model response as text
            
        Raises:
            VLMError: If inference fails
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the VLM model.
        
        Returns:
            str: Model name (e.g., "llava-1.5-7b", "gpt-4-vision", "claude-3-vision")
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the VLM is available and ready for inference.
        
        Returns:
            bool: True if model is loaded and ready, False otherwise
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the VLM model.
        
        Returns:
            Dict[str, Any]: Model information including name, version, capabilities
        """
        return {
            "name": self.get_model_name(),
            "available": self.is_available(),
            "type": "vision_language_model"
        }


class VLMError(Exception):
    """Base exception for VLM-related errors."""
    
    def __init__(self, message: str, model_name: str = "unknown"):
        """
        Initialize VLM error.
        
        Args:
            message: Error message
            model_name: Name of the VLM that caused the error
        """
        self.message = message
        self.model_name = model_name
        super().__init__(f"[{model_name}] {message}")
    
    def __str__(self):
        return f"VLM Error [{self.model_name}]: {self.message}"


class VLMNotAvailableError(VLMError):
    """Exception raised when VLM is not available or not loaded."""
    pass


class VLMInferenceError(VLMError):
    """Exception raised when VLM inference fails."""
    pass


class VLMConfigurationError(VLMError):
    """Exception raised when VLM configuration is invalid."""
    pass


# VLM Provider Registry for easy switching
class VLMRegistry:
    """Registry for managing available VLM providers."""
    
    _providers = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """
        Register a VLM provider.
        
        Args:
            name: Provider name (e.g., "llava", "gpt4v", "claude")
            provider_class: Class implementing VLMInterface
        """
        if not issubclass(provider_class, VLMInterface):
            raise VLMConfigurationError(f"Provider {name} must implement VLMInterface")
        cls._providers[name] = provider_class
    
    @classmethod
    def get_provider(cls, name: str) -> type:
        """
        Get a registered VLM provider class.
        
        Args:
            name: Provider name
            
        Returns:
            type: Provider class
            
        Raises:
            VLMConfigurationError: If provider not found
        """
        if name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise VLMConfigurationError(f"VLM provider '{name}' not found. Available: {available}")
        return cls._providers[name]
    
    @classmethod
    def list_providers(cls) -> list:
        """
        List all registered VLM providers.
        
        Returns:
            list: List of provider names
        """
        return list(cls._providers.keys())
    
    @classmethod
    def create_provider(cls, name: str, **kwargs) -> VLMInterface:
        """
        Create an instance of a VLM provider.
        
        Args:
            name: Provider name
            **kwargs: Arguments to pass to provider constructor
            
        Returns:
            VLMInterface: VLM provider instance
        """
        provider_class = cls.get_provider(name)
        return provider_class(**kwargs)


# Default VLM configuration
DEFAULT_VLM_PROVIDER = "llava"
DEFAULT_VLM_CONFIG = {
    "temperature": 0.1,
    "max_tokens": 512,
    "timeout": 30
}