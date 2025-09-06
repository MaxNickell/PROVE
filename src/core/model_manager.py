"""
ModelManager singleton for managing all model instances across the pipeline.
Critical for memory efficiency - ensures only one instance of each model.
"""

import threading
from typing import Optional, Dict, Any
import torch

from src.vision.florence2 import Florence2
from src.vision.llava import Llava
from src.language.llm_client import LLMClient
from src.core.vlm_interface import VLMInterface, VLMRegistry, DEFAULT_VLM_PROVIDER


class ModelManager:
    """Singleton class for managing all model instances across the pipeline."""
    
    _instance: Optional['ModelManager'] = None
    _lock = threading.Lock()
    _models: Dict[str, Any] = {}
    _vlm_provider: str = DEFAULT_VLM_PROVIDER
    
    def __new__(cls) -> 'ModelManager':
        """Ensure only one instance exists (thread-safe singleton pattern)."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the ModelManager (only called once)."""
        # Prevent re-initialization of singleton
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
    
    def get_florence2(self) -> Florence2:
        """
        Get Florence-2 model instance (lazy loaded).
        
        Returns:
            Florence2: The Florence-2 model instance
        """
        if 'florence2' not in self._models:
            print("Loading Florence-2 model...")
            self._models['florence2'] = Florence2()
            print("Florence-2 model loaded successfully.")
        return self._models['florence2']
    
    def get_llava(self) -> Llava:
        """
        Get LLaVA model instance (lazy loaded).
        
        Returns:
            Llava: The LLaVA model instance
            
        Deprecated: Use get_vlm() instead for VLM abstraction
        """
        if 'llava' not in self._models:
            print("Loading LLaVA model...")
            self._models['llava'] = Llava()
            print("LLaVA model loaded successfully.")
        return self._models['llava']
    
    def get_vlm(self) -> VLMInterface:
        """
        Get VLM model instance (lazy loaded, supports multiple providers).
        
        Returns:
            VLMInterface: The VLM model instance
        """
        vlm_key = f"vlm_{self._vlm_provider}"
        
        if vlm_key not in self._models:
            print(f"Loading {self._vlm_provider} VLM model...")
            self._models[vlm_key] = VLMRegistry.create_provider(self._vlm_provider)
            print(f"{self._vlm_provider} VLM model loaded successfully.")
        
        return self._models[vlm_key]
    
    def set_vlm_provider(self, provider: str) -> None:
        """
        Set the VLM provider (e.g., 'llava', 'gpt4v', 'claude').
        
        Args:
            provider: Name of the VLM provider
            
        Raises:
            ValueError: If provider is not registered
        """
        if provider not in VLMRegistry.list_providers():
            available = ", ".join(VLMRegistry.list_providers())
            raise ValueError(f"VLM provider '{provider}' not available. Available: {available}")
        
        # Clear existing VLM if different provider
        if provider != self._vlm_provider:
            old_key = f"vlm_{self._vlm_provider}"
            if old_key in self._models:
                del self._models[old_key]
                print(f"Cleared previous VLM provider: {self._vlm_provider}")
            
            self._vlm_provider = provider
            print(f"VLM provider set to: {provider}")
    
    def get_vlm_provider(self) -> str:
        """
        Get the current VLM provider name.
        
        Returns:
            str: Current VLM provider name
        """
        return self._vlm_provider
        
    def get_llm_client(self) -> LLMClient:
        """
        Get LLM client instance (lazy loaded).
        
        Returns:
            LLMClient: The LLM client instance
        """
        if 'llm_client' not in self._models:
            print("Initializing LLM client...")
            self._models['llm_client'] = LLMClient()
            print("LLM client initialized successfully.")
        return self._models['llm_client']
    
    def is_model_loaded(self, model_name: str) -> bool:
        """
        Check if a specific model is loaded.
        
        Args:
            model_name: Name of the model ('florence2', 'llava', 'llm_client')
            
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return model_name in self._models
    
    def get_loaded_models(self) -> list:
        """
        Get list of currently loaded model names.
        
        Returns:
            list: List of loaded model names
        """
        return list(self._models.keys())
    
    def get_memory_usage(self) -> Dict[str, str]:
        """
        Get GPU memory usage information.
        
        Returns:
            Dict[str, str]: Memory usage information
        """
        memory_info = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3  # Convert to GB
                reserved = torch.cuda.memory_reserved(i) / 1024**3    # Convert to GB
                memory_info[f'gpu_{i}'] = f"Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
        else:
            memory_info['gpu'] = "CUDA not available"
            
        return memory_info
    
    def cleanup(self):
        """
        Clean up GPU memory and model instances.
        Use with caution - will require reloading models.
        """
        print("Cleaning up ModelManager...")
        
        # Clear model references
        for model_name in list(self._models.keys()):
            print(f"Cleaning up {model_name}...")
            del self._models[model_name]
        
        self._models.clear()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared.")
        
        print("ModelManager cleanup completed.")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()
    
    def status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information.
        
        Returns:
            Dict[str, Any]: Status information including loaded models and memory usage
        """
        return {
            'loaded_models': self.get_loaded_models(),
            'memory_usage': self.get_memory_usage(),
            'cuda_available': torch.cuda.is_available(),
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }


# Global function to get ModelManager instance
def get_model_manager() -> ModelManager:
    """
    Global function to get the ModelManager singleton instance.
    
    Returns:
        ModelManager: The singleton instance
    """
    return ModelManager()


# Example usage and testing
if __name__ == "__main__":
    # Test singleton behavior
    manager1 = ModelManager()
    manager2 = ModelManager()
    
    print(f"Same instance: {manager1 is manager2}")  # Should be True
    print(f"Status: {manager1.status()}")
    
    # Test lazy loading (commented out to avoid loading models during testing)
    # florence2 = manager1.get_florence2()
    # llm_client = manager1.get_llm_client()
    
    print("ModelManager test completed successfully!")