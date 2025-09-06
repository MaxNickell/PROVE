"""
ModelManager singleton for managing all model instances across the pipeline.
Critical for memory efficiency - ensures only one instance of each model.
"""

import threading
from typing import Optional, Dict, Any
import torch

from src.vision.florence2 import Florence2
from src.vision.qwen_vl import QwenVL
from src.language.llm_client import LLMClient


class ModelManager:
    """Singleton class for managing all model instances across the pipeline."""
    
    _instance: Optional['ModelManager'] = None
    _lock = threading.Lock()
    _models: Dict[str, Any] = {}
    
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
    
    def get_qwen_vl(self) -> QwenVL:
        """
        Get Qwen VL model instance (lazy loaded).
        
        Returns:
            QwenVL: The Qwen 2.5-VL-7B model instance
        """
        if 'qwen_vl' not in self._models:
            print("Loading Qwen 2.5-VL-7B model...")
            self._models['qwen_vl'] = QwenVL()
            print("Qwen VL model loaded successfully.")
        return self._models['qwen_vl']
        
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
            model_name: Name of the model ('florence2', 'qwen_vl', 'llm_client')
            
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