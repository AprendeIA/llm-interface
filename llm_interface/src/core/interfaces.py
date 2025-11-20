from abc import ABC, abstractmethod
from langchain_core.language_models import BaseLanguageModel

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def get_model(self) -> BaseLanguageModel:
        pass
    
    @abstractmethod
    def get_chat_model(self) -> BaseLanguageModel:
        pass
    
    @abstractmethod
    def get_embeddings(self):
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        pass
