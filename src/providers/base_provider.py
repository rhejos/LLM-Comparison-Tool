"""Base provider interface for LLM integrations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime


@dataclass
class LLMResponse:
    """Standardized response from an LLM provider."""

    provider: str
    model: str
    prompt: str
    response: str
    tokens_used: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    response_time: Optional[float] = None
    timestamp: datetime = None
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            'provider': self.provider,
            'model': self.model,
            'prompt': self.prompt,
            'response': self.response,
            'tokens_used': self.tokens_used,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'response_time': self.response_time,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'error': self.error
        }


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, api_key: str, model_config: Dict[str, Any]):
        """
        Initialize the provider.

        Args:
            api_key: API key for the provider
            model_config: Model configuration dictionary
        """
        self.api_key = api_key
        self.model_config = model_config
        self.model_id = model_config.get('model_id')
        self.max_tokens = model_config.get('max_tokens', 4096)
        self.temperature = model_config.get('temperature', 1.0)
        self.display_name = model_config.get('display_name', 'Unknown Model')

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a response from the LLM.

        Args:
            prompt: The input prompt
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse object containing the response and metadata
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the name of the provider."""
        pass

    def validate_config(self) -> bool:
        """Validate the provider configuration."""
        if not self.api_key:
            raise ValueError(f"API key is required for {self.get_provider_name()}")
        if not self.model_id:
            raise ValueError(f"Model ID is required for {self.get_provider_name()}")
        return True
