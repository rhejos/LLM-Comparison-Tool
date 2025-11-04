"""OpenAI (ChatGPT) provider implementation."""

import time
from typing import Dict, Any
from openai import AsyncOpenAI, OpenAI

from .base_provider import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    """Provider for OpenAI models (GPT-3.5, GPT-4, etc.)."""

    def __init__(self, api_key: str, model_config: Dict[str, Any]):
        """Initialize OpenAI provider."""
        super().__init__(api_key, model_config)
        self.validate_config()
        self.client = AsyncOpenAI(api_key=api_key)
        self.sync_client = OpenAI(api_key=api_key)

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return "OpenAI"

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a response using OpenAI.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (system, temperature, etc.)

        Returns:
            LLMResponse object
        """
        start_time = time.time()

        try:
            # Prepare messages
            messages = []
            if 'system' in kwargs:
                messages.append({'role': 'system', 'content': kwargs['system']})
            messages.append({'role': 'user', 'content': prompt})

            # Prepare parameters
            params = {
                'model': self.model_id,
                'messages': messages,
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'temperature': kwargs.get('temperature', self.temperature),
            }

            # Make API call
            response = await self.client.chat.completions.create(**params)

            response_time = time.time() - start_time

            # Extract response text
            response_text = response.choices[0].message.content if response.choices else ""

            # Create LLMResponse object
            return LLMResponse(
                provider=self.get_provider_name(),
                model=self.display_name,
                prompt=prompt,
                response=response_text,
                input_tokens=response.usage.prompt_tokens if response.usage else None,
                output_tokens=response.usage.completion_tokens if response.usage else None,
                tokens_used=response.usage.total_tokens if response.usage else None,
                response_time=response_time,
                raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
            )

        except Exception as e:
            response_time = time.time() - start_time
            return LLMResponse(
                provider=self.get_provider_name(),
                model=self.display_name,
                prompt=prompt,
                response="",
                response_time=response_time,
                error=str(e)
            )
