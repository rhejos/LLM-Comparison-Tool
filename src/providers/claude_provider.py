"""Claude (Anthropic) provider implementation."""

import time
from typing import Dict, Any
from anthropic import Anthropic, AsyncAnthropic

from .base_provider import BaseLLMProvider, LLMResponse


class ClaudeProvider(BaseLLMProvider):
    """Provider for Claude models from Anthropic."""

    def __init__(self, api_key: str, model_config: Dict[str, Any]):
        """Initialize Claude provider."""
        super().__init__(api_key, model_config)
        self.validate_config()
        self.client = AsyncAnthropic(api_key=api_key)
        self.sync_client = Anthropic(api_key=api_key)

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return "Claude (Anthropic)"

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a response using Claude.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (system, temperature, etc.)

        Returns:
            LLMResponse object
        """
        start_time = time.time()

        try:
            # Prepare parameters
            params = {
                'model': self.model_id,
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'temperature': kwargs.get('temperature', self.temperature),
                'messages': [
                    {'role': 'user', 'content': prompt}
                ]
            }

            # Add system message if provided
            if 'system' in kwargs:
                params['system'] = kwargs['system']

            # Make API call
            response = await self.client.messages.create(**params)

            response_time = time.time() - start_time

            # Extract response text
            response_text = response.content[0].text if response.content else ""

            # Create LLMResponse object
            return LLMResponse(
                provider=self.get_provider_name(),
                model=self.display_name,
                prompt=prompt,
                response=response_text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
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
