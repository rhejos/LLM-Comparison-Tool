"""Google Gemini provider implementation."""

import time
from typing import Dict, Any
import google.generativeai as genai

from .base_provider import BaseLLMProvider, LLMResponse


class GeminiProvider(BaseLLMProvider):
    """Provider for Google Gemini models."""

    def __init__(self, api_key: str, model_config: Dict[str, Any]):
        """Initialize Gemini provider."""
        super().__init__(api_key, model_config)
        self.validate_config()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_id)

    def get_provider_name(self) -> str:
        """Return the provider name."""
        return "Google Gemini"

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Generate a response using Gemini.

        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (system, temperature, etc.)

        Returns:
            LLMResponse object
        """
        start_time = time.time()

        try:
            # Prepare generation config
            generation_config = {
                'max_output_tokens': kwargs.get('max_tokens', self.max_tokens),
                'temperature': kwargs.get('temperature', self.temperature),
            }

            # Add system instruction if provided
            full_prompt = prompt
            if 'system' in kwargs:
                full_prompt = f"{kwargs['system']}\n\n{prompt}"

            # Make API call (Gemini doesn't have native async support, so we'll use sync)
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            response_time = time.time() - start_time

            # Extract response text
            response_text = response.text if hasattr(response, 'text') else ""

            # Estimate tokens (Gemini doesn't always provide this)
            tokens_used = None
            if hasattr(response, 'usage_metadata'):
                tokens_used = (
                    getattr(response.usage_metadata, 'total_token_count', None)
                )

            # Create LLMResponse object
            return LLMResponse(
                provider=self.get_provider_name(),
                model=self.display_name,
                prompt=prompt,
                response=response_text,
                tokens_used=tokens_used,
                response_time=response_time,
                raw_response=None  # Gemini responses are complex objects
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
