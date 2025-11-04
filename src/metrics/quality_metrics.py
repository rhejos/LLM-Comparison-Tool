"""Quality metrics for evaluating LLM responses."""

import re
from typing import Dict, Any, List
import textstat
from dataclasses import dataclass


@dataclass
class QualityScore:
    """Quality score breakdown."""

    coherence: float  # 0-10
    relevance: float  # 0-10
    completeness: float  # 0-10
    clarity: float  # 0-10
    overall: float  # 0-10
    details: Dict[str, Any] = None


class QualityMetrics:
    """Evaluates the quality of LLM responses."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize quality metrics.

        Args:
            config: Configuration dictionary with weights
        """
        self.config = config or {}
        self.coherence_weight = self.config.get('coherence_weight', 0.3)
        self.relevance_weight = self.config.get('relevance_weight', 0.3)
        self.completeness_weight = self.config.get('completeness_weight', 0.2)
        self.clarity_weight = self.config.get('clarity_weight', 0.2)

    def evaluate(self, prompt: str, response: str) -> QualityScore:
        """
        Evaluate the quality of a response.

        Args:
            prompt: The input prompt
            response: The LLM response

        Returns:
            QualityScore object
        """
        coherence = self._evaluate_coherence(response)
        relevance = self._evaluate_relevance(prompt, response)
        completeness = self._evaluate_completeness(prompt, response)
        clarity = self._evaluate_clarity(response)

        # Calculate weighted overall score
        overall = (
            coherence * self.coherence_weight +
            relevance * self.relevance_weight +
            completeness * self.completeness_weight +
            clarity * self.clarity_weight
        )

        details = {
            'readability_score': textstat.flesch_reading_ease(response),
            'word_count': len(response.split()),
            'sentence_count': len(re.split(r'[.!?]+', response)),
            'avg_sentence_length': self._avg_sentence_length(response),
        }

        return QualityScore(
            coherence=round(coherence, 2),
            relevance=round(relevance, 2),
            completeness=round(completeness, 2),
            clarity=round(clarity, 2),
            overall=round(overall, 2),
            details=details
        )

    def _evaluate_coherence(self, response: str) -> float:
        """
        Evaluate how coherent and logical the response is.

        Args:
            response: The response text

        Returns:
            Score from 0-10
        """
        score = 10.0

        # Check for repetition
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        if len(sentences) > 1:
            unique_sentences = len(set(sentences))
            repetition_ratio = unique_sentences / len(sentences)
            if repetition_ratio < 0.8:
                score -= (1 - repetition_ratio) * 5

        # Check for contradictions (simple heuristic)
        contradiction_words = ['however', 'but', 'although', 'despite', 'nevertheless']
        contradiction_count = sum(response.lower().count(word) for word in contradiction_words)
        if contradiction_count > 3:
            score -= min(2, (contradiction_count - 3) * 0.5)

        # Check for logical flow indicators
        transition_words = ['first', 'second', 'third', 'finally', 'moreover', 'furthermore', 'therefore']
        has_transitions = any(word in response.lower() for word in transition_words)
        if has_transitions:
            score += 1

        return max(0, min(10, score))

    def _evaluate_relevance(self, prompt: str, response: str) -> float:
        """
        Evaluate how relevant the response is to the prompt.

        Args:
            prompt: The input prompt
            response: The response text

        Returns:
            Score from 0-10
        """
        score = 5.0  # Base score

        # Extract key words from prompt
        prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
        prompt_words = {w for w in prompt_words if len(w) > 3}  # Filter short words

        # Extract key words from response
        response_words = set(re.findall(r'\b\w+\b', response.lower()))

        # Calculate overlap
        if prompt_words:
            overlap = len(prompt_words & response_words) / len(prompt_words)
            score += overlap * 5

        # Check if response directly addresses the prompt
        question_words = ['what', 'why', 'how', 'when', 'where', 'who']
        prompt_lower = prompt.lower()

        for qword in question_words:
            if qword in prompt_lower:
                # Check if response attempts to answer
                if any(indicator in response.lower() for indicator in [qword, 'because', 'due to', 'result']):
                    score += 0.5

        return max(0, min(10, score))

    def _evaluate_completeness(self, prompt: str, response: str) -> float:
        """
        Evaluate how complete and thorough the response is.

        Args:
            prompt: The input prompt
            response: The response text

        Returns:
            Score from 0-10
        """
        score = 5.0

        # Length-based heuristic
        word_count = len(response.split())
        if word_count < 20:
            score -= 3
        elif word_count < 50:
            score -= 1
        elif word_count > 100:
            score += 2
        elif word_count > 200:
            score += 3

        # Check for examples
        if 'for example' in response.lower() or 'such as' in response.lower():
            score += 1

        # Check for multiple points/sections
        if response.count('\n\n') > 0 or response.count('•') > 0 or response.count('-') > 2:
            score += 1

        # Check if it's too verbose (penalize extremely long responses)
        if word_count > 1000:
            score -= 1

        return max(0, min(10, score))

    def _evaluate_clarity(self, response: str) -> float:
        """
        Evaluate how clear and easy to understand the response is.

        Args:
            response: The response text

        Returns:
            Score from 0-10
        """
        score = 5.0

        # Use readability score
        try:
            flesch_score = textstat.flesch_reading_ease(response)
            # Flesch scale: 90-100 = very easy, 0-30 = very difficult
            # Map to our 0-10 scale
            if flesch_score >= 60:  # Easy to read
                score += 3
            elif flesch_score >= 30:  # Moderate
                score += 1
            else:  # Difficult
                score -= 1
        except:
            pass

        # Check average sentence length
        avg_length = self._avg_sentence_length(response)
        if 10 <= avg_length <= 25:  # Ideal range
            score += 1
        elif avg_length > 40:  # Too long
            score -= 2

        # Check for jargon overuse (very long words)
        words = response.split()
        if words:
            long_words = [w for w in words if len(w) > 12]
            if len(long_words) / len(words) > 0.1:  # More than 10% long words
                score -= 1

        # Check for good structure
        if response.count('\n') > 0:  # Has paragraphs
            score += 1

        return max(0, min(10, score))

    def _avg_sentence_length(self, text: str) -> float:
        """Calculate average sentence length in words."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        if not sentences:
            return 0
        total_words = sum(len(s.split()) for s in sentences)
        return total_words / len(sentences)

    def compare_responses(self, responses: List[tuple]) -> Dict[str, QualityScore]:
        """
        Compare quality across multiple responses.

        Args:
            responses: List of (model_name, prompt, response) tuples

        Returns:
            Dictionary mapping model names to QualityScore objects
        """
        results = {}
        for model_name, prompt, response in responses:
            results[model_name] = self.evaluate(prompt, response)
        return results
