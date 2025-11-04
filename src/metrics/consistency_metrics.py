"""Consistency metrics for evaluating LLM response stability."""

from typing import List, Dict, Any
from dataclasses import dataclass
import difflib
from collections import Counter
import re


@dataclass
class ConsistencyScore:
    """Consistency score breakdown."""

    similarity_score: float  # 0-10, how similar responses are
    variance_score: float  # 0-10, how much responses vary
    stability_score: float  # 0-10, overall consistency
    details: Dict[str, Any] = None


class ConsistencyMetrics:
    """Evaluates consistency of LLM responses across multiple runs."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize consistency metrics.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.similarity_threshold = self.config.get('similarity_threshold', 0.8)

    def evaluate(self, responses: List[str]) -> ConsistencyScore:
        """
        Evaluate consistency across multiple responses.

        Args:
            responses: List of response texts from the same prompt

        Returns:
            ConsistencyScore object
        """
        if len(responses) < 2:
            return ConsistencyScore(
                similarity_score=10.0,
                variance_score=10.0,
                stability_score=10.0,
                details={'message': 'Need at least 2 responses for consistency check'}
            )

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = self._calculate_similarity(responses[i], responses[j])
                similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities)
        min_similarity = min(similarities)
        max_similarity = max(similarities)

        # Calculate variance in response lengths
        lengths = [len(r.split()) for r in responses]
        avg_length = sum(lengths) / len(lengths)
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        length_std = length_variance ** 0.5

        # Normalize length variance to 0-10 scale (inverse)
        # Lower variance = higher score
        length_consistency = max(0, 10 - (length_std / avg_length * 20 if avg_length > 0 else 0))

        # Calculate key point consistency
        key_points_score = self._evaluate_key_points_consistency(responses)

        # Overall similarity score (0-10 scale)
        similarity_score = avg_similarity * 10

        # Variance score (combination of similarity variance and length variance)
        similarity_variance = max_similarity - min_similarity
        variance_score = max(0, 10 - (similarity_variance * 10))

        # Overall stability score
        stability_score = (
            similarity_score * 0.5 +
            variance_score * 0.3 +
            key_points_score * 0.2
        )

        details = {
            'avg_similarity': round(avg_similarity, 3),
            'min_similarity': round(min_similarity, 3),
            'max_similarity': round(max_similarity, 3),
            'avg_length': round(avg_length, 1),
            'length_std': round(length_std, 1),
            'length_consistency': round(length_consistency, 2),
            'key_points_score': round(key_points_score, 2),
            'num_comparisons': len(similarities)
        }

        return ConsistencyScore(
            similarity_score=round(similarity_score, 2),
            variance_score=round(variance_score, 2),
            stability_score=round(stability_score, 2),
            details=details
        )

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score from 0-1
        """
        # Use difflib's SequenceMatcher for character-level similarity
        seq_matcher = difflib.SequenceMatcher(None, text1, text2)
        char_similarity = seq_matcher.ratio()

        # Calculate word-level similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            word_similarity = 1.0
        elif not words1 or not words2:
            word_similarity = 0.0
        else:
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            word_similarity = intersection / union if union > 0 else 0

        # Combine both measures
        return (char_similarity * 0.6 + word_similarity * 0.4)

    def _evaluate_key_points_consistency(self, responses: List[str]) -> float:
        """
        Evaluate if key points are consistent across responses.

        Args:
            responses: List of response texts

        Returns:
            Score from 0-10
        """
        # Extract potential key points (sentences)
        all_sentences = []
        for response in responses:
            sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
            all_sentences.append(set(sentences))

        if not all_sentences:
            return 5.0

        # Find common sentences
        common_sentences = all_sentences[0]
        for sentence_set in all_sentences[1:]:
            common_sentences &= sentence_set

        # Calculate what percentage of sentences are common
        avg_sentence_count = sum(len(s) for s in all_sentences) / len(all_sentences)
        if avg_sentence_count == 0:
            return 5.0

        consistency_ratio = len(common_sentences) / avg_sentence_count

        # Also check for semantic similarity of unique sentences
        # (simplified: check if key words appear consistently)
        all_words = []
        for response in responses:
            words = [w.lower() for w in re.findall(r'\b\w+\b', response) if len(w) > 4]
            all_words.append(words)

        # Find words that appear in all responses
        word_counter = Counter()
        for words in all_words:
            word_counter.update(set(words))

        common_words = [w for w, count in word_counter.items() if count == len(responses)]
        avg_word_count = sum(len(w) for w in all_words) / len(all_words)

        if avg_word_count == 0:
            word_consistency = 0
        else:
            word_consistency = len(common_words) / avg_word_count

        # Combine scores
        score = (consistency_ratio * 5 + word_consistency * 10) / 1.5
        return min(10, max(0, score))

    def compare_models(self, model_responses: Dict[str, List[str]]) -> Dict[str, ConsistencyScore]:
        """
        Compare consistency across multiple models.

        Args:
            model_responses: Dictionary mapping model names to lists of responses

        Returns:
            Dictionary mapping model names to ConsistencyScore objects
        """
        results = {}
        for model_name, responses in model_responses.items():
            results[model_name] = self.evaluate(responses)
        return results
