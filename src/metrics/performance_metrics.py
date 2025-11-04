"""Performance metrics for evaluating LLM speed and efficiency."""

from typing import List, Dict, Any
from dataclasses import dataclass
import statistics


@dataclass
class PerformanceScore:
    """Performance score breakdown."""

    avg_response_time: float  # Average time in seconds
    response_time_std: float  # Standard deviation
    tokens_per_second: float  # Average tokens generated per second
    efficiency_score: float  # 0-10, overall efficiency
    details: Dict[str, Any] = None


class PerformanceMetrics:
    """Evaluates performance characteristics of LLM responses."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize performance metrics.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

    def evaluate(
        self,
        response_times: List[float],
        token_counts: List[int] = None
    ) -> PerformanceScore:
        """
        Evaluate performance metrics.

        Args:
            response_times: List of response times in seconds
            token_counts: List of token counts (optional)

        Returns:
            PerformanceScore object
        """
        if not response_times:
            return PerformanceScore(
                avg_response_time=0,
                response_time_std=0,
                tokens_per_second=0,
                efficiency_score=0,
                details={'message': 'No response times provided'}
            )

        avg_response_time = statistics.mean(response_times)
        response_time_std = statistics.stdev(response_times) if len(response_times) > 1 else 0

        # Calculate tokens per second
        tokens_per_second = 0
        if token_counts and len(token_counts) == len(response_times):
            rates = [
                tokens / time if time > 0 else 0
                for tokens, time in zip(token_counts, response_times)
            ]
            tokens_per_second = statistics.mean(rates) if rates else 0

        # Calculate efficiency score (0-10 scale)
        # Lower response time = higher score
        # Baseline: 1 second = 10 points, 10 seconds = 1 point
        time_score = max(0, min(10, 11 - avg_response_time))

        # Consistency bonus (lower std = more consistent = higher score)
        if avg_response_time > 0:
            consistency_ratio = response_time_std / avg_response_time
            consistency_score = max(0, 10 - (consistency_ratio * 10))
        else:
            consistency_score = 5

        # Throughput score (tokens per second)
        throughput_score = min(10, tokens_per_second / 10) if tokens_per_second > 0 else 5

        # Overall efficiency score
        efficiency_score = (
            time_score * 0.5 +
            consistency_score * 0.25 +
            throughput_score * 0.25
        )

        details = {
            'min_response_time': min(response_times),
            'max_response_time': max(response_times),
            'median_response_time': statistics.median(response_times),
            'time_score': round(time_score, 2),
            'consistency_score': round(consistency_score, 2),
            'throughput_score': round(throughput_score, 2),
            'num_samples': len(response_times)
        }

        if token_counts:
            details['total_tokens'] = sum(token_counts)
            details['avg_tokens'] = statistics.mean(token_counts)

        return PerformanceScore(
            avg_response_time=round(avg_response_time, 3),
            response_time_std=round(response_time_std, 3),
            tokens_per_second=round(tokens_per_second, 1),
            efficiency_score=round(efficiency_score, 2),
            details=details
        )

    def compare_models(
        self,
        model_data: Dict[str, Dict[str, List]]
    ) -> Dict[str, PerformanceScore]:
        """
        Compare performance across multiple models.

        Args:
            model_data: Dictionary mapping model names to
                       {'response_times': [...], 'token_counts': [...]}

        Returns:
            Dictionary mapping model names to PerformanceScore objects
        """
        results = {}
        for model_name, data in model_data.items():
            results[model_name] = self.evaluate(
                response_times=data.get('response_times', []),
                token_counts=data.get('token_counts')
            )
        return results

    def calculate_cost_efficiency(
        self,
        token_counts: List[int],
        cost_per_1k_tokens: float
    ) -> Dict[str, Any]:
        """
        Calculate cost efficiency metrics.

        Args:
            token_counts: List of token counts
            cost_per_1k_tokens: Cost per 1000 tokens

        Returns:
            Dictionary with cost metrics
        """
        if not token_counts:
            return {'total_cost': 0, 'avg_cost_per_request': 0}

        total_tokens = sum(token_counts)
        total_cost = (total_tokens / 1000) * cost_per_1k_tokens
        avg_cost = total_cost / len(token_counts)

        return {
            'total_tokens': total_tokens,
            'total_cost': round(total_cost, 4),
            'avg_cost_per_request': round(avg_cost, 4),
            'cost_per_1k_tokens': cost_per_1k_tokens
        }
