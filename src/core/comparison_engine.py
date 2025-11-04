"""Core comparison engine for LLM evaluation."""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json
import time

from ..providers import BaseLLMProvider, LLMResponse
from ..metrics import QualityMetrics, ConsistencyMetrics, PerformanceMetrics
from ..analyzers import BiasDetector, DiscrepancyAnalyzer, TopicAnalyzer, HallucinationDetector


@dataclass
class ComparisonResult:
    """Results from comparing LLMs."""

    prompt: str
    model_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    rankings: Dict[str, List[tuple]] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'prompt': self.prompt,
            'model_results': self.model_results,
            'rankings': self.rankings,
            'summary': self.summary,
            'metadata': self.metadata
        }


class ComparisonEngine:
    """Main engine for comparing LLM responses."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the comparison engine.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.providers: Dict[str, BaseLLMProvider] = {}

        # Initialize metrics
        self.quality_metrics = QualityMetrics(config.get('metrics', {}).get('quality'))
        self.consistency_metrics = ConsistencyMetrics(config.get('metrics', {}).get('consistency'))
        self.performance_metrics = PerformanceMetrics()

        # Initialize analyzers
        self.bias_detector = BiasDetector(config.get('analysis', {}).get('bias_detection'))
        self.discrepancy_analyzer = DiscrepancyAnalyzer(config.get('analysis', {}).get('discrepancy_detection'))
        self.topic_analyzer = TopicAnalyzer(config.get('analysis', {}).get('topic_analysis'))
        self.hallucination_detector = HallucinationDetector(config.get('analysis', {}).get('hallucination_detection'))

    def register_provider(self, name: str, provider: BaseLLMProvider):
        """
        Register an LLM provider.

        Args:
            name: Name for the provider
            provider: Provider instance
        """
        self.providers[name] = provider

    async def compare_single_prompt(
        self,
        prompt: str,
        model_names: Optional[List[str]] = None,
        analyze_bias: bool = True,
        analyze_discrepancies: bool = True,
        analyze_topics: bool = True,
        analyze_hallucinations: bool = True,
        consistency_runs: int = 1
    ) -> ComparisonResult:
        """
        Compare models on a single prompt.

        Args:
            prompt: The prompt to test
            model_names: List of model names to compare (None = all)
            analyze_bias: Whether to perform bias analysis
            analyze_discrepancies: Whether to perform discrepancy analysis
            analyze_topics: Whether to perform topic analysis
            analyze_hallucinations: Whether to perform hallucination analysis
            consistency_runs: Number of runs for consistency testing

        Returns:
            ComparisonResult object
        """
        if model_names is None:
            model_names = list(self.providers.keys())

        model_results = {}

        # Run comparisons for each model
        for model_name in model_names:
            if model_name not in self.providers:
                print(f"Warning: Provider '{model_name}' not registered, skipping...")
                continue

            print(f"Testing {model_name}...")
            provider = self.providers[model_name]

            # Generate multiple responses for consistency testing
            responses = []
            response_times = []
            token_counts = []

            for i in range(consistency_runs):
                try:
                    response = await provider.generate(prompt)
                    responses.append(response)

                    if response.response_time:
                        response_times.append(response.response_time)
                    if response.tokens_used:
                        token_counts.append(response.tokens_used)

                except Exception as e:
                    print(f"Error generating response from {model_name}: {e}")
                    continue

            if not responses:
                print(f"No successful responses from {model_name}, skipping...")
                continue

            # Use the first response for most analyses
            primary_response = responses[0]
            response_texts = [r.response for r in responses if r.response]

            if not response_texts:
                continue

            # Perform analyses
            result = {
                'responses': [r.to_dict() for r in responses],
                'primary_response': primary_response.response
            }

            # Quality metrics
            quality = self.quality_metrics.evaluate(prompt, primary_response.response)
            result['quality'] = {
                'coherence': quality.coherence,
                'relevance': quality.relevance,
                'completeness': quality.completeness,
                'clarity': quality.clarity,
                'overall': quality.overall,
                'details': quality.details
            }

            # Consistency metrics (if multiple runs)
            if len(response_texts) > 1:
                consistency = self.consistency_metrics.evaluate(response_texts)
                result['consistency'] = {
                    'similarity_score': consistency.similarity_score,
                    'variance_score': consistency.variance_score,
                    'stability_score': consistency.stability_score,
                    'details': consistency.details
                }

            # Performance metrics
            if response_times:
                performance = self.performance_metrics.evaluate(response_times, token_counts if token_counts else None)
                result['performance'] = {
                    'avg_response_time': performance.avg_response_time,
                    'response_time_std': performance.response_time_std,
                    'tokens_per_second': performance.tokens_per_second,
                    'efficiency_score': performance.efficiency_score,
                    'details': performance.details
                }

            # Bias analysis
            if analyze_bias:
                bias = self.bias_detector.analyze(prompt, primary_response.response)
                result['bias'] = {
                    'overall_score': bias.overall_bias_score,
                    'categories': bias.bias_categories,
                    'detected_biases': bias.detected_biases,
                    'recommendations': bias.recommendations,
                    'details': bias.details
                }

            # Discrepancy analysis
            if analyze_discrepancies and len(response_texts) > 1:
                discrepancy = self.discrepancy_analyzer.analyze(prompt, response_texts)
                result['discrepancy'] = {
                    'consistency_score': discrepancy.consistency_score,
                    'contradictions': discrepancy.factual_contradictions,
                    'logical_issues': discrepancy.logical_inconsistencies,
                    'variations': discrepancy.response_variations,
                    'details': discrepancy.details
                }

            # Topic analysis
            if analyze_topics:
                topic = self.topic_analyzer.analyze(prompt, primary_response.response, quality.overall)
                result['topic'] = {
                    'best_topics': topic.best_topics,
                    'topic_scores': topic.topic_scores,
                    'recommendations': topic.recommendations,
                    'characteristics': topic.response_characteristics
                }

            # Hallucination analysis
            if analyze_hallucinations:
                hallucination = self.hallucination_detector.analyze(prompt, primary_response.response)
                result['hallucination'] = {
                    'confidence_score': hallucination.confidence_score,
                    'potential_hallucinations': hallucination.potential_hallucinations,
                    'uncertainty_markers': hallucination.uncertainty_markers,
                    'details': hallucination.details
                }

            model_results[model_name] = result

        # Calculate rankings
        rankings = self._calculate_rankings(model_results)

        # Generate summary
        summary = self._generate_summary(model_results, rankings)

        return ComparisonResult(
            prompt=prompt,
            model_results=model_results,
            rankings=rankings,
            summary=summary,
            metadata={
                'timestamp': time.time(),
                'num_models': len(model_results),
                'consistency_runs': consistency_runs
            }
        )

    def _calculate_rankings(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, List[tuple]]:
        """Calculate rankings for different metrics."""
        rankings = {}

        # Quality ranking
        quality_scores = [(name, result['quality']['overall'])
                         for name, result in model_results.items()
                         if 'quality' in result]
        rankings['quality'] = sorted(quality_scores, key=lambda x: x[1], reverse=True)

        # Bias ranking (higher is better - less biased)
        bias_scores = [(name, result['bias']['overall_score'])
                      for name, result in model_results.items()
                      if 'bias' in result]
        rankings['bias'] = sorted(bias_scores, key=lambda x: x[1], reverse=True)

        # Performance ranking
        perf_scores = [(name, result['performance']['efficiency_score'])
                      for name, result in model_results.items()
                      if 'performance' in result]
        rankings['performance'] = sorted(perf_scores, key=lambda x: x[1], reverse=True)

        # Hallucination ranking (higher confidence = better)
        hall_scores = [(name, result['hallucination']['confidence_score'])
                      for name, result in model_results.items()
                      if 'hallucination' in result]
        rankings['hallucination'] = sorted(hall_scores, key=lambda x: x[1], reverse=True)

        # Overall ranking (composite score)
        overall_scores = []
        for name, result in model_results.items():
            score = 0
            count = 0

            if 'quality' in result:
                score += result['quality']['overall']
                count += 1
            if 'bias' in result:
                score += result['bias']['overall_score']
                count += 1
            if 'performance' in result:
                score += result['performance']['efficiency_score']
                count += 1
            if 'hallucination' in result:
                score += result['hallucination']['confidence_score']
                count += 1

            if count > 0:
                overall_scores.append((name, score / count))

        rankings['overall'] = sorted(overall_scores, key=lambda x: x[1], reverse=True)

        return rankings

    def _generate_summary(
        self,
        model_results: Dict[str, Dict[str, Any]],
        rankings: Dict[str, List[tuple]]
    ) -> Dict[str, Any]:
        """Generate a summary of the comparison."""
        summary = {
            'winner': rankings['overall'][0][0] if rankings.get('overall') else None,
            'best_quality': rankings['quality'][0][0] if rankings.get('quality') else None,
            'least_biased': rankings['bias'][0][0] if rankings.get('bias') else None,
            'fastest': rankings['performance'][0][0] if rankings.get('performance') else None,
            'most_factual': rankings['hallucination'][0][0] if rankings.get('hallucination') else None,
        }

        # Add score comparisons
        if rankings.get('overall') and len(rankings['overall']) > 1:
            winner_score = rankings['overall'][0][1]
            runner_up_score = rankings['overall'][1][1]
            summary['score_margin'] = round(winner_score - runner_up_score, 2)

        return summary

    async def compare_multiple_prompts(
        self,
        prompts: List[str],
        model_names: Optional[List[str]] = None,
        **kwargs
    ) -> List[ComparisonResult]:
        """
        Compare models across multiple prompts.

        Args:
            prompts: List of prompts to test
            model_names: List of model names to compare
            **kwargs: Additional arguments for compare_single_prompt

        Returns:
            List of ComparisonResult objects
        """
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\n{'='*60}")
            print(f"Prompt {i+1}/{len(prompts)}: {prompt[:100]}...")
            print(f"{'='*60}\n")

            result = await self.compare_single_prompt(prompt, model_names, **kwargs)
            results.append(result)

        return results

    def save_results(self, results: ComparisonResult, filepath: str):
        """Save comparison results to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)

    def load_results(self, filepath: str) -> ComparisonResult:
        """Load comparison results from a JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        return ComparisonResult(
            prompt=data['prompt'],
            model_results=data['model_results'],
            rankings=data['rankings'],
            summary=data['summary'],
            metadata=data.get('metadata', {})
        )
