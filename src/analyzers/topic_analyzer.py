"""Topic and use-case suitability analyzer for LLMs."""

import re
from typing import Dict, List, Any
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class TopicSuitability:
    """Topic suitability analysis results."""

    best_topics: List[str] = field(default_factory=list)
    topic_scores: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    response_characteristics: Dict[str, Any] = field(default_factory=dict)


class TopicAnalyzer:
    """Analyzes LLM responses to determine topic suitability."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize topic analyzer.

        Args:
            config: Configuration dictionary with topic categories
        """
        self.config = config or {}
        self.categories = self.config.get('categories', [
            'creative_writing',
            'technical_explanation',
            'factual_research',
            'code_generation',
            'mathematical_reasoning',
            'conversational',
            'analytical'
        ])

        # Topic indicators
        self.topic_indicators = {
            'creative_writing': {
                'keywords': [
                    'story', 'narrative', 'character', 'plot', 'scene', 'dialogue',
                    'metaphor', 'imagery', 'creative', 'imaginative', 'fiction'
                ],
                'patterns': [
                    r'once upon', r'he said', r'she replied', r'it was',
                    r'in the', r'the story', r'the narrative'
                ],
                'style_markers': ['descriptive', 'vivid', 'emotional', 'expressive']
            },
            'technical_explanation': {
                'keywords': [
                    'system', 'process', 'mechanism', 'component', 'function',
                    'technical', 'architecture', 'implementation', 'protocol',
                    'algorithm', 'method', 'procedure', 'operation'
                ],
                'patterns': [
                    r'how\s+\w+\s+works?', r'the\s+system', r'this\s+process',
                    r'step\s+\d+', r'first,', r'second,', r'finally'
                ],
                'style_markers': ['precise', 'structured', 'detailed', 'systematic']
            },
            'factual_research': {
                'keywords': [
                    'research', 'study', 'evidence', 'data', 'fact', 'statistic',
                    'according to', 'source', 'reference', 'published', 'scholar',
                    'finding', 'result', 'conclusion', 'analysis'
                ],
                'patterns': [
                    r'\d+%', r'in\s+\d{4}', r'studies? show', r'research indicates',
                    r'according to', r'based on', r'evidence suggests'
                ],
                'style_markers': ['factual', 'referenced', 'objective', 'verified']
            },
            'code_generation': {
                'keywords': [
                    'function', 'class', 'variable', 'method', 'code', 'syntax',
                    'program', 'script', 'algorithm', 'debug', 'compile', 'execute'
                ],
                'patterns': [
                    r'```', r'def\s+\w+', r'class\s+\w+', r'function\s+\w+',
                    r'import\s+\w+', r'return\s+\w+', r'if\s+.*:', r'for\s+.*:'
                ],
                'style_markers': ['code_blocks', 'syntax_highlighted', 'executable']
            },
            'mathematical_reasoning': {
                'keywords': [
                    'equation', 'formula', 'calculate', 'solve', 'proof', 'theorem',
                    'mathematical', 'algebra', 'geometry', 'calculus', 'variable'
                ],
                'patterns': [
                    r'\d+\s*[+\-*/]\s*\d+', r'=\s*\d+', r'x\s*=', r'y\s*=',
                    r'∫', r'∑', r'√', r'therefore', r'thus', r'hence'
                ],
                'style_markers': ['mathematical', 'logical', 'deductive', 'precise']
            },
            'conversational': {
                'keywords': [
                    'hello', 'hi', 'thanks', 'please', 'you', 'I', 'we', 'let me',
                    'sure', 'of course', 'certainly', 'absolutely', 'help'
                ],
                'patterns': [
                    r'how can I', r'I\'d be happy', r'let me help', r'I think',
                    r'in my opinion', r'you might', r'you could', r'would you'
                ],
                'style_markers': ['friendly', 'personal', 'engaging', 'helpful']
            },
            'analytical': {
                'keywords': [
                    'analyze', 'analysis', 'compare', 'contrast', 'evaluate',
                    'assess', 'examine', 'consider', 'implication', 'factor',
                    'aspect', 'perspective', 'viewpoint', 'approach'
                ],
                'patterns': [
                    r'on one hand', r'on the other hand', r'however', r'nevertheless',
                    r'in contrast', r'compared to', r'analysis shows', r'factors include'
                ],
                'style_markers': ['balanced', 'nuanced', 'comprehensive', 'critical']
            }
        }

    def analyze(self, prompt: str, response: str, quality_score: float = None) -> TopicSuitability:
        """
        Analyze topic suitability of a response.

        Args:
            prompt: The input prompt
            response: The LLM response
            quality_score: Optional quality score to factor in

        Returns:
            TopicSuitability object
        """
        # Score each topic category
        topic_scores = {}
        for category in self.categories:
            score = self._score_topic_category(category, prompt, response)
            topic_scores[category] = score

        # Identify best topics (score > 6.0)
        best_topics = [
            topic for topic, score in sorted(
                topic_scores.items(), key=lambda x: x[1], reverse=True
            ) if score > 6.0
        ][:3]  # Top 3

        # Generate recommendations
        recommendations = self._generate_recommendations(topic_scores, prompt, response)

        # Analyze response characteristics
        characteristics = self._analyze_characteristics(response)

        return TopicSuitability(
            best_topics=best_topics,
            topic_scores={k: round(v, 2) for k, v in topic_scores.items()},
            recommendations=recommendations,
            response_characteristics=characteristics
        )

    def _score_topic_category(self, category: str, prompt: str, response: str) -> float:
        """
        Score how well a response fits a topic category.

        Args:
            category: Topic category to score
            prompt: The input prompt
            response: The response text

        Returns:
            Score from 0-10
        """
        if category not in self.topic_indicators:
            return 5.0

        score = 0.0
        indicators = self.topic_indicators[category]
        response_lower = response.lower()
        prompt_lower = prompt.lower()

        # Check keywords in response
        keyword_count = sum(
            response_lower.count(keyword) for keyword in indicators['keywords']
        )
        keyword_score = min(5.0, keyword_count * 0.5)
        score += keyword_score

        # Check patterns
        pattern_count = sum(
            len(re.findall(pattern, response_lower)) for pattern in indicators['patterns']
        )
        pattern_score = min(3.0, pattern_count * 0.5)
        score += pattern_score

        # Check prompt relevance
        prompt_keywords = sum(
            prompt_lower.count(keyword) for keyword in indicators['keywords']
        )
        if prompt_keywords > 0:
            score += 1.0

        # Special handling for code generation
        if category == 'code_generation':
            if '```' in response:
                score += 3.0
            if any(lang in response_lower for lang in ['python', 'java', 'javascript', 'c++', 'ruby']):
                score += 1.0

        # Special handling for mathematical reasoning
        if category == 'mathematical_reasoning':
            # Count mathematical operators
            math_operators = len(re.findall(r'[+\-*/=<>≤≥∫∑√]', response))
            score += min(2.0, math_operators * 0.3)

        return min(10.0, score)

    def _analyze_characteristics(self, response: str) -> Dict[str, Any]:
        """Analyze general response characteristics."""
        characteristics = {
            'length': len(response),
            'word_count': len(response.split()),
            'sentence_count': len(re.split(r'[.!?]+', response)),
            'has_code_blocks': '```' in response,
            'has_lists': bool(re.search(r'^\s*[-*]\s', response, re.MULTILINE)),
            'has_numbers': bool(re.search(r'\d+', response)),
            'has_questions': '?' in response,
            'paragraph_count': len(response.split('\n\n'))
        }

        # Determine dominant style
        styles = []
        if characteristics['has_code_blocks']:
            styles.append('technical')
        if characteristics['has_lists']:
            styles.append('structured')
        if characteristics['has_questions']:
            styles.append('interactive')
        if characteristics['paragraph_count'] > 3:
            styles.append('detailed')

        characteristics['dominant_styles'] = styles

        return characteristics

    def _generate_recommendations(
        self,
        topic_scores: Dict[str, float],
        prompt: str,
        response: str
    ) -> List[str]:
        """Generate recommendations based on topic analysis."""
        recommendations = []

        # Find top scoring topics
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        top_topic, top_score = sorted_topics[0] if sorted_topics else (None, 0)

        if top_score > 7.0:
            recommendations.append(
                f"Excellent fit for {top_topic.replace('_', ' ')} tasks (score: {top_score:.1f}/10)"
            )
        elif top_score > 5.0:
            recommendations.append(
                f"Good fit for {top_topic.replace('_', ' ')} tasks (score: {top_score:.1f}/10)"
            )

        # Identify weak areas
        weak_topics = [topic for topic, score in topic_scores.items() if score < 4.0]
        if weak_topics:
            recommendations.append(
                f"Less suitable for: {', '.join(t.replace('_', ' ') for t in weak_topics[:2])}"
            )

        # Specific recommendations
        if topic_scores.get('code_generation', 0) > 7.0:
            recommendations.append("Strong code generation capabilities detected")

        if topic_scores.get('factual_research', 0) > 7.0:
            recommendations.append("Good at providing factual, researched content")

        if topic_scores.get('creative_writing', 0) > 7.0:
            recommendations.append("Shows creativity in narrative and descriptive writing")

        if topic_scores.get('analytical', 0) > 7.0:
            recommendations.append("Excels at analytical and comparative tasks")

        return recommendations

    def compare_models(
        self,
        model_responses: Dict[str, tuple]
    ) -> Dict[str, TopicSuitability]:
        """
        Compare topic suitability across multiple models.

        Args:
            model_responses: Dictionary mapping model names to (prompt, response) tuples

        Returns:
            Dictionary mapping model names to TopicSuitability objects
        """
        results = {}
        for model_name, (prompt, response) in model_responses.items():
            results[model_name] = self.analyze(prompt, response)
        return results

    def recommend_model_for_topic(
        self,
        topic: str,
        model_analyses: Dict[str, TopicSuitability]
    ) -> List[tuple]:
        """
        Recommend models for a specific topic.

        Args:
            topic: Topic category
            model_analyses: Dictionary mapping model names to TopicSuitability objects

        Returns:
            List of (model_name, score) tuples, sorted by score descending
        """
        recommendations = []
        for model_name, analysis in model_analyses.items():
            score = analysis.topic_scores.get(topic, 0)
            recommendations.append((model_name, score))

        return sorted(recommendations, key=lambda x: x[1], reverse=True)
