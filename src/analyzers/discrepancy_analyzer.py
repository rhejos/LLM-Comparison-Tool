"""Discrepancy detection and analysis for LLM responses."""

import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
import difflib


@dataclass
class DiscrepancyAnalysis:
    """Discrepancy analysis results."""

    consistency_score: float  # 0-10, higher = more consistent
    factual_contradictions: List[Dict[str, Any]] = field(default_factory=list)
    logical_inconsistencies: List[Dict[str, Any]] = field(default_factory=list)
    response_variations: List[Dict[str, Any]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class DiscrepancyAnalyzer:
    """Analyzes discrepancies and inconsistencies in LLM responses."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize discrepancy analyzer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.similarity_threshold = self.config.get('similarity_threshold', 0.85)

        # Contradiction indicators
        self.contradiction_patterns = [
            (r'is\s+(\w+)', r'is\s+not\s+(\w+)'),
            (r'can\s+(\w+)', r'cannot\s+(\w+)'),
            (r'will\s+(\w+)', r'will\s+not\s+(\w+)'),
            (r'does\s+(\w+)', r'does\s+not\s+(\w+)'),
            (r'(\w+)\s+is\s+true', r'(\w+)\s+is\s+false'),
            (r'always', r'never'),
            (r'all', r'none'),
        ]

    def analyze(self, prompt: str, responses: List[str]) -> DiscrepancyAnalysis:
        """
        Analyze discrepancies across multiple responses to the same prompt.

        Args:
            prompt: The input prompt
            responses: List of response texts

        Returns:
            DiscrepancyAnalysis object
        """
        if len(responses) < 2:
            return DiscrepancyAnalysis(
                consistency_score=10.0,
                details={'message': 'Need at least 2 responses to detect discrepancies'}
            )

        # Detect factual contradictions
        contradictions = self._detect_contradictions(responses)

        # Detect logical inconsistencies within each response
        logical_issues = []
        for i, response in enumerate(responses):
            issues = self._detect_logical_inconsistencies(response)
            for issue in issues:
                issue['response_index'] = i
                logical_issues.append(issue)

        # Analyze variations in key information
        variations = self._analyze_variations(responses)

        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(
            contradictions, logical_issues, variations, len(responses)
        )

        details = {
            'num_responses': len(responses),
            'num_contradictions': len(contradictions),
            'num_logical_issues': len(logical_issues),
            'num_variations': len(variations),
            'avg_response_similarity': self._calculate_avg_similarity(responses)
        }

        return DiscrepancyAnalysis(
            consistency_score=round(consistency_score, 2),
            factual_contradictions=contradictions,
            logical_inconsistencies=logical_issues,
            response_variations=variations,
            details=details
        )

    def _detect_contradictions(self, responses: List[str]) -> List[Dict[str, Any]]:
        """
        Detect contradictions between responses.

        Args:
            responses: List of response texts

        Returns:
            List of detected contradictions
        """
        contradictions = []

        # Extract factual statements from each response
        statements = []
        for i, response in enumerate(responses):
            facts = self._extract_factual_statements(response)
            statements.append((i, facts))

        # Compare statements across responses
        for i in range(len(statements)):
            for j in range(i + 1, len(statements)):
                idx1, facts1 = statements[i]
                idx2, facts2 = statements[j]

                for fact1 in facts1:
                    for fact2 in facts2:
                        if self._are_contradictory(fact1, fact2):
                            contradictions.append({
                                'type': 'factual_contradiction',
                                'response_indices': [idx1, idx2],
                                'statement_1': fact1,
                                'statement_2': fact2,
                                'severity': 'high'
                            })

        return contradictions

    def _extract_factual_statements(self, text: str) -> List[str]:
        """
        Extract potential factual statements from text.

        Args:
            text: Response text

        Returns:
            List of factual statements
        """
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        # Filter for factual-looking statements
        factual = []
        factual_indicators = [
            r'\bis\b', r'\bare\b', r'\bwas\b', r'\bwere\b',
            r'\bhas\b', r'\bhave\b', r'\bcan\b', r'\bcannot\b',
            r'\bwill\b', r'\bwould\b', r'\bmust\b', r'\bshould\b',
            r'\d+', r'percent', r'percentage', r'number', r'amount'
        ]

        for sentence in sentences:
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in factual_indicators):
                factual.append(sentence)

        return factual

    def _are_contradictory(self, statement1: str, statement2: str) -> bool:
        """
        Check if two statements are contradictory.

        Args:
            statement1: First statement
            statement2: Second statement

        Returns:
            True if statements are contradictory
        """
        s1_lower = statement1.lower()
        s2_lower = statement2.lower()

        # Check for direct contradiction patterns
        for pos_pattern, neg_pattern in self.contradiction_patterns:
            pos_match1 = re.search(pos_pattern, s1_lower)
            neg_match2 = re.search(neg_pattern, s2_lower)

            if pos_match1 and neg_match2:
                # Check if they're talking about similar things
                similarity = difflib.SequenceMatcher(None, s1_lower, s2_lower).ratio()
                if similarity > 0.5:  # Similar context
                    return True

            # Check reverse
            neg_match1 = re.search(neg_pattern, s1_lower)
            pos_match2 = re.search(pos_pattern, s2_lower)

            if neg_match1 and pos_match2:
                similarity = difflib.SequenceMatcher(None, s1_lower, s2_lower).ratio()
                if similarity > 0.5:
                    return True

        # Check for numerical contradictions
        numbers1 = re.findall(r'\d+(?:\.\d+)?', statement1)
        numbers2 = re.findall(r'\d+(?:\.\d+)?', statement2)

        if numbers1 and numbers2:
            # If talking about same thing but different numbers
            # Remove numbers and check similarity
            s1_no_nums = re.sub(r'\d+(?:\.\d+)?', 'NUM', s1_lower)
            s2_no_nums = re.sub(r'\d+(?:\.\d+)?', 'NUM', s2_lower)

            if difflib.SequenceMatcher(None, s1_no_nums, s2_no_nums).ratio() > 0.7:
                # Same context, different numbers
                if numbers1[0] != numbers2[0]:
                    return True

        return False

    def _detect_logical_inconsistencies(self, response: str) -> List[Dict[str, Any]]:
        """
        Detect logical inconsistencies within a single response.

        Args:
            response: Response text

        Returns:
            List of detected inconsistencies
        """
        inconsistencies = []

        # Extract all statements
        statements = self._extract_factual_statements(response)

        # Check for internal contradictions
        for i in range(len(statements)):
            for j in range(i + 1, len(statements)):
                if self._are_contradictory(statements[i], statements[j]):
                    inconsistencies.append({
                        'type': 'internal_contradiction',
                        'statement_1': statements[i],
                        'statement_2': statements[j],
                        'severity': 'high'
                    })

        # Check for circular reasoning
        circular = self._detect_circular_reasoning(response)
        if circular:
            inconsistencies.append({
                'type': 'circular_reasoning',
                'description': circular,
                'severity': 'medium'
            })

        return inconsistencies

    def _detect_circular_reasoning(self, text: str) -> str:
        """
        Detect potential circular reasoning.

        Args:
            text: Response text

        Returns:
            Description of circular reasoning if detected, empty string otherwise
        """
        # Simple heuristic: look for repeated concepts in justifications
        justification_patterns = [
            r'because\s+(.+?)[.!?]',
            r'since\s+(.+?)[.!?]',
            r'as\s+(.+?)[.!?]',
            r'therefore\s+(.+?)[.!?]'
        ]

        justifications = []
        for pattern in justification_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            justifications.extend(matches)

        # Check if justifications are too similar to claims
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        for just in justifications:
            for sent in sentences:
                if just not in sent:  # Don't compare to itself
                    similarity = difflib.SequenceMatcher(None, just.lower(), sent.lower()).ratio()
                    if similarity > 0.8:
                        return f"Potential circular reasoning: justification too similar to claim"

        return ""

    def _analyze_variations(self, responses: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze variations in key information across responses.

        Args:
            responses: List of response texts

        Returns:
            List of detected variations
        """
        variations = []

        # Extract numbers from each response
        numbers_by_response = []
        for response in responses:
            numbers = re.findall(r'\d+(?:\.\d+)?', response)
            numbers_by_response.append(numbers)

        # Check for numerical variations
        if all(numbers_by_response):
            # Compare first number in each (often the most important)
            first_numbers = [nums[0] if nums else None for nums in numbers_by_response]
            unique_numbers = set(n for n in first_numbers if n is not None)

            if len(unique_numbers) > 1:
                variations.append({
                    'type': 'numerical_variation',
                    'values': list(unique_numbers),
                    'description': 'Different numerical values across responses'
                })

        # Extract named entities (simple version - capitalized words)
        entities_by_response = []
        for response in responses:
            entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', response))
            entities_by_response.append(entities)

        # Check for entity variations
        if len(entities_by_response) > 1:
            # Find entities that appear in some but not all responses
            all_entities = set().union(*entities_by_response)
            for entity in all_entities:
                appearances = sum(1 for entities in entities_by_response if entity in entities)
                if 0 < appearances < len(responses):
                    variations.append({
                        'type': 'entity_variation',
                        'entity': entity,
                        'appearances': f'{appearances}/{len(responses)} responses',
                        'description': f'Entity "{entity}" appears inconsistently'
                    })

        return variations

    def _calculate_consistency_score(
        self,
        contradictions: List[Dict],
        logical_issues: List[Dict],
        variations: List[Dict],
        num_responses: int
    ) -> float:
        """Calculate overall consistency score."""
        score = 10.0

        # Penalize contradictions heavily
        score -= len(contradictions) * 2.5

        # Penalize logical issues
        score -= len(logical_issues) * 1.5

        # Penalize variations moderately
        score -= len(variations) * 0.5

        # Normalize by number of responses
        if num_responses > 2:
            penalty_factor = 1 + (num_responses - 2) * 0.1
            score /= penalty_factor

        return max(0, min(10, score))

    def _calculate_avg_similarity(self, responses: List[str]) -> float:
        """Calculate average similarity between all response pairs."""
        if len(responses) < 2:
            return 1.0

        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = difflib.SequenceMatcher(None, responses[i], responses[j]).ratio()
                similarities.append(sim)

        return round(sum(similarities) / len(similarities), 3) if similarities else 0

    def compare_models(
        self,
        prompt: str,
        model_responses: Dict[str, List[str]]
    ) -> Dict[str, DiscrepancyAnalysis]:
        """
        Compare discrepancies across multiple models.

        Args:
            prompt: The input prompt
            model_responses: Dictionary mapping model names to lists of responses

        Returns:
            Dictionary mapping model names to DiscrepancyAnalysis objects
        """
        results = {}
        for model_name, responses in model_responses.items():
            results[model_name] = self.analyze(prompt, responses)
        return results
