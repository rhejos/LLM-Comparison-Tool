"""Hallucination and factual accuracy detection for LLM responses."""

import re
from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class HallucinationAnalysis:
    """Hallucination analysis results."""

    confidence_score: float  # 0-10, higher = more confident/factual
    potential_hallucinations: List[Dict[str, Any]] = field(default_factory=list)
    factual_claims: List[str] = field(default_factory=list)
    uncertainty_markers: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class HallucinationDetector:
    """Detects potential hallucinations and factual errors in LLM responses."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize hallucination detector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)

        # Uncertainty markers (indicate model is being cautious)
        self.uncertainty_markers = [
            'might', 'may', 'could', 'possibly', 'perhaps', 'likely', 'probably',
            'seems', 'appears', 'suggests', 'indicates', 'I think', 'I believe',
            'in my opinion', 'it\'s possible', 'it\'s likely', 'generally',
            'typically', 'usually', 'often', 'sometimes', 'approximately'
        ]

        # Over-confidence markers (red flags)
        self.overconfidence_markers = [
            'definitely', 'certainly', 'absolutely', 'without a doubt',
            'undoubtedly', 'unquestionably', 'always', 'never', 'impossible',
            'guaranteed', 'proven fact', 'scientifically proven', 'everyone knows'
        ]

        # Hallucination patterns
        self.hallucination_patterns = [
            # Fake citations
            r'according to\s+(?:a\s+)?(?:recent\s+)?(?:study|research|report)(?:\s+by\s+[\w\s]+)?(?:\s+in\s+\d{4})?(?:\s+published\s+in\s+[\w\s]+)?(?:\s*,|\s+that)',
            # Specific numbers without source
            r'\d+(?:\.\d+)?%\s+of',
            # False specificity
            r'exactly\s+\d+',
            r'precisely\s+\d+',
            # Invented quotes
            r'(?:he|she|they)\s+said\s+"[^"]+"',
            # Fake experts
            r'experts?\s+(?:say|claim|believe|state)',
        ]

    def analyze(self, prompt: str, response: str) -> HallucinationAnalysis:
        """
        Analyze response for potential hallucinations.

        Args:
            prompt: The input prompt
            response: The LLM response

        Returns:
            HallucinationAnalysis object
        """
        response_lower = response.lower()

        # Detect uncertainty markers
        uncertainty_found = self._find_uncertainty_markers(response_lower)

        # Detect over-confidence
        overconfidence_found = self._find_overconfidence_markers(response_lower)

        # Detect potential hallucinations
        potential_hallucinations = self._detect_hallucination_patterns(response)

        # Extract factual claims
        factual_claims = self._extract_factual_claims(response)

        # Check for unsupported specificity
        unsupported_specifics = self._detect_unsupported_specificity(response)
        potential_hallucinations.extend(unsupported_specifics)

        # Check for inconsistencies within response
        internal_inconsistencies = self._detect_internal_inconsistencies(response)
        potential_hallucinations.extend(internal_inconsistencies)

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            uncertainty_found,
            overconfidence_found,
            potential_hallucinations,
            factual_claims
        )

        details = {
            'num_uncertainty_markers': len(uncertainty_found),
            'num_overconfidence_markers': len(overconfidence_found),
            'num_factual_claims': len(factual_claims),
            'num_potential_hallucinations': len(potential_hallucinations),
            'has_sources': self._has_verifiable_sources(response)
        }

        return HallucinationAnalysis(
            confidence_score=round(confidence_score, 2),
            potential_hallucinations=potential_hallucinations,
            factual_claims=factual_claims,
            uncertainty_markers=uncertainty_found,
            details=details
        )

    def _find_uncertainty_markers(self, text: str) -> List[str]:
        """Find uncertainty markers in text."""
        found = []
        for marker in self.uncertainty_markers:
            if marker in text:
                count = text.count(marker)
                found.extend([marker] * count)
        return found

    def _find_overconfidence_markers(self, text: str) -> List[str]:
        """Find over-confidence markers in text."""
        found = []
        for marker in self.overconfidence_markers:
            if marker in text:
                count = text.count(marker)
                found.extend([marker] * count)
        return found

    def _detect_hallucination_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Detect potential hallucination patterns."""
        hallucinations = []

        for pattern in self.hallucination_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                hallucinations.append({
                    'type': 'suspicious_pattern',
                    'pattern': pattern,
                    'text': match.group(0),
                    'position': match.start(),
                    'severity': 'medium',
                    'reason': 'Potentially unsourced or fabricated claim'
                })

        return hallucinations

    def _extract_factual_claims(self, text: str) -> List[str]:
        """Extract statements that appear to be factual claims."""
        claims = []

        # Look for definitive statements
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        factual_patterns = [
            r'\bis\b.*\bnot\b',
            r'\bis\b.*\ba\b',
            r'\bare\b',
            r'\bwas\b',
            r'\bwere\b',
            r'\bhas\b.*\bbeen\b',
            r'\bhave\b.*\bbeen\b',
            r'\d+',
            r'percent',
            r'percentage'
        ]

        for sentence in sentences:
            if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in factual_patterns):
                # Exclude questions and uncertain statements
                if '?' not in sentence and not any(um in sentence.lower() for um in ['might', 'may', 'could', 'possibly']):
                    claims.append(sentence)

        return claims

    def _detect_unsupported_specificity(self, text: str) -> List[Dict[str, Any]]:
        """Detect overly specific claims without supporting evidence."""
        issues = []

        # Check for very specific numbers
        specific_numbers = re.finditer(r'\b\d+\.\d{2,}\b', text)
        for match in specific_numbers:
            # Check if there's a source nearby
            context_start = max(0, match.start() - 100)
            context_end = min(len(text), match.end() + 100)
            context = text[context_start:context_end].lower()

            source_indicators = ['according to', 'source:', 'from', 'cited', 'reference']
            has_source = any(ind in context for ind in source_indicators)

            if not has_source:
                issues.append({
                    'type': 'unsupported_precision',
                    'value': match.group(0),
                    'severity': 'low',
                    'reason': 'Very specific number without apparent source'
                })

        # Check for specific dates/years
        specific_dates = re.finditer(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b', text)
        for match in specific_dates:
            context_start = max(0, match.start() - 100)
            context_end = min(len(text), match.end() + 100)
            context = text[context_start:context_end].lower()

            # Historical dates are okay, recent specific dates without source are suspicious
            year_match = re.search(r'\d{4}', match.group(0))
            if year_match:
                year = int(year_match.group(0))
                if year > 2010:  # Recent event
                    source_indicators = ['according to', 'source:', 'reported']
                    has_source = any(ind in context for ind in source_indicators)

                    if not has_source:
                        issues.append({
                            'type': 'unsupported_date',
                            'value': match.group(0),
                            'severity': 'low',
                            'reason': 'Specific recent date without source'
                        })

        return issues

    def _detect_internal_inconsistencies(self, text: str) -> List[Dict[str, Any]]:
        """Detect internal inconsistencies that might indicate hallucination."""
        issues = []

        # Extract all numbers and check for contradictions
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

        # Build a simple fact base
        facts = {}
        for sentence in sentences:
            # Look for "X is Y" patterns
            is_patterns = re.finditer(r'(\b\w+\b)\s+is\s+(\w+)', sentence, re.IGNORECASE)
            for match in is_patterns:
                subject = match.group(1).lower()
                predicate = match.group(2).lower()

                if subject in facts and facts[subject] != predicate:
                    issues.append({
                        'type': 'internal_contradiction',
                        'severity': 'high',
                        'detail': f'Conflicting statements about "{subject}": "{facts[subject]}" vs "{predicate}"'
                    })
                else:
                    facts[subject] = predicate

        return issues

    def _has_verifiable_sources(self, text: str) -> bool:
        """Check if response includes verifiable sources."""
        source_indicators = [
            r'https?://',
            r'doi:',
            r'published in',
            r'source:',
            r'reference:',
            r'cited in',
            r'available at'
        ]

        return any(re.search(pattern, text, re.IGNORECASE) for pattern in source_indicators)

    def _calculate_confidence_score(
        self,
        uncertainty_markers: List[str],
        overconfidence_markers: List[str],
        hallucinations: List[Dict],
        factual_claims: List[str]
    ) -> float:
        """
        Calculate confidence/factual accuracy score.

        Higher score = more likely to be factual
        Lower score = more likely to contain hallucinations
        """
        score = 7.0  # Start at neutral-positive

        # Uncertainty markers are good (shows the model is careful)
        if len(factual_claims) > 0:
            uncertainty_ratio = len(uncertainty_markers) / len(factual_claims)
            if uncertainty_ratio > 0.3:  # Healthy amount of uncertainty
                score += 1.5
            elif uncertainty_ratio > 0.1:
                score += 0.5

        # Over-confidence is bad (often precedes hallucinations)
        score -= len(overconfidence_markers) * 0.5

        # Hallucinations are bad
        high_severity = sum(1 for h in hallucinations if h.get('severity') == 'high')
        medium_severity = sum(1 for h in hallucinations if h.get('severity') == 'medium')
        low_severity = sum(1 for h in hallucinations if h.get('severity') == 'low')

        score -= high_severity * 2.0
        score -= medium_severity * 1.0
        score -= low_severity * 0.3

        return max(0, min(10, score))

    def compare_models(
        self,
        model_responses: Dict[str, tuple]
    ) -> Dict[str, HallucinationAnalysis]:
        """
        Compare hallucination tendencies across multiple models.

        Args:
            model_responses: Dictionary mapping model names to (prompt, response) tuples

        Returns:
            Dictionary mapping model names to HallucinationAnalysis objects
        """
        results = {}
        for model_name, (prompt, response) in model_responses.items():
            results[model_name] = self.analyze(prompt, response)
        return results
