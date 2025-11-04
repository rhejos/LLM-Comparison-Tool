"""Prompt management for LLM comparisons."""

from typing import List, Dict, Any
from dataclasses import dataclass
import json


@dataclass
class TestCase:
    """A test case for LLM comparison."""

    prompt: str
    category: str = "general"
    expected_characteristics: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.expected_characteristics is None:
            self.expected_characteristics = []
        if self.metadata is None:
            self.metadata = {}


class PromptManager:
    """Manages prompts and test cases for LLM comparison."""

    def __init__(self):
        """Initialize the prompt manager."""
        self.test_cases: List[TestCase] = []

    def add_test_case(
        self,
        prompt: str,
        category: str = "general",
        expected_characteristics: List[str] = None,
        metadata: Dict[str, Any] = None
    ):
        """
        Add a test case.

        Args:
            prompt: The prompt text
            category: Category of the test
            expected_characteristics: Expected characteristics of good responses
            metadata: Additional metadata
        """
        test_case = TestCase(
            prompt=prompt,
            category=category,
            expected_characteristics=expected_characteristics or [],
            metadata=metadata or {}
        )
        self.test_cases.append(test_case)

    def load_from_file(self, filepath: str):
        """
        Load test cases from a JSON file.

        Args:
            filepath: Path to the JSON file

        Expected format:
        {
            "test_cases": [
                {
                    "prompt": "...",
                    "category": "...",
                    "expected_characteristics": [...],
                    "metadata": {...}
                },
                ...
            ]
        }
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        for case_data in data.get('test_cases', []):
            self.add_test_case(
                prompt=case_data['prompt'],
                category=case_data.get('category', 'general'),
                expected_characteristics=case_data.get('expected_characteristics'),
                metadata=case_data.get('metadata')
            )

    def get_test_cases_by_category(self, category: str) -> List[TestCase]:
        """Get all test cases for a specific category."""
        return [tc for tc in self.test_cases if tc.category == category]

    def get_all_categories(self) -> List[str]:
        """Get all unique categories."""
        return list(set(tc.category for tc in self.test_cases))

    def create_variations(self, base_prompt: str, num_variations: int = 3) -> List[str]:
        """
        Create variations of a prompt for consistency testing.

        Args:
            base_prompt: The base prompt
            num_variations: Number of variations to create

        Returns:
            List of prompt variations (including the original)
        """
        variations = [base_prompt]

        # Simple variations - in practice, you might want more sophisticated methods
        variation_templates = [
            "{prompt}",
            "Please {prompt}",
            "{prompt} Can you help with this?",
            "I need you to {prompt}",
            "{prompt} Please be detailed.",
        ]

        for i in range(min(num_variations, len(variation_templates) - 1)):
            template = variation_templates[i + 1]
            # Make first letter lowercase for template insertion
            prompt_lower = base_prompt[0].lower() + base_prompt[1:] if base_prompt else base_prompt
            variation = template.format(prompt=prompt_lower)
            variations.append(variation)

        return variations[:num_variations + 1]

    def get_all_test_cases(self) -> List[TestCase]:
        """Get all test cases."""
        return self.test_cases

    def clear(self):
        """Clear all test cases."""
        self.test_cases = []

    def __len__(self):
        """Return the number of test cases."""
        return len(self.test_cases)
