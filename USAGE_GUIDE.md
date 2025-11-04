# LLM Comparison Tool - Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Understanding the Metrics](#understanding-the-metrics)
6. [Interpreting Results](#interpreting-results)
7. [Best Practices](#best-practices)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/LLM-as-a-judge.git
cd LLM-as-a-judge
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download spaCy language model (for advanced text analysis):**
```bash
python -m spacy download en_core_web_sm
```

5. **Set up configuration:**
```bash
cp config/config.example.json config/config.json
```

6. **Edit `config/config.json` with your API keys:**
```json
{
  "api_keys": {
    "anthropic": "your-claude-api-key",
    "openai": "your-openai-api-key",
    "perplexity": "your-perplexity-api-key",
    "google": "your-google-api-key"
  }
}
```

## Configuration

### API Keys

Get API keys from:
- **Claude (Anthropic):** https://console.anthropic.com/
- **OpenAI (ChatGPT):** https://platform.openai.com/api-keys
- **Perplexity:** https://www.perplexity.ai/settings/api
- **Google (Gemini):** https://makersuite.google.com/app/apikey

### Model Configuration

In `config/config.json`, you can configure which models to use:

```json
{
  "models": {
    "claude": {
      "model_id": "claude-3-5-sonnet-20241022",
      "max_tokens": 4096,
      "temperature": 1.0,
      "display_name": "Claude 3.5 Sonnet"
    },
    "gpt4": {
      "model_id": "gpt-4-turbo-preview",
      "max_tokens": 4096,
      "temperature": 1.0,
      "display_name": "GPT-4 Turbo"
    }
  }
}
```

### Analysis Configuration

Enable/disable specific analyses:

```json
{
  "analysis": {
    "bias_detection": {
      "enabled": true,
      "categories": ["gender", "race", "political", "age", "religion", "socioeconomic"]
    },
    "discrepancy_detection": {
      "enabled": true,
      "similarity_threshold": 0.85
    },
    "topic_analysis": {
      "enabled": true
    },
    "hallucination_detection": {
      "enabled": true,
      "confidence_threshold": 0.7
    }
  }
}
```

## Basic Usage

### 1. Compare Models on a Single Prompt

```bash
python src/main.py compare \
  --prompt "Explain quantum computing in simple terms" \
  --models "claude,gpt4" \
  --output-dir outputs/
```

### 2. Run a Test Suite

```bash
python src/main.py batch \
  --test-suite examples/test_suite.json \
  --models "claude,gpt4,perplexity" \
  --output-dir outputs/batch_results/
```

### 3. List Available Models

```bash
python src/main.py list-models
```

## Advanced Features

### Consistency Testing

Run multiple iterations to test response consistency:

```bash
python src/main.py compare \
  --prompt "What is artificial intelligence?" \
  --models "claude,gpt4" \
  --consistency-runs 3 \
  --output-dir outputs/
```

### Selective Analysis

Enable/disable specific analyses:

```bash
python src/main.py compare \
  --prompt "Discuss political ideologies" \
  --models "claude,gpt4" \
  --analyze-bias \
  --no-analyze-hallucinations \
  --output-dir outputs/
```

### Custom Test Suites

Create your own test suite JSON:

```json
{
  "test_cases": [
    {
      "prompt": "Your custom prompt here",
      "category": "technical_explanation",
      "expected_characteristics": ["accuracy", "clarity"],
      "metadata": {
        "difficulty": "hard",
        "topic": "your_topic"
      }
    }
  ]
}
```

Then run:

```bash
python src/main.py batch --test-suite my_tests.json --output-dir outputs/
```

## Understanding the Metrics

### Quality Metrics (0-10 scale)

- **Coherence:** Logical flow and internal consistency
- **Relevance:** How well the response addresses the prompt
- **Completeness:** Thoroughness and depth of the response
- **Clarity:** How easy the response is to understand
- **Overall:** Weighted combination of all quality metrics

### Bias Detection (0-10 scale, higher = less biased)

Detects bias in six categories:
- **Gender:** Imbalanced gender representation or stereotypes
- **Race/Ethnicity:** Racial stereotypes or generalizations
- **Political:** Political leaning or polarizing language
- **Age:** Age-related stereotypes
- **Religion:** Religious bias or stereotypes
- **Socioeconomic:** Class-based assumptions

### Performance Metrics

- **Response Time:** How quickly the model responds (seconds)
- **Tokens/Second:** Generation speed
- **Efficiency Score:** Combined measure of speed and consistency

### Hallucination Detection (0-10 scale, higher = more factual)

- **Confidence Score:** Likelihood of factual accuracy
- **Uncertainty Markers:** Instances of cautious language (good)
- **Potential Hallucinations:** Suspicious patterns or unsourced claims

### Discrepancy Analysis (0-10 scale, higher = more consistent)

- **Consistency Score:** Overall response stability
- **Contradictions:** Factual conflicts between responses
- **Logical Issues:** Internal inconsistencies
- **Variations:** Differences in key information

### Topic Suitability

Analyzes performance across categories:
- Creative writing
- Technical explanation
- Factual research
- Code generation
- Mathematical reasoning
- Conversational
- Analytical

## Interpreting Results

### Overall Rankings

The tool ranks models across multiple dimensions:

```
Overall Rankings:
1. Claude 3.5 Sonnet - 8.7/10
2. GPT-4 Turbo - 8.4/10
3. Perplexity - 8.1/10
```

### Reading the Reports

#### Markdown Reports (`.md`)
- Human-readable format
- Contains all metrics and detailed responses
- Good for sharing and documentation

#### JSON Reports (`.json`)
- Machine-readable format
- Contains all raw data
- Good for further processing or analysis

#### HTML Reports (`.html`)
- Interactive, styled reports
- Easy to view in a browser
- Good for presentations

### Key Indicators

**High Quality Response:**
- Quality score > 8.0
- Bias score > 8.0
- Hallucination confidence > 7.0
- Few or no detected issues

**Potential Issues:**
- Quality score < 6.0
- Bias score < 6.0
- Multiple hallucination warnings
- Internal contradictions

## Best Practices

### 1. Choose Appropriate Test Cases

- **Technical questions** → Test accuracy and clarity
- **Creative prompts** → Test creativity and coherence
- **Factual queries** → Test accuracy and hallucination resistance
- **Controversial topics** → Test bias and balance

### 2. Use Consistency Testing

Run multiple iterations (3-5) for important evaluations:

```bash
--consistency-runs 5
```

### 3. Compare Apples to Apples

- Use the same temperature settings
- Compare models of similar capability levels
- Test on diverse prompt types

### 4. Interpret Scores Contextually

- Different tasks favor different strengths
- A "lower" score doesn't always mean "worse"
- Consider the specific use case

### 5. Review Detailed Output

Don't just look at scores:
- Read the actual responses
- Check the detected bias/hallucination examples
- Look for patterns across multiple tests

### 6. Combine Multiple Metrics

No single metric tells the whole story:
- High quality + high bias = potentially problematic
- High speed + low accuracy = not always better
- Consider the overall profile

### 7. Update Your Test Suite

Regularly add new test cases:
- Real-world prompts you use
- Edge cases and challenging scenarios
- Domain-specific queries

### 8. Track Changes Over Time

Save your reports and compare:
- Model updates and improvements
- Different configuration settings
- Various prompt formulations

## Troubleshooting

### API Key Issues

**Error:** "No providers initialized"
- Check that API keys are correctly set in `config/config.json`
- Ensure keys are valid and have sufficient credits

### Rate Limiting

**Error:** "Rate limit exceeded"
- Reduce `--consistency-runs`
- Add delays between tests
- Check API rate limits for your tier

### Installation Issues

**Error:** "Module not found"
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version (3.8+)

### Memory Issues

**Error:** "Out of memory"
- Reduce batch size
- Run tests sequentially instead of in parallel
- Close other applications

## Examples

### Example 1: Testing for Bias

```bash
python src/main.py compare \
  --prompt "Discuss leadership qualities in business" \
  --models "claude,gpt4" \
  --analyze-bias \
  --consistency-runs 3 \
  --output-dir outputs/bias_test/
```

### Example 2: Code Generation Comparison

```bash
python src/main.py compare \
  --prompt "Write a Python function to merge two sorted lists" \
  --models "claude,gpt4,gpt35" \
  --analyze-topics \
  --output-dir outputs/code_test/
```

### Example 3: Factual Accuracy Test

```bash
python src/main.py compare \
  --prompt "What are the main causes of World War II?" \
  --models "claude,gpt4,perplexity" \
  --analyze-hallucinations \
  --consistency-runs 2 \
  --output-dir outputs/factual_test/
```

### Example 4: Comprehensive Topic Analysis

```bash
python src/main.py batch \
  --test-suite examples/topic_tests.json \
  --models "claude,gpt4" \
  --analyze-topics \
  --analyze-bias \
  --output-dir outputs/comprehensive/
```

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/yourusername/LLM-as-a-judge/issues
- Documentation: See README.md

## Next Steps

- Experiment with different prompts
- Create custom test suites for your use case
- Compare models on domain-specific tasks
- Track model performance over time
- Share your findings with the community
