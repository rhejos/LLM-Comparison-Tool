# LLM Comparison Tool 

A comprehensive tool for comparing different Large Language Models (LLMs) with detailed metrics, bias detection, and discrepancy analysis.

## Features

### 🎯 Multi-Model Support
- **Claude** (Anthropic)
- **ChatGPT** (OpenAI)
- **Perplexity AI**
- **Google Gemini**
- **Other OpenAI-compatible APIs**

### 📊 Comprehensive Metrics
- **Response Quality**: Coherence, relevance, completeness
- **Consistency**: Response stability across similar prompts
- **Speed**: Response time and throughput
- **Token Usage**: Efficiency analysis
- **Factual Accuracy**: Verification against known facts

### 🔍 Advanced Analysis
- **Bias Detection**: Identifies gender, racial, political, and other biases
- **Discrepancy Analysis**: Finds inconsistencies in responses
- **Topic Suitability**: Determines best use cases for each model
- **Hallucination Detection**: Identifies factual errors and fabrications
- **Sentiment Analysis**: Measures tone and emotional content

### 📈 Reporting
- Detailed comparison reports (Markdown, JSON, HTML)
- Visual charts and graphs
- Side-by-side response comparisons
- Statistical summaries

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/LLM-as-a-judge.git
cd LLM-as-a-judge

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp config/config.example.json config/config.json
# Edit config/config.json with your API keys
```

### Basic Usage

```bash
# Run a simple comparison
python src/main.py --prompt "Explain quantum computing" --models claude,gpt4,perplexity

# Run with all analysis features
python src/main.py --prompt "Explain quantum computing" --models claude,gpt4 --analyze-bias --detect-discrepancies

# Run a test suite
python src/main.py --test-suite examples/test_suite.json --output-dir outputs/

# Compare on specific topics
python src/main.py --topic-analysis --test-suite examples/topic_tests.json
```

## Configuration

Edit `config/config.json` to add your API keys:

```json
{
  "api_keys": {
    "anthropic": "your-claude-api-key",
    "openai": "your-openai-api-key",
    "perplexity": "your-perplexity-api-key",
    "google": "your-google-api-key"
  },
  "models": {
    "claude": "claude-3-5-sonnet-20241022",
    "gpt4": "gpt-4-turbo-preview",
    "perplexity": "llama-3.1-sonar-large-128k-online"
  }
}
```

## Project Structure

```
LLM-as-a-judge/
├── src/
│   ├── main.py                 # Main CLI entry point
│   ├── core/
│   │   ├── comparison_engine.py    # Core comparison logic
│   │   └── prompt_manager.py       # Prompt handling
│   ├── providers/
│   │   ├── base_provider.py        # Base LLM provider interface
│   │   ├── claude_provider.py      # Claude integration
│   │   ├── openai_provider.py      # ChatGPT integration
│   │   ├── perplexity_provider.py  # Perplexity integration
│   │   └── gemini_provider.py      # Google Gemini integration
│   ├── metrics/
│   │   ├── quality_metrics.py      # Response quality evaluation
│   │   ├── consistency_metrics.py  # Consistency analysis
│   │   └── performance_metrics.py  # Speed and efficiency
│   ├── analyzers/
│   │   ├── bias_detector.py        # Bias detection
│   │   ├── discrepancy_analyzer.py # Discrepancy detection
│   │   ├── topic_analyzer.py       # Topic suitability
│   │   └── hallucination_detector.py # Factual accuracy
│   └── reporting/
│       ├── report_generator.py     # Report creation
│       └── visualizer.py           # Charts and graphs
├── config/
│   ├── config.json             # Configuration file (user-created)
│   └── config.example.json     # Example configuration
├── examples/
│   ├── test_suite.json         # Example test cases
│   └── topic_tests.json        # Topic-specific tests
├── outputs/                    # Generated reports (created on run)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Metrics Explained

### Bias Detection
The tool analyzes responses for:
- Gender bias
- Racial/ethnic bias
- Political bias
- Age bias
- Religious bias
- Socioeconomic bias

### Discrepancy Analysis
Identifies:
- Factual contradictions
- Logical inconsistencies
- Varying responses to similar prompts
- Statistical anomalies

### Topic Suitability
Evaluates each model's performance on:
- Creative writing
- Technical explanations
- Factual research
- Code generation
- Mathematical reasoning
- Conversational tasks
- Analytical tasks

## Example Output

```markdown
# LLM Comparison Report
Generated: 2024-01-15 14:30:00

## Summary
- Prompt: "Explain quantum computing"
- Models Compared: Claude, GPT-4, Perplexity
- Total Tests: 5 variations

## Results

### Overall Scores
| Model | Quality | Bias | Consistency | Speed |
|-------|---------|------|-------------|-------|
| Claude | 9.2/10 | 8.5/10 | 9.0/10 | 2.3s |
| GPT-4 | 8.8/10 | 8.2/10 | 8.5/10 | 3.1s |
| Perplexity | 8.5/10 | 9.0/10 | 8.0/10 | 1.8s |

### Bias Analysis
- Claude: Minimal bias detected
- GPT-4: Slight technical bias
- Perplexity: Most neutral

### Best Use Cases
- Claude: Technical explanations, creative writing
- GPT-4: General knowledge, code generation
- Perplexity: Research, fact-checking
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Support

For issues and questions, please open an issue on GitHub.
