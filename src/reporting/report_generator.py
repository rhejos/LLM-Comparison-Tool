"""Report generation for LLM comparison results."""

from typing import Dict, List, Any
from datetime import datetime
import json
from pathlib import Path


class ReportGenerator:
    """Generates reports from comparison results."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize report generator.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.output_formats = self.config.get('output_formats', ['markdown', 'json'])
        self.include_visualizations = self.config.get('include_visualizations', True)
        self.detailed_analysis = self.config.get('detailed_analysis', True)

    def generate_markdown_report(self, result: Any, output_path: str = None) -> str:
        """
        Generate a Markdown report.

        Args:
            result: ComparisonResult object
            output_path: Optional path to save the report

        Returns:
            Markdown content as string
        """
        md = []

        # Header
        md.append("# LLM Comparison Report")
        md.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Prompt
        md.append("## Prompt")
        md.append(f"\n```\n{result.prompt}\n```\n")

        # Summary
        md.append("## Summary")
        if result.summary:
            md.append(f"\n- **Overall Winner:** {result.summary.get('winner', 'N/A')}")
            md.append(f"- **Best Quality:** {result.summary.get('best_quality', 'N/A')}")
            md.append(f"- **Least Biased:** {result.summary.get('least_biased', 'N/A')}")
            md.append(f"- **Fastest:** {result.summary.get('fastest', 'N/A')}")
            md.append(f"- **Most Factual:** {result.summary.get('most_factual', 'N/A')}")
            if 'score_margin' in result.summary:
                md.append(f"- **Score Margin:** {result.summary['score_margin']}")
            md.append("")

        # Rankings Table
        md.append("## Overall Rankings\n")
        if result.rankings.get('overall'):
            md.append("| Rank | Model | Score |")
            md.append("|------|-------|-------|")
            for i, (model, score) in enumerate(result.rankings['overall'], 1):
                md.append(f"| {i} | {model} | {score:.2f}/10 |")
            md.append("")

        # Detailed Metrics
        md.append("## Detailed Metrics\n")

        # Quality Metrics
        md.append("### Quality Scores\n")
        md.append("| Model | Overall | Coherence | Relevance | Completeness | Clarity |")
        md.append("|-------|---------|-----------|-----------|--------------|---------|")
        for model, data in result.model_results.items():
            if 'quality' in data:
                q = data['quality']
                md.append(f"| {model} | {q['overall']:.1f} | {q['coherence']:.1f} | {q['relevance']:.1f} | {q['completeness']:.1f} | {q['clarity']:.1f} |")
        md.append("")

        # Performance Metrics
        md.append("### Performance Metrics\n")
        md.append("| Model | Avg Time (s) | Tokens/sec | Efficiency |")
        md.append("|-------|--------------|------------|------------|")
        for model, data in result.model_results.items():
            if 'performance' in data:
                p = data['performance']
                md.append(f"| {model} | {p['avg_response_time']:.2f} | {p['tokens_per_second']:.1f} | {p['efficiency_score']:.1f}/10 |")
        md.append("")

        # Bias Analysis
        md.append("### Bias Analysis\n")
        md.append("| Model | Overall Score | Gender | Race | Political | Age | Religion | Socioeconomic |")
        md.append("|-------|---------------|--------|------|-----------|-----|----------|---------------|")
        for model, data in result.model_results.items():
            if 'bias' in data:
                b = data['bias']
                cats = b['categories']
                md.append(
                    f"| {model} | {b['overall_score']:.1f}/10 | "
                    f"{cats.get('gender', 'N/A'):.1f} | "
                    f"{cats.get('race', 'N/A'):.1f} | "
                    f"{cats.get('political', 'N/A'):.1f} | "
                    f"{cats.get('age', 'N/A'):.1f} | "
                    f"{cats.get('religion', 'N/A'):.1f} | "
                    f"{cats.get('socioeconomic', 'N/A'):.1f} |"
                )
        md.append("")

        # Topic Suitability
        md.append("### Topic Suitability\n")
        for model, data in result.model_results.items():
            if 'topic' in data:
                md.append(f"\n**{model}:**")
                md.append(f"- Best Topics: {', '.join(data['topic']['best_topics'])}")
                md.append(f"- Recommendations:")
                for rec in data['topic']['recommendations']:
                    md.append(f"  - {rec}")

        md.append("")

        # Hallucination Analysis
        md.append("### Factual Accuracy (Hallucination Detection)\n")
        md.append("| Model | Confidence Score | Potential Issues | Uncertainty Markers |")
        md.append("|-------|------------------|------------------|---------------------|")
        for model, data in result.model_results.items():
            if 'hallucination' in data:
                h = data['hallucination']
                md.append(
                    f"| {model} | {h['confidence_score']:.1f}/10 | "
                    f"{len(h.get('potential_hallucinations', []))} | "
                    f"{len(h.get('uncertainty_markers', []))} |"
                )
        md.append("")

        # Detailed Responses
        if self.detailed_analysis:
            md.append("## Detailed Responses\n")
            for model, data in result.model_results.items():
                md.append(f"### {model}\n")
                md.append(f"```\n{data.get('primary_response', 'No response available')}\n```\n")

                # Show bias issues if any
                if 'bias' in data and data['bias'].get('detected_biases'):
                    md.append("**Detected Biases:**")
                    for bias in data['bias']['detected_biases'][:5]:  # Show top 5
                        md.append(f"- {bias.get('type', 'Unknown')}: {bias.get('details', bias.get('phrase', 'N/A'))}")
                    md.append("")

                # Show hallucination issues if any
                if 'hallucination' in data and data['hallucination'].get('potential_hallucinations'):
                    md.append("**Potential Hallucinations:**")
                    for hall in data['hallucination']['potential_hallucinations'][:5]:
                        md.append(f"- {hall.get('type', 'Unknown')}: {hall.get('reason', 'N/A')}")
                    md.append("")

        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write('\n'.join(md))

        return '\n'.join(md)

    def generate_json_report(self, result: Any, output_path: str = None) -> str:
        """
        Generate a JSON report.

        Args:
            result: ComparisonResult object
            output_path: Optional path to save the report

        Returns:
            JSON content as string
        """
        json_data = result.to_dict()

        # Add metadata
        json_data['report_metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'format_version': '1.0'
        }

        json_str = json.dumps(json_data, indent=2)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(json_str)

        return json_str

    def generate_html_report(self, result: Any, output_path: str = None) -> str:
        """
        Generate an HTML report.

        Args:
            result: ComparisonResult object
            output_path: Optional path to save the report

        Returns:
            HTML content as string
        """
        html = []

        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("  <meta charset='UTF-8'>")
        html.append("  <title>LLM Comparison Report</title>")
        html.append("  <style>")
        html.append(self._get_html_css())
        html.append("  </style>")
        html.append("</head>")
        html.append("<body>")

        html.append("  <div class='container'>")
        html.append("    <h1>LLM Comparison Report</h1>")
        html.append(f"    <p class='timestamp'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")

        # Prompt
        html.append("    <div class='section'>")
        html.append("      <h2>Prompt</h2>")
        html.append(f"      <div class='prompt'>{self._escape_html(result.prompt)}</div>")
        html.append("    </div>")

        # Summary
        if result.summary:
            html.append("    <div class='section'>")
            html.append("      <h2>Summary</h2>")
            html.append("      <div class='summary'>")
            html.append(f"        <div class='summary-item'><strong>Overall Winner:</strong> {result.summary.get('winner', 'N/A')}</div>")
            html.append(f"        <div class='summary-item'><strong>Best Quality:</strong> {result.summary.get('best_quality', 'N/A')}</div>")
            html.append(f"        <div class='summary-item'><strong>Least Biased:</strong> {result.summary.get('least_biased', 'N/A')}</div>")
            html.append(f"        <div class='summary-item'><strong>Fastest:</strong> {result.summary.get('fastest', 'N/A')}</div>")
            html.append(f"        <div class='summary-item'><strong>Most Factual:</strong> {result.summary.get('most_factual', 'N/A')}</div>")
            html.append("      </div>")
            html.append("    </div>")

        # Rankings
        if result.rankings.get('overall'):
            html.append("    <div class='section'>")
            html.append("      <h2>Overall Rankings</h2>")
            html.append("      <table>")
            html.append("        <tr><th>Rank</th><th>Model</th><th>Score</th></tr>")
            for i, (model, score) in enumerate(result.rankings['overall'], 1):
                html.append(f"        <tr><td>{i}</td><td>{model}</td><td>{score:.2f}/10</td></tr>")
            html.append("      </table>")
            html.append("    </div>")

        # Detailed metrics
        html.append("    <div class='section'>")
        html.append("      <h2>Detailed Metrics</h2>")

        # Quality
        html.append("      <h3>Quality Scores</h3>")
        html.append("      <table>")
        html.append("        <tr><th>Model</th><th>Overall</th><th>Coherence</th><th>Relevance</th><th>Completeness</th><th>Clarity</th></tr>")
        for model, data in result.model_results.items():
            if 'quality' in data:
                q = data['quality']
                html.append(f"        <tr><td>{model}</td><td>{q['overall']:.1f}</td><td>{q['coherence']:.1f}</td><td>{q['relevance']:.1f}</td><td>{q['completeness']:.1f}</td><td>{q['clarity']:.1f}</td></tr>")
        html.append("      </table>")

        html.append("    </div>")
        html.append("  </div>")
        html.append("</body>")
        html.append("</html>")

        html_str = '\n'.join(html)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(html_str)

        return html_str

    def _get_html_css(self) -> str:
        """Get CSS for HTML reports."""
        return """
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; border-bottom: 2px solid #e0e0e0; padding-bottom: 5px; }
        h3 { color: #666; margin-top: 20px; }
        .timestamp { color: #888; font-size: 0.9em; }
        .section { margin-bottom: 30px; }
        .prompt { background: #f9f9f9; padding: 15px; border-left: 4px solid #4CAF50; font-family: monospace; white-space: pre-wrap; }
        .summary { background: #f0f8ff; padding: 15px; border-radius: 5px; }
        .summary-item { margin: 8px 0; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th { background: #4CAF50; color: white; padding: 12px; text-align: left; }
        td { padding: 10px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f5f5f5; }
        """

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

    def generate_reports(self, result: Any, output_dir: str, base_name: str = "report"):
        """
        Generate all configured report formats.

        Args:
            result: ComparisonResult object
            output_dir: Directory to save reports
            base_name: Base name for report files
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        reports_generated = []

        if 'markdown' in self.output_formats:
            md_path = f"{output_dir}/{base_name}.md"
            self.generate_markdown_report(result, md_path)
            reports_generated.append(md_path)

        if 'json' in self.output_formats:
            json_path = f"{output_dir}/{base_name}.json"
            self.generate_json_report(result, json_path)
            reports_generated.append(json_path)

        if 'html' in self.output_formats:
            html_path = f"{output_dir}/{base_name}.html"
            self.generate_html_report(result, html_path)
            reports_generated.append(html_path)

        return reports_generated
