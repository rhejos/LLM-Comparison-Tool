#!/usr/bin/env python3
"""Main CLI entry point for LLM Comparison Tool."""

import asyncio
import click
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.comparison_engine import ComparisonEngine
from core.prompt_manager import PromptManager
from providers import ClaudeProvider, OpenAIProvider, PerplexityProvider, GeminiProvider
from reporting import ReportGenerator


console = Console()


def load_config(config_path: str = "config/config.json") -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        console.print(f"[red]Error: Config file not found at {config_path}[/red]")
        console.print("[yellow]Please copy config/config.example.json to config/config.json and add your API keys.[/yellow]")
        sys.exit(1)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON in config file: {e}[/red]")
        sys.exit(1)


def initialize_providers(config: dict, selected_models: list = None) -> dict:
    """Initialize LLM providers based on config."""
    providers = {}
    api_keys = config.get('api_keys', {})
    models_config = config.get('models', {})

    # Determine which models to initialize
    if selected_models:
        models_to_init = [m for m in selected_models if m in models_config]
    else:
        models_to_init = list(models_config.keys())

    for model_name in models_to_init:
        model_config = models_config[model_name]

        try:
            # Determine provider type from model_id
            model_id = model_config.get('model_id', '')

            if 'claude' in model_id.lower():
                if api_keys.get('anthropic'):
                    providers[model_name] = ClaudeProvider(api_keys['anthropic'], model_config)
                    console.print(f"[green]✓[/green] Initialized {model_name}")
                else:
                    console.print(f"[yellow]⚠[/yellow] Skipping {model_name}: No Anthropic API key")

            elif 'gpt' in model_id.lower():
                if api_keys.get('openai'):
                    providers[model_name] = OpenAIProvider(api_keys['openai'], model_config)
                    console.print(f"[green]✓[/green] Initialized {model_name}")
                else:
                    console.print(f"[yellow]⚠[/yellow] Skipping {model_name}: No OpenAI API key")

            elif 'sonar' in model_id.lower() or 'perplexity' in model_name.lower():
                if api_keys.get('perplexity'):
                    providers[model_name] = PerplexityProvider(api_keys['perplexity'], model_config)
                    console.print(f"[green]✓[/green] Initialized {model_name}")
                else:
                    console.print(f"[yellow]⚠[/yellow] Skipping {model_name}: No Perplexity API key")

            elif 'gemini' in model_id.lower():
                if api_keys.get('google'):
                    providers[model_name] = GeminiProvider(api_keys['google'], model_config)
                    console.print(f"[green]✓[/green] Initialized {model_name}")
                else:
                    console.print(f"[yellow]⚠[/yellow] Skipping {model_name}: No Google API key")

        except Exception as e:
            console.print(f"[red]✗[/red] Failed to initialize {model_name}: {e}")

    return providers


@click.group()
def cli():
    """LLM Comparison Tool - Compare different LLMs with detailed metrics and analysis."""
    pass


@cli.command()
@click.option('--prompt', '-p', help='Single prompt to test')
@click.option('--models', '-m', help='Comma-separated list of models to compare')
@click.option('--config', '-c', default='config/config.json', help='Path to config file')
@click.option('--output-dir', '-o', default='outputs', help='Output directory for reports')
@click.option('--analyze-bias/--no-analyze-bias', default=True, help='Perform bias analysis')
@click.option('--analyze-discrepancies/--no-analyze-discrepancies', default=True, help='Perform discrepancy analysis')
@click.option('--analyze-topics/--no-analyze-topics', default=True, help='Perform topic analysis')
@click.option('--analyze-hallucinations/--no-analyze-hallucinations', default=True, help='Perform hallucination analysis')
@click.option('--consistency-runs', '-r', default=1, type=int, help='Number of consistency test runs')
def compare(prompt, models, config, output_dir, analyze_bias, analyze_discrepancies,
           analyze_topics, analyze_hallucinations, consistency_runs):
    """Compare LLMs on a single prompt."""
    console.print("\n[bold blue]LLM Comparison Tool[/bold blue]\n")

    # Load configuration
    config_data = load_config(config)

    # Parse selected models
    selected_models = models.split(',') if models else None

    # Initialize providers
    console.print("\n[bold]Initializing providers...[/bold]")
    providers = initialize_providers(config_data, selected_models)

    if not providers:
        console.print("[red]Error: No providers initialized. Check your API keys.[/red]")
        sys.exit(1)

    # Initialize comparison engine
    engine = ComparisonEngine(config_data)
    for name, provider in providers.items():
        engine.register_provider(name, provider)

    # Run comparison
    console.print(f"\n[bold]Running comparison...[/bold]")
    console.print(f"Prompt: {prompt[:100]}...")
    console.print(f"Models: {', '.join(providers.keys())}")
    console.print(f"Consistency runs: {consistency_runs}\n")

    result = asyncio.run(engine.compare_single_prompt(
        prompt=prompt,
        model_names=list(providers.keys()),
        analyze_bias=analyze_bias,
        analyze_discrepancies=analyze_discrepancies,
        analyze_topics=analyze_topics,
        analyze_hallucinations=analyze_hallucinations,
        consistency_runs=consistency_runs
    ))

    # Display summary
    console.print("\n[bold green]Comparison Complete![/bold green]\n")

    # Rankings table
    if result.rankings.get('overall'):
        table = Table(title="Overall Rankings")
        table.add_column("Rank", style="cyan", justify="center")
        table.add_column("Model", style="magenta")
        table.add_column("Score", style="green", justify="right")

        for i, (model, score) in enumerate(result.rankings['overall'], 1):
            table.add_row(str(i), model, f"{score:.2f}/10")

        console.print(table)

    # Generate reports
    console.print(f"\n[bold]Generating reports...[/bold]")
    report_gen = ReportGenerator(config_data.get('reporting', {}))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    reports = report_gen.generate_reports(result, output_dir, "comparison_report")

    console.print(f"\n[green]Reports generated:[/green]")
    for report_path in reports:
        console.print(f"  - {report_path}")

    console.print(f"\n[bold blue]Done![/bold blue] 🎉\n")


@cli.command()
@click.option('--test-suite', '-t', required=True, help='Path to test suite JSON file')
@click.option('--models', '-m', help='Comma-separated list of models to compare')
@click.option('--config', '-c', default='config/config.json', help='Path to config file')
@click.option('--output-dir', '-o', default='outputs', help='Output directory for reports')
@click.option('--consistency-runs', '-r', default=1, type=int, help='Number of consistency test runs')
def batch(test_suite, models, config, output_dir, consistency_runs):
    """Run comparisons on a batch of test cases."""
    console.print("\n[bold blue]LLM Comparison Tool - Batch Mode[/bold blue]\n")

    # Load configuration
    config_data = load_config(config)

    # Load test suite
    prompt_manager = PromptManager()
    try:
        prompt_manager.load_from_file(test_suite)
        console.print(f"[green]✓[/green] Loaded {len(prompt_manager)} test cases")
    except Exception as e:
        console.print(f"[red]Error loading test suite: {e}[/red]")
        sys.exit(1)

    # Parse selected models
    selected_models = models.split(',') if models else None

    # Initialize providers
    console.print("\n[bold]Initializing providers...[/bold]")
    providers = initialize_providers(config_data, selected_models)

    if not providers:
        console.print("[red]Error: No providers initialized. Check your API keys.[/red]")
        sys.exit(1)

    # Initialize comparison engine
    engine = ComparisonEngine(config_data)
    for name, provider in providers.items():
        engine.register_provider(name, provider)

    # Run comparisons
    console.print(f"\n[bold]Running {len(prompt_manager)} test cases...[/bold]\n")

    prompts = [tc.prompt for tc in prompt_manager.get_all_test_cases()]

    results = asyncio.run(engine.compare_multiple_prompts(
        prompts=prompts,
        model_names=list(providers.keys()),
        consistency_runs=consistency_runs
    ))

    # Generate reports for each result
    console.print(f"\n[bold]Generating reports...[/bold]")
    report_gen = ReportGenerator(config_data.get('reporting', {}))

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(results):
        base_name = f"report_{i+1}"
        reports = report_gen.generate_reports(result, output_dir, base_name)

    console.print(f"\n[green]Generated {len(results)} report sets in {output_dir}[/green]")
    console.print(f"\n[bold blue]Done![/bold blue] 🎉\n")


@cli.command()
@click.option('--config', '-c', default='config/config.json', help='Path to config file')
def list_models(config):
    """List all configured models."""
    config_data = load_config(config)

    table = Table(title="Configured Models")
    table.add_column("Name", style="cyan")
    table.add_column("Model ID", style="magenta")
    table.add_column("Display Name", style="green")
    table.add_column("Max Tokens", style="yellow", justify="right")

    for name, model_config in config_data.get('models', {}).items():
        table.add_row(
            name,
            model_config.get('model_id', 'N/A'),
            model_config.get('display_name', 'N/A'),
            str(model_config.get('max_tokens', 'N/A'))
        )

    console.print(table)


if __name__ == '__main__':
    cli()
